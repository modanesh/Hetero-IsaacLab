# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import omni.usd
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.direct.hetero_humanoid.hetero_humanoid_env import HeterogeneousHumanoidVelocityEnv

from .hetero_legged_env_cfg import HUMANOIDS, QUADRUPEDS, HeteroLeggedRoughEnvCfg
from .robot_configs import FEET, ROBOT_CONFIGS

# fmt: off
JOINT_DESCRIPTION_NAMES = (
    "pos_x", "pos_y", "pos_z",  # m, joint in the base frame, links in their zero-angle pose
    "axis_x", "axis_y", "axis_z",  # rotation axis in the base frame
    "limit_lower", "limit_upper", "default_pos",  # rad
    "effort_limit", "velocity_limit", "stiffness", "damping", "armature",  # the joint's actuator
    "action_scale",  # rad of target per unit of action
)
MORPHOLOGY_NAMES = (
    "total_mass", "base_mass_fraction",  # kg, -
    "num_feet", "num_joints",  # controlled joints
    "standing_height", "leg_length",  # m: spawn height of the base; base to feet, zero-angle pose
    "stance_length", "stance_width",  # m: spread of the feet along x / y, zero-angle pose
    "torque_per_weight",  # m: sum of joint effort limits / (total mass * g)
    "mean_velocity_limit", "mean_stiffness", "mean_damping", "action_scale",
)
# fmt: on


class HeteroLeggedVelocityEnv(HeterogeneousHumanoidVelocityEnv):
    """Quadrupeds and humanoids in one simulation, each robot with its own joints and its own scan points.

    Observations and actions are per robot, ``{robot_name: (that robot's number of envs, its own size)}``.
    One robot's observation, in order:

    * base linear velocity (3), base angular velocity (3), projected gravity (3), command (3)
    * base height above the ground under the robot (1)
    * joint position relative to default, joint velocity, last action: one value per own joint each, in the
      robot's joint order
    * terrain height at each scan point relative to the ground under the robot (m, + = higher, clipped to +-1)

    "The ground under the robot" is the median hit height of the 9 scan points nearest the base.

    Fixed descriptions per robot, for models that take any number of joints or scan points:

    * ``joint_descriptions[name]``: one row per own joint, columns ``JOINT_DESCRIPTION_NAMES``
    * ``scan_points[name]``: x, y of each scan point in m, in the robot's heading frame
    * ``morphology[name]``: whole-robot description, entries ``MORPHOLOGY_NAMES``

    ``final_obs[name]``: after a step in which some envs ended (fell or timed out), every env's observation just
    before those envs were reset. ``step()`` already returns the new episodes' first observations for them, so this
    is the only place the last state of an episode (e.g. the robot on the ground) can be seen.
    """

    cfg: HeteroLeggedRoughEnvCfg

    all_robot_names = QUADRUPEDS + HUMANOIDS
    robot_configs = ROBOT_CONFIGS

    def __init__(self, cfg: HeteroLeggedRoughEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.humanoids = list(cfg.robots)  # the parent env reads its robot list from cfg.humanoids
        for name, (length, width, resolution) in cfg.scan_patterns.items():
            pattern = getattr(cfg.scene, f"{name}_scanner").pattern_cfg
            pattern.size, pattern.resolution = [length, width], resolution

        super().__init__(cfg, render_mode, **kwargs)

        self.scan_points = {}
        self._ground_rays = {}
        for name, scanner in self.robot_scanners.items():
            xy = self._pattern_xy(scanner.cfg)
            self.scan_points[name] = xy
            self._ground_rays[name] = torch.topk(xy.norm(dim=1), k=min(9, len(xy)), largest=False).indices
        self._setup_descriptions()

    @staticmethod
    def _pattern_xy(scanner_cfg) -> torch.Tensor:
        """x, y of each scan point in the sensor's heading frame, in the order of the ray hits."""
        starts, _ = scanner_cfg.pattern_cfg.func(scanner_cfg.pattern_cfg, "cpu")
        return starts[:, :2] + torch.tensor(scanner_cfg.offset.pos[:2])

    def _setup_spaces(self, cfg):
        """Per-robot sizes: 13 base values + joint pos, joint vel and last action per own joint + one height per
        scan point."""
        num_joints = {
            name: sum(joint is not None for joint in self.robot_configs[name].canonical_joints)
            for name in self.humanoids_list
        }
        num_rays = {name: len(self._pattern_xy(getattr(cfg.scene, f"{name}_scanner"))) for name in self.humanoids_list}
        cfg.action_space = num_joints
        cfg.observation_space = {name: 13 + 3 * num_joints[name] + num_rays[name] for name in self.humanoids_list}

    def _get_observations(self) -> dict:
        noise = self.cfg.observation_noise
        add_noise = self.cfg.domain_randomization and noise.enabled
        obs = {}

        for name, robot in self.robots.items():
            env_ids = self.robot_env_ids[name]
            phys_idx = self.physical_joint_indices[name]

            lin_vel = robot.data.root_lin_vel_b.clone()
            ang_vel = robot.data.root_ang_vel_b.clone()
            proj_gravity = robot.data.projected_gravity_b.clone()
            joint_pos_rel = (robot.data.joint_pos - robot.data.default_joint_pos)[:, phys_idx]
            joint_vel = robot.data.joint_vel[:, phys_idx]

            hits_z = self.robot_scanners[name].data.ray_hits_w[..., 2]
            # a ray that hits nothing (off the terrain) reads inf: count it as the nominal ground under the robot
            nominal_ground = robot.data.root_pos_w[:, 2:3] - robot.cfg.init_state.pos[2]
            hits_z = torch.where(torch.isfinite(hits_z), hits_z, nominal_ground)
            ground_z = hits_z[:, self._ground_rays[name]].median(dim=1).values
            heights = hits_z - ground_z.unsqueeze(1)
            base_height = (robot.data.root_pos_w[:, 2] - ground_z).unsqueeze(1)

            if add_noise:
                lin_vel += torch.empty_like(lin_vel).uniform_(*noise.lin_vel_noise)
                ang_vel += torch.empty_like(ang_vel).uniform_(*noise.ang_vel_noise)
                proj_gravity += torch.empty_like(proj_gravity).uniform_(*noise.projected_gravity_noise)
                joint_pos_rel += torch.empty_like(joint_pos_rel).uniform_(*noise.joint_pos_noise)
                joint_vel += torch.empty_like(joint_vel).uniform_(*noise.joint_vel_noise)
                heights += torch.empty_like(heights).uniform_(*noise.height_measurement_noise)
                base_height += torch.empty_like(base_height).uniform_(*noise.height_measurement_noise)

            obs[name] = torch.cat(
                [
                    lin_vel,
                    ang_vel,
                    proj_gravity,
                    self._commands[env_ids],
                    base_height,
                    joint_pos_rel,
                    joint_vel,
                    self.actions[name],
                    heights.clamp(-1.0, 1.0),
                ],
                dim=-1,
            )

        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is not None and hasattr(self, "_ground_rays"):
            self.final_obs = self._get_observations()["policy"]
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # DirectRLEnv.reset() does this after its reset, but step() does not: without it the physics still reports
        # the pre-reset poses, and the new episodes' first height scans are taken where the robots ended (measured:
        # 2.4 m away on average, up to 9 m). Then make the scanners re-read at the new poses.
        self.scene.write_data_to_sim()
        self.sim.forward()
        for name, robot in self.robots.items():
            robot_env_ids_all = self.robot_env_ids[name]
            local_indices = torch.searchsorted(robot_env_ids_all, env_ids[torch.isin(env_ids, robot_env_ids_all)])
            if len(local_indices) > 0:
                self.robot_scanners[name].reset(local_indices)  # per-robot sensor: its own env indices
                # actuators with a state (the ANYmal actuator network) start each episode fresh, as in hetero_quadruped
                for actuator in robot.actuators.values():
                    actuator.reset(local_indices)

    def _setup_descriptions(self):
        """Per-joint and whole-robot descriptions, from each robot's USD and its actuator settings."""
        self.joint_descriptions, self.morphology = {}, {}
        stage = omni.usd.get_context().get_stage()

        for name, robot in self.robots.items():
            config = self.robot_configs[name]
            joint_frames, body_positions = self._usd_frames_in_base(stage, robot.cfg.prim_path[0], config.base_link)
            joint_names = robot.data.joint_names

            actuator_values = {key: torch.zeros(robot.num_joints) for key in JOINT_DESCRIPTION_NAMES[9:14]}
            for actuator in robot.actuators.values():
                ids = actuator.joint_indices
                ids = torch.arange(robot.num_joints) if isinstance(ids, slice) else torch.as_tensor(ids).cpu()
                for key in actuator_values:
                    actuator_values[key][ids] = getattr(actuator, key)[0].float().cpu()
            action_scale = getattr(self.cfg, f"action_scale_{name}", getattr(self.cfg, "action_scale", config.action_scale))
            limits = robot.data.joint_pos_limits[0].cpu()
            default_pos = robot.data.default_joint_pos[0].cpu()

            rows = []
            for i in self.physical_joint_indices[name].tolist():
                position, axis = joint_frames[joint_names[i]]
                rows.append(
                    [*position, *axis, limits[i, 0], limits[i, 1], default_pos[i]]
                    + [actuator_values[key][i] for key in actuator_values]
                    + [action_scale]
                )
                rows[-1] = [float(value) for value in rows[-1]]
            self.joint_descriptions[name] = torch.tensor(rows, dtype=torch.float, device=self.device)

            controlled = self.physical_joint_indices[name].cpu()
            masses = robot.data.default_mass[0].cpu()
            base_ids, _ = robot.find_bodies(config.base_link)
            _, feet_names = robot.find_bodies(FEET[name])
            feet = torch.tensor([body_positions[foot] for foot in feet_names])
            total_mass = masses.sum()
            morphology = (
                total_mass,
                masses[base_ids[0]] / total_mass,
                len(feet_names),
                len(controlled),
                robot.cfg.init_state.pos[2],
                -feet[:, 2].mean(),
                feet[:, 0].max() - feet[:, 0].min(),
                feet[:, 1].max() - feet[:, 1].min(),
                actuator_values["effort_limit"][controlled].sum() / (total_mass * 9.81),
                actuator_values["velocity_limit"][controlled].mean(),
                actuator_values["stiffness"][controlled].mean(),
                actuator_values["damping"][controlled].mean(),
                action_scale,
            )
            self.morphology[name] = torch.tensor([float(v) for v in morphology], device=self.device)
            print(f"[INFO - {name.upper()}] description: {len(rows)} joints, {len(feet_names)} feet, "
                  + ", ".join(f"{k} {v:.3g}" for k, v in zip(MORPHOLOGY_NAMES, self.morphology[name].tolist())))

    @staticmethod
    def _usd_frames_in_base(stage, robot_path: str, base_link: str):
        """Revolute joints ({name: (position, axis)}) and bodies ({name: position}) in the base frame, as authored
        in the USD (links in their zero-angle pose)."""
        cache = UsdGeom.XformCache()
        prims = list(Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()))
        bodies = {prim.GetName(): prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        to_base = cache.GetLocalToWorldTransform(bodies[base_link]).GetInverse()
        body_positions = {
            body: list((cache.GetLocalToWorldTransform(prim) * to_base).ExtractTranslation())
            for body, prim in bodies.items()
        }

        joint_frames = {}
        for prim in prims:
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            joint = UsdPhysics.RevoluteJoint(prim)
            parents = joint.GetBody0Rel().GetTargets()
            if not parents:
                continue
            rot, pos = joint.GetLocalRot0Attr().Get(), joint.GetLocalPos0Attr().Get()
            in_parent = Gf.Matrix4d().SetTransform(
                Gf.Rotation(Gf.Quatd(rot.GetReal(), Gf.Vec3d(*rot.GetImaginary()))), Gf.Vec3d(*pos)
            )
            frame = in_parent * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parents[0])) * to_base
            axis = Gf.Vec3d(*[float(letter == joint.GetAxisAttr().Get()) for letter in "XYZ"])
            joint_frames[prim.GetName()] = (list(frame.ExtractTranslation()), list(frame.TransformDir(axis).GetNormalized()))
        return joint_frames, body_positions
