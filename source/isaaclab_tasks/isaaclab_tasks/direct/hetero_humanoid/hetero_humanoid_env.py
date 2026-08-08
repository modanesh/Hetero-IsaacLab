# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Dict, List, Tuple

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import RewardManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor, RayCaster

from .hetero_humanoid_velocity_env_cfg import HeterogeneousHumanoidVelocityEnvCfg
from .robot_configs import ROBOT_CONFIGS


class HeterogeneousHumanoidVelocityEnv(DirectRLEnv):
    cfg: HeterogeneousHumanoidVelocityEnvCfg

    def __init__(self, cfg: HeterogeneousHumanoidVelocityEnvCfg, render_mode: str | None = None, **kwargs):
        self.all_humanoids = ["cassie", "digit", "g1", "h1"]
        self.humanoids_list = getattr(cfg, "humanoids", self.all_humanoids)

        print(f"[INFO] Instantiating environment with humanoids: {self.humanoids_list}")

        if cfg.scene.terrain.terrain_generator is not None:
            tg = cfg.scene.terrain.terrain_generator
            if "boxes" in tg.sub_terrains:
                tg.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
            if "random_rough" in tg.sub_terrains:
                tg.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
                tg.sub_terrains["random_rough"].noise_step = 0.01
            if getattr(cfg, "include_height_scanners", False):
                tg.curriculum = True

        self._setup_robot_distribution(cfg, self.humanoids_list)
        self._filter_rewards(cfg, self.humanoids_list)

        super().__init__(cfg, render_mode, **kwargs)

        self.scene.filter_collisions()

        self.robots: Dict[str, Articulation] = dict()
        self.robot_sensors: Dict[str, ContactSensor] = dict()
        self.robot_scanners: Dict[str, RayCaster] = dict()
        self.robot_env_ids: Dict[str, torch.Tensor] = dict()

        for robot_name in self.humanoids_list:
            self.robots[robot_name] = self.scene[robot_name]
            self.robot_sensors[robot_name] = self.scene[f"{robot_name}_contacts"]

            if getattr(self.cfg, "include_height_scanners", False):
                self.robot_scanners[robot_name] = self.scene[f"{robot_name}_scanner"]

            ids = getattr(self.cfg.scene, robot_name).env_ids
            ids.sort()
            self.robot_env_ids[robot_name] = torch.tensor(ids, device=self.device)

        if self.cfg.domain_randomization:
            self._apply_startup_randomization()
            self._apply_morphology_randomization()

        self._push_interval_s = (10.0, 15.0)
        self._next_push_time = torch.empty(self.num_envs, device=self.device).uniform_(*self._push_interval_s)
        self._elapsed_time = torch.zeros(self.num_envs, device=self.device)

        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._heading_target = torch.zeros(self.num_envs, device=self.device)

        self._is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._next_command_resample = torch.zeros(self.num_envs, device=self.device)

        self._metrics = {
            "error_vel_xy": torch.zeros(self.num_envs, device=self.device),
            "error_vel_yaw": torch.zeros(self.num_envs, device=self.device),
        }

        self.reward_manager = RewardManager(self.cfg.rewards, self)

        self.termination_results = {
            "time_out": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            "base_contact": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            "base_orientation": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

        if self.sim.has_gui():
            self.set_debug_vis(True)

        print("\n[INFO] generating action/observation mappings for biped humanoids...")
        self._setup_joint_mappings()

    def _setup_joint_mappings(self):
        """Build mappings between physical robot joint ordering and virtual policy ordering."""
        self.active_policy_indices = {}
        self.physical_joint_indices = {}
        self.joint_signs = {}

        for robot_name, robot in self.robots.items():
            if robot_name not in ROBOT_CONFIGS:
                raise ValueError(f"Robot '{robot_name}' not found in ROBOT_CONFIGS.")
                
            config = ROBOT_CONFIGS[robot_name]
            joint_names = robot.data.joint_names
            
            active_policy = []
            physical_joints = []

            for policy_idx, target_str in enumerate(config.canonical_joints):
                if target_str is None:
                    continue
                
                matched_idx = None
                for idx, jname in enumerate(joint_names):
                    if target_str == jname and idx not in physical_joints:
                        matched_idx = idx
                        break
                if matched_idx is None:
                    # Fallback search if exact match fails
                    for idx, jname in enumerate(joint_names):
                        if target_str in jname and idx not in physical_joints:
                            matched_idx = idx
                            break
                            
                if matched_idx is not None:
                    active_policy.append(policy_idx)
                    physical_joints.append(matched_idx)
                else:
                    print(f"[WARNING] Could not find joint matching '{target_str}' for robot {robot_name}")

            # Do NOT map remaining physical joints (e.g. passive springs on Digit).
            # If a robot has fingers that need to be actuated, they should be explicitly defined in canonical_joints.

            print(f"[INFO - {robot_name.upper()}] Mapped {len(physical_joints)} joints.")
            
            self.active_policy_indices[robot_name] = torch.tensor(active_policy, dtype=torch.long, device=self.device)
            self.physical_joint_indices[robot_name] = torch.tensor(physical_joints, dtype=torch.long, device=self.device)

            # Define kinematic sign multipliers to align human-legs (G1, H1) with bird-legs (Cassie, Digit)
            signs = torch.ones(len(active_policy), device=self.device)
            if config.kinematic_type == "humanoid":
                # Only apply flips to canonical leg joints (indices 0-11)
                for i, pol_idx in enumerate(active_policy):
                    if pol_idx < 12:
                        # Standard bird leg uses identity. Humanoid needs pitch/knee flipped.
                        # We also check for 'ankle' to catch H1's ankle, which doesn't have 'pitch' in its name.
                        jname = joint_names[physical_joints[i]]
                        if "pitch" in jname or "knee" in jname or "ankle" in jname:
                            signs[i] = -1.0
            
            self.joint_signs[robot_name] = signs




    def _setup_robot_distribution(self, cfg, active_humanoids: list):
        """Partition num_envs across active humanoids before scene creation."""
        num_envs = cfg.scene.num_envs
        num_robots = len(active_humanoids)
        if num_robots == 0:
            raise ValueError("The list of humanoids to use cannot be empty.")

        envs_per_robot = num_envs // num_robots
        start_idx = 0

        for i, robot_name in enumerate(active_humanoids):
            num_robot_envs = envs_per_robot + (1 if i < num_envs % num_robots else 0)
            env_ids = list(range(start_idx, start_idx + num_robot_envs))

            robot_cfg = getattr(cfg.scene, robot_name)
            robot_cfg.env_ids = env_ids
            robot_cfg.prim_path = [f"/World/envs/env_{j}/{robot_name}" for j in env_ids]

            sensor_cfg = getattr(cfg.scene, f"{robot_name}_contacts")
            sensor_cfg.env_ids = env_ids

            if getattr(cfg, "include_height_scanners", False):
                scanner_cfg = getattr(cfg.scene, f"{robot_name}_scanner")
                scanner_cfg.env_ids = env_ids

            start_idx += num_robot_envs
            print(f"[INFO] Robot '{robot_name}' assigned to envs: {env_ids[0]} to {env_ids[-1]}")

        # Delete unused robots from the scene
        for robot_name in self.all_humanoids:
            if robot_name not in active_humanoids:
                if hasattr(cfg.scene, robot_name):
                    delattr(cfg.scene, robot_name)
                sensor_name = f"{robot_name}_contacts"
                if hasattr(cfg.scene, sensor_name):
                    delattr(cfg.scene, sensor_name)
                if getattr(cfg, "include_height_scanners", False):
                    scanner_name = f"{robot_name}_scanner"
                    if hasattr(cfg.scene, scanner_name):
                        delattr(cfg.scene, scanner_name)


    def _filter_rewards(self, cfg, active_humanoids: list):
        """Delete reward terms for inactive robots to prevent SceneEntityCfg resolution errors."""
        attrs_to_delete = []
        for attr_name in dir(cfg.rewards):
            if attr_name.startswith("__"):
                continue
            for robot in self.all_humanoids:
                if robot not in active_humanoids and robot in attr_name:
                    attrs_to_delete.append(attr_name)
        
        for attr_name in attrs_to_delete:
            delattr(cfg.rewards, attr_name)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions = self.actions.clone()
        self.actions = actions.clone()

        self._resample_commands()

        # Process actions (scale + default joint pos)
        self.processed_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        for robot_name, robot in self.robots.items():
            env_ids = self.robot_env_ids[robot_name]
            active_policy = self.active_policy_indices[robot_name]
            phys_idx = self.physical_joint_indices[robot_name]

            config_scale = ROBOT_CONFIGS[robot_name].action_scale
            action_scale = getattr(self.cfg, f"action_scale_{robot_name}", getattr(self.cfg, "action_scale", config_scale))
            
            robot_actions = self.actions[env_ids][:, active_policy]
            scaled_actions = robot_actions * self.joint_signs[robot_name] * action_scale
            default_pos = robot.data.default_joint_pos[:, phys_idx]
            
            self.processed_actions[env_ids, :len(active_policy)] = scaled_actions + default_pos



    def _apply_action(self):
        for robot_name, robot in self.robots.items():
            env_ids = self.robot_env_ids[robot_name]
            local_indices = torch.arange(len(env_ids), device=self.device)
            active_policy = self.active_policy_indices[robot_name]
            phys_idx = self.physical_joint_indices[robot_name]

            # Write targets ONLY to native actuated joints using the joint_ids parameter
            targets = self.processed_actions[env_ids, :len(active_policy)]
            robot.set_joint_position_target(targets, env_ids=local_indices, joint_ids=phys_idx)

    def get_action_masks(self) -> torch.Tensor:
        """Get boolean mask indicating valid vs padded action dimensions for each environment.
        True = Valid action dimension, False = Padded dimension.
        """
        mask = torch.zeros((self.num_envs, self.cfg.num_actions), dtype=torch.bool, device=self.device)
        for robot_name, robot in self.robots.items():
            env_ids = self.robot_env_ids[robot_name]
            active_policy = self.active_policy_indices[robot_name]
            # Use advanced indexing to set the specific active policy slots to True
            mask[env_ids.unsqueeze(1), active_policy.unsqueeze(0)] = True
        return mask

    def _get_observations(self) -> dict:
        obs_buf = torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device)

        for robot_name, robot in self.robots.items():
            env_ids = self.robot_env_ids[robot_name]
            lin_vel = robot.data.root_lin_vel_b.clone()
            ang_vel = robot.data.root_ang_vel_b.clone()
            proj_gravity = robot.data.projected_gravity_b.clone()
            commands = self._commands[env_ids].clone()

            active_policy = self.active_policy_indices[robot_name]
            phys_idx = self.physical_joint_indices[robot_name]
            
            raw_joint_pos = (robot.data.joint_pos - robot.data.default_joint_pos)
            raw_joint_vel = robot.data.joint_vel

            joint_pos_rel = raw_joint_pos[:, phys_idx] * self.joint_signs[robot_name]
            joint_vel = raw_joint_vel[:, phys_idx] * self.joint_signs[robot_name]
            
            padded_joint_pos = torch.zeros(len(env_ids), self.cfg.num_actions, device=self.device)
            padded_joint_vel = torch.zeros(len(env_ids), self.cfg.num_actions, device=self.device)
            padded_joint_pos[:, active_policy] = joint_pos_rel
            padded_joint_vel[:, active_policy] = joint_vel

            padded_actions = torch.zeros(len(env_ids), self.cfg.num_actions, device=self.device)
            padded_actions[:, active_policy] = self.actions[env_ids][:, active_policy]

            if self.cfg.domain_randomization and self.cfg.observation_noise.enabled:
                noise_cfg = self.cfg.observation_noise
                lin_vel += torch.empty_like(lin_vel).uniform_(*noise_cfg.lin_vel_noise)
                ang_vel += torch.empty_like(ang_vel).uniform_(*noise_cfg.ang_vel_noise)
                proj_gravity += torch.empty_like(proj_gravity).uniform_(*noise_cfg.projected_gravity_noise)
                padded_joint_pos[:, active_policy] += torch.empty_like(joint_pos_rel).uniform_(*noise_cfg.joint_pos_noise)
                padded_joint_vel[:, active_policy] += torch.empty_like(joint_vel).uniform_(*noise_cfg.joint_vel_noise)

            robot_obs_list = [lin_vel, ang_vel, proj_gravity, commands, padded_joint_pos, padded_joint_vel, padded_actions]


            if getattr(self.cfg, "include_height_scanners", False):
                height_scanner = self.robot_scanners[robot_name]
                hits_z = height_scanner.data.ray_hits_w[..., 2]
                height_measurements = height_scanner.data.pos_w[:, 2].unsqueeze(1) - hits_z - self.cfg.height_scanner_offset
                if self.cfg.domain_randomization and self.cfg.observation_noise.enabled:
                    height_measurements += torch.empty_like(height_measurements).uniform_(*self.cfg.observation_noise.height_measurement_noise)
                height_measurements = torch.clip(height_measurements, -1.0, 1.0)
                robot_obs_list.append(height_measurements)

            robot_obs = torch.cat(robot_obs_list, dim=-1)
            obs_buf[env_ids, :] = robot_obs

        action_masks = self.get_action_masks().float()

        return {"policy": obs_buf, "action_mask": action_masks}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards using the reward manager."""
        return self.reward_manager.compute(dt=self.step_dt)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:

        """Check termination conditions for each robot type."""
        time_out = self.episode_length_buf >= self.max_episode_length
        self.termination_results["time_out"][:] = time_out
        base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for robot_name, sensor in self.robot_sensors.items():
            env_ids = self.robot_env_ids[robot_name]
            net_contact_forces = sensor.data.net_forces_w_history
            
            base_link_name = "torso"
            if robot_name in ROBOT_CONFIGS:
                base_link_name = ROBOT_CONFIGS[robot_name].base_link

            try:
                base_link, _ = sensor.find_bodies(base_link_name)
            except ValueError:
                base_link = [0]  # fallback
            terminated_robot = torch.any(
                torch.max(torch.norm(net_contact_forces[:, :, base_link], dim=-1), dim=1)[0] > self.cfg.contact_threshold,
                dim=1,
            )
            base_contact[env_ids] = terminated_robot
        self.termination_results["base_contact"][:] = base_contact
        
        base_orientation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for robot_name, robot in self.robots.items():
            if robot_name in ROBOT_CONFIGS:
                limit_angle = ROBOT_CONFIGS[robot_name].orientation_limit
                if limit_angle is not None:
                    env_ids = self.robot_env_ids[robot_name]
                    projected_gravity_z = robot.data.projected_gravity_b[:, 2]
                    # Clamp to prevent NaN in acos from floating point errors near 1.0 or -1.0
                    projected_gravity_z = torch.clamp(projected_gravity_z, -1.0, 1.0)
                    terminated_robot = torch.acos(-projected_gravity_z).abs() > limit_angle
                    base_orientation[env_ids] = terminated_robot
        
        self.termination_results["base_orientation"][:] = base_orientation

        dones = base_contact | base_orientation
        return dones, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if getattr(self.cfg, "include_height_scanners", False):
            self._update_terrain_curriculum(env_ids)

        if len(env_ids) > 0 and hasattr(self, "reward_manager") and self.reward_manager is not None:
            episode_sums = self.reward_manager._episode_sums
            log_data = {}
            for term_name, term_values in episode_sums.items():
                base_name = term_name
                is_active = True
                for robot in self.all_humanoids:
                    if term_name.endswith(f"_{robot}"):
                        if robot not in self.humanoids_list:
                            is_active = False
                        elif len(self.humanoids_list) == 1:
                            # Only strip the suffix if we are isolating a single robot
                            base_name = term_name[:-(len(robot)+1)]
                        break
                
                if is_active:
                    log_data[f"Episode_Reward/{base_name}"] = torch.mean(term_values[env_ids]).item() / self.cfg.episode_length_s

            for reason, results_tensor in self.termination_results.items():
                count = torch.count_nonzero(results_tensor[env_ids]).item()
                log_data[f"Episode_Termination/{reason}"] = count

            log_data["Metrics/base_velocity/error_vel_xy"] = torch.mean(self._metrics["error_vel_xy"][env_ids]).item()
            log_data["Metrics/base_velocity/error_vel_yaw"] = torch.mean(self._metrics["error_vel_yaw"][env_ids]).item()

            if getattr(self.cfg, "include_height_scanners", False) and hasattr(self.scene.terrain, "terrain_levels"):
                terrain_levels = self.scene.terrain.terrain_levels
                log_data["Curriculum/terrain_levels"] = terrain_levels[env_ids].float().mean().item()

            self.extras["log"] = log_data
            self._metrics["error_vel_xy"][env_ids] = 0.0
            self._metrics["error_vel_yaw"][env_ids] = 0.0

        # Always reset the physical states (this was previously hidden under domain_randomization)
        self._apply_reset_randomization(env_ids)

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self._next_command_resample[env_ids] = 0.0
        self._is_standing_env[env_ids] = False
        self._is_heading_env[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        if hasattr(self, "reward_manager") and self.reward_manager is not None:
            self.reward_manager.reset(env_ids)

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        if self.scene.terrain.cfg.terrain_generator is None:
            return
        if not getattr(self.scene.terrain.cfg.terrain_generator, "curriculum", False):
            return

        terrain_size = self.scene.terrain.cfg.terrain_generator.size[0]
        move_up = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        move_down = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        for robot_name, robot in self.robots.items():
            robot_env_ids_all = self.robot_env_ids[robot_name]
            mask = torch.isin(env_ids, robot_env_ids_all)
            if not mask.any():
                continue

            robot_global_ids = env_ids[mask]
            local_indices = torch.searchsorted(robot_env_ids_all, robot_global_ids)

            root_pos_2d = robot.data.root_pos_w[local_indices, :2]
            spawn_pos_2d = self.scene.env_origins[robot_global_ids, :2]
            distance = torch.norm(root_pos_2d - spawn_pos_2d, dim=1)

            robot_move_up = distance > (terrain_size / 2)
            tolerance_multiplier = 0.9
            commanded_distance = torch.norm(self._commands[robot_global_ids, :2], dim=1) * self.max_episode_length_s * 0.5 * tolerance_multiplier
            robot_move_down = distance < commanded_distance
            robot_move_down = robot_move_down & ~robot_move_up

            move_up[mask] = robot_move_up
            move_down[mask] = robot_move_down

        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)

    def _resample_commands(self):
        resample_mask = self.episode_length_buf >= self._next_command_resample
        resample_env_ids = torch.nonzero(resample_mask, as_tuple=False).flatten()

        if len(resample_env_ids) > 0:
            rand_time = torch.empty(len(resample_env_ids), device=self.device).uniform_(*self.cfg.resampling_time_range)
            self._next_command_resample[resample_env_ids] = self.episode_length_buf[resample_env_ids] + (rand_time / self.step_dt)

            r_lin_x = torch.empty(len(resample_env_ids), device=self.device).uniform_(*self.cfg.command_ranges_default["lin_vel_x"])
            r_lin_y = torch.empty(len(resample_env_ids), device=self.device).uniform_(*self.cfg.command_ranges_default["lin_vel_y"])
            r_ang_z = torch.empty(len(resample_env_ids), device=self.device).uniform_(*self.cfg.command_ranges_default["ang_vel_z"])

            self._commands[resample_env_ids, 0] = r_lin_x
            self._commands[resample_env_ids, 1] = r_lin_y
            self._commands[resample_env_ids, 2] = r_ang_z

            is_standing = torch.rand(len(resample_env_ids), device=self.device) < self.cfg.standing_probability
            self._is_standing_env[resample_env_ids] = is_standing
            self._commands[resample_env_ids[is_standing], :] = 0.0

    def _apply_startup_randomization(self):
        """Apply startup friction domain randomization."""
        print("[INFO - DR] Applying startup friction randomization for humanoids...")

    def _apply_morphology_randomization(self):
        """Apply mass and CoM randomization."""
        print("[INFO - DR] Applying morphology randomization for humanoids...")

    def _apply_reset_randomization(self, env_ids: torch.Tensor):
        """Apply reset pose randomization."""
        for robot_name, robot in self.robots.items():
            robot_env_ids_all = self.robot_env_ids[robot_name]
            mask = torch.isin(env_ids, robot_env_ids_all)
            robot_global_ids_to_reset = env_ids[mask]

            if len(robot_global_ids_to_reset) == 0:
                continue

            local_indices = torch.searchsorted(robot_env_ids_all, robot_global_ids_to_reset)
            num_resets = len(local_indices)

            joint_pos = robot.data.default_joint_pos[local_indices].clone()
            joint_vel = torch.zeros_like(robot.data.default_joint_vel[local_indices])

            default_root_state = robot.data.default_root_state[local_indices].clone()
            default_root_state[:, :3] += self.scene.env_origins[robot_global_ids_to_reset]

            robot.write_root_state_to_sim(default_root_state, env_ids=local_indices)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=local_indices)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                goal_marker_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
                goal_marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.goal_vel_visualizer = VisualizationMarkers(goal_marker_cfg)

                current_marker_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
                current_marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.current_vel_visualizer = VisualizationMarkers(current_marker_cfg)

            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robots or not any(robot.is_initialized for robot in self.robots.values()):
            return

        base_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        root_lin_vel_b = torch.zeros(self.num_envs, 2, device=self.device)
        base_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

        for robot_name, robot in self.robots.items():
            if not robot.is_initialized:
                continue
            env_ids = self.robot_env_ids[robot_name]
            base_pos_w[env_ids] = robot.data.root_pos_w.clone()
            root_lin_vel_b[env_ids] = robot.data.root_lin_vel_b[:, :2]
            base_quat_w[env_ids] = robot.data.root_quat_w

        base_pos_w[:, 2] += 0.8  # 0.8m above robot base
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2], base_quat_w)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(root_lin_vel_b, base_quat_w)

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor, base_quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

