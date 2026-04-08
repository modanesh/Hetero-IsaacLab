# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import List, Dict, Tuple

import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import RewardManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor, RayCaster

from .hetero_quadruped_velocity_env_cfg import HeterogeneousQuadrupedVelocityEnvCfg


class HeterogeneousQuadrupedVelocityEnv(DirectRLEnv):
    cfg: HeterogeneousQuadrupedVelocityEnvCfg

    def __init__(self, cfg: HeterogeneousQuadrupedVelocityEnvCfg, render_mode: str | None = None, **kwargs):
        # This is a safe way to get the argument
        self.all_quadrupeds = ["anymal_d", "anymal_c", "anymal_b", "unitree_a1", "unitree_go1", "unitree_go2", "unitree_b2", "spot"]
        quads_arg = kwargs.get("quadrupeds")
        if quads_arg is None:
            # Default to all if the argument is not passed
            self.quadrupeds_list = self.all_quadrupeds
        else:
            self.quadrupeds_list = quads_arg

        if getattr(cfg, "include_height_scanners", False) and "spot" in self.quadrupeds_list:
            raise ValueError("Remove 'spot' from the quadrupeds list for rough terrain.")

        print(f"[INFO] Instantiating environment with quadrupeds: {self.quadrupeds_list}")

        if cfg.scene.terrain.terrain_generator is not None:
            # Reduce grid height noise and step heights
            tg = cfg.scene.terrain.terrain_generator
            if "boxes" in tg.sub_terrains:
                tg.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
            if "random_rough" in tg.sub_terrains:
                tg.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
                tg.sub_terrains["random_rough"].noise_step = 0.01
            # Enable terrain curriculum for rough terrain training
            if getattr(cfg, "include_height_scanners", False):
                tg.curriculum = True

        # Set up robot distribution BEFORE calling super().__init__()
        # This must happen before the scene is created
        self._setup_robot_distribution(cfg, self.quadrupeds_list)

        # Filter rewards to only include those for active robots
        self._filter_rewards(cfg, self.quadrupeds_list)

        # Now call parent init which will create the scene
        super().__init__(cfg, render_mode, **kwargs)

        # Dictionaries to hold assets and their data
        self.robots: Dict[str, Articulation] = dict()
        self.robot_sensors: Dict[str, ContactSensor] = dict()
        self.robot_scanners: Dict[str, RayCaster] = dict()
        self.robot_env_ids: Dict[str, torch.Tensor] = dict()

        # Populate the dictionaries with active robots
        for robot_name in self.quadrupeds_list:
            self.robots[robot_name] = self.scene[robot_name]
            self.robot_sensors[robot_name] = self.scene[f"{robot_name}_contacts"]

            if getattr(self.cfg, "include_height_scanners", False):
                self.robot_scanners[robot_name] = self.scene[f"{robot_name}_scanner"]

            ids = getattr(self.cfg.scene, robot_name).env_ids
            ids.sort()
            self.robot_env_ids[robot_name] = torch.tensor(ids, device=self.device)

        # Apply Standard Domain Randomization (Friction, Mass, CoM)
        if self.cfg.domain_randomization:
            self._apply_startup_randomization()
            self._apply_morphology_randomization()

        # Initialize push robot timer for interval events
        self._push_interval_s = (10.0, 15.0)
        self._next_push_time = torch.empty(self.num_envs, device=self.device).uniform_(*self._push_interval_s)
        self._elapsed_time = torch.zeros(self.num_envs, device=self.device)

        # Buffers for actions
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)

        # Command State Buffers
        # Velocity commands (vx, vy, wz)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Desired heading for heading control
        self._heading_target = torch.zeros(self.num_envs, device=self.device)

        # State flags
        self._is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Resampling timer
        self._next_command_resample = torch.zeros(self.num_envs, device=self.device)

        # Metrics Buffers (Replicating _update_metrics)
        self._metrics = {
            "error_vel_xy": torch.zeros(self.num_envs, device=self.device),
            "error_vel_yaw": torch.zeros(self.num_envs, device=self.device)
        }

        self.reward_manager = RewardManager(self.cfg.rewards, self)

        self.termination_results = {
            "time_out": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            "base_contact": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

        if self.sim.has_gui():
            self.set_debug_vis(True)

        # We enforce "ANYMAL Joint-Major" format for the Policy:
        #   Leg Order:   LF, LH, RF, RH
        #   Joint Order: HAA (Hip), HFE (Thigh), KFE (Calf)
        print("\n[INFO] generating action/observation mappings to match ANYMAL format...")

        # The Target Policy Sequence (Anymal Order)
        target_joint_types = ["HAA", "HFE", "KFE"]  # The "Types"
        target_leg_sequence = ["LF", "LH", "RF", "RH"]  # The "Legs"

        virtual_joint_names = []
        # Outer Loop: JOINT TYPE (Matches ANYmal.py regex order)
        for j_type in target_joint_types:
            # Inner Loop: LEG SEQUENCE (Matches ANYmal physical order)
            for leg in target_leg_sequence:
                virtual_joint_names.append(f"{leg}_{j_type}")

        # Result: [LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, LH_HFE...]
        print(f"[INFO] Target Policy Action Order: {virtual_joint_names}")

        self.action_indices = {}
        self.obs_indices = {}

        # Build Mappings for Each Robot
        for robot_name, robot in self.robots.items():
            physical_joint_names = robot.data.joint_names
            if "anymal" in robot_name:
                # ANYmal matches the reference exactly.
                leg_map = {"LF": "LF", "LH": "LH", "RF": "RF", "RH": "RH"}
                joint_map = {"HAA": "HAA", "HFE": "HFE", "KFE": "KFE"}

            elif "spot" in robot_name:
                # Spot uses "fl" (Front Left) for "LF", etc.
                # Spot uses "hl" (Hind Left) for "LH".
                leg_map = {"LF": "fl", "LH": "hl", "RF": "fr", "RH": "hr"}
                joint_map = {"HAA": "hx", "HFE": "hy", "KFE": "kn"}

            else:  # Unitree (A1, Go1, Go2, B2)
                # Policy "LF" (Left Front) -> Unitree "FL" (Front Left)
                # Policy "LH" (Left Hind)  -> Unitree "RL" (Rear Left)
                # Policy "RF" (Right Front)-> Unitree "FR" (Front Right)
                # Policy "RH" (Right Hind) -> Unitree "RR" (Rear Right)
                leg_map = {"LF": "FL", "LH": "RL", "RF": "FR", "RH": "RR"}
                joint_map = {"HAA": "hip", "HFE": "thigh", "KFE": "calf"}

            # Helper to find Physical Index
            def get_phys_index(virt_leg, virt_joint):
                p_leg = leg_map[virt_leg]
                p_joint = joint_map[virt_joint]

                # Robust search: find the joint name containing both strings
                for i, name in enumerate(physical_joint_names):
                    if p_leg in name and p_joint in name:
                        return i
                raise ValueError(f"[{robot_name}] Phys Joint not found for {virt_leg}_{virt_joint}")

            # Action Reordering (Policy -> Robot)
            # We want to know: For Physical Index 0 (e.g., Unitree FL_hip),
            # which index in the Policy Vector should we grab?
            to_robot_indices = []

            # Helper reverse lookup for parsing physical names back to virtual
            rev_leg_map = {v: k for k, v in leg_map.items()}
            rev_joint_map = {v: k for k, v in joint_map.items()}

            for phys_name in physical_joint_names:
                # Identify what this physical joint is (e.g. "FL_hip")
                # Find the Virtual Leg key (e.g. "FL" maps back to "LF")
                v_leg = next((v_k for p_v, v_k in rev_leg_map.items() if p_v in phys_name), None)
                v_joint = next((v_k for p_v, v_k in rev_joint_map.items() if p_v in phys_name), None)

                if v_leg is None or v_joint is None:
                    raise ValueError(f"[{robot_name}] Could not parse physical joint: {phys_name}")

                # Find where "LF_HAA" lives in the Virtual list
                target_key = f"{v_leg}_{v_joint}"
                policy_idx = virtual_joint_names.index(target_key)
                to_robot_indices.append(policy_idx)

            # Observation Reordering (Robot -> Policy)
            # We want to fill the Policy Vector. For Virtual Index 0 (LF_HAA),
            # which Physical Index do we grab?
            to_policy_indices = []

            for virt_name in virtual_joint_names:
                v_leg, v_jtype = virt_name.split("_")
                phys_idx = get_phys_index(v_leg, v_jtype)
                to_policy_indices.append(phys_idx)

            self.action_indices[robot_name] = torch.tensor(to_robot_indices, device=self.device, dtype=torch.long)
            self.obs_indices[robot_name] = torch.tensor(to_policy_indices, device=self.device, dtype=torch.long)

            # Debug print for verification for all joints and robots
            reordered_actions = [physical_joint_names[i] for i in to_robot_indices]
            print(f"[{robot_name}] Action Reorder Map: {reordered_actions}")

        print("[INFO] Action/Observation mapping complete.\n")

    def _filter_rewards(self, cfg: HeterogeneousQuadrupedVelocityEnvCfg, quadrupeds_to_use: List[str]):
        """Filter reward terms to only include those for active robots."""
        all_reward_terms = list(vars(cfg.rewards).keys())

        for term_name in all_reward_terms:
            # Skip private attributes and methods
            if term_name.startswith('_'): continue

            # Check if this reward term is for a robot we're NOT using
            should_remove = False
            for robot_name in self.all_quadrupeds:
                if robot_name not in quadrupeds_to_use and robot_name in term_name:
                    should_remove = True
                    break

            if should_remove:
                # Remove this reward term
                delattr(cfg.rewards, term_name)

    def _setup_robot_distribution(self, cfg: HeterogeneousQuadrupedVelocityEnvCfg, quadrupeds_to_use: List[str]):
        """Set up robot distribution across environments before scene creation."""
        # Dynamically assign envs to the selected robots
        num_envs = cfg.scene.num_envs
        num_robots_to_use = len(quadrupeds_to_use)
        if num_robots_to_use == 0:
            raise ValueError("The list of quadrupeds to use cannot be empty.")

        envs_per_robot = num_envs // num_robots_to_use
        start_idx = 0

        # Assign env_ids for the robots we want to use
        for i, robot_name in enumerate(quadrupeds_to_use):
            num_robot_envs = envs_per_robot + (1 if i < num_envs % num_robots_to_use else 0)
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

        # For robots we are NOT using, REMOVE them from the scene entirely
        for robot_name in self.all_quadrupeds:
            if robot_name not in quadrupeds_to_use:
                # Delete the robot configuration from the scene
                if hasattr(cfg.scene, robot_name):
                    delattr(cfg.scene, robot_name)
                    print(f"[INFO] Removed robot config: {robot_name}")

                # Delete the sensor configuration from the scene
                sensor_name = f"{robot_name}_contacts"
                if hasattr(cfg.scene, sensor_name):
                    delattr(cfg.scene, sensor_name)

                if getattr(cfg, "include_height_scanners", False):
                    scanner_name = f"{robot_name}_scanner"
                    if hasattr(cfg.scene, scanner_name):
                        delattr(cfg.scene, scanner_name)

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
            base_link = [0]  # base link is always body 0
            terminated_robot = torch.any(torch.max(torch.norm(net_contact_forces[:, :, base_link], dim=-1), dim=1)[0] > self.cfg.contact_threshold, dim=1)
            base_contact[env_ids] = terminated_robot
        self.termination_results["base_contact"][:] = base_contact
        return base_contact, time_out

    def _pre_physics_step(self, actions: torch.Tensor):
        """Preprocess actions (apply scales and offsets) before physics."""
        self.previous_actions[:] = self.actions
        self.actions[:] = actions
        self.processed_actions = torch.zeros_like(self.actions)

        for robot_name, robot in self.robots.items():
            env_ids = self.robot_env_ids[robot_name]

            # Get Policy Actions
            raw_actions = self.actions[env_ids].clone()

            # Reorder: Policy (Virtual) -> Robot (Physical)
            # Use action_indices
            reorder_idx = self.action_indices[robot_name]
            robot_ordered_actions = raw_actions[:, reorder_idx]

            # Apply scales
            action_scale_cfg_name = f"action_scale_{robot_name}"
            action_scale = getattr(self.cfg, action_scale_cfg_name)

            self.processed_actions[env_ids] = (action_scale * robot_ordered_actions + robot.data.default_joint_pos)

        # Resample if timer expired
        self._resample_commands()

        # Update commands (Heading control + Standing enforcement)
        self._update_velocity_commands()

        # Update Metrics (Tracking error)
        self._update_metrics()

        # Interval randomization (Pushing)
        self._apply_interval_randomization()

    def _resample_commands(self):
        """Resample velocity commands periodically."""
        # Update timer
        self._next_command_resample -= self.step_dt

        # Find environments that need new commands
        resample_mask = self._next_command_resample <= 0.0

        if resample_mask.any():
            env_ids = torch.where(resample_mask)[0]
            num_resample = len(env_ids)
            ranges = self.cfg.command_ranges_default

            # Sample uniform random numbers
            r = torch.empty(num_resample, device=self.device)

            # Linear Velocity X
            self._commands[env_ids, 0] = r.uniform_(*ranges["lin_vel_x"])
            # Linear Velocity Y
            self._commands[env_ids, 1] = r.uniform_(*ranges["lin_vel_y"])
            # Angular Velocity Z
            self._commands[env_ids, 2] = r.uniform_(*ranges["ang_vel_z"])

            # Heading Target
            if self.cfg.heading_command:
                self._heading_target[env_ids] = r.uniform_(*ranges["heading"])
                # Update heading envs based on probability
                self._is_heading_env[env_ids] = torch.rand(num_resample, device=self.device) <= self.cfg.rel_heading_envs

            # Standing Envs
            self._is_standing_env[env_ids] = torch.rand(num_resample, device=self.device) <= self.cfg.rel_standing_envs

            # Reset Timer
            min_t, max_t = self.cfg.resampling_time_range
            self._next_command_resample[env_ids] = torch.empty(num_resample, device=self.device).uniform_(min_t, max_t)

    def _update_velocity_commands(self):
        """Compute angular velocity from heading error and enforce standing (zero velocity)."""
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # Resolve indices of heading envs (Only those marked as heading envs!)
            heading_env_ids = torch.nonzero(self._is_heading_env, as_tuple=False).flatten()

            if len(heading_env_ids) > 0:
                # We need to gather current yaw from all robots.
                # Since robots are heterogeneous, we iterate and fill a buffer or process per robot.
                for robot_name, robot in self.robots.items():
                    # Find intersection of this robot's envs and the heading_env_ids
                    robot_indices = self.robot_env_ids[robot_name]
                    # Intersection logic
                    mask = torch.isin(robot_indices, heading_env_ids)
                    active_ids = robot_indices[mask]

                    if len(active_ids) > 0:
                        # Get current Heading (Yaw)
                        # DirectRL doesn't have .heading_w property on articulation, compute from quat
                        forward = math_utils.quat_apply(
                            robot.data.root_quat_w[mask],
                            torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(len(active_ids), 1)
                        )
                        current_yaw = torch.atan2(forward[:, 1], forward[:, 0])

                        # Calculate Error
                        heading_error = math_utils.wrap_to_pi(self._heading_target[active_ids] - current_yaw)

                        # P-Control
                        cmd_ang_vel = self.cfg.heading_control_stiffness * heading_error

                        # Clip
                        min_w, max_w = self.cfg.command_ranges_default["ang_vel_z"]
                        cmd_ang_vel = torch.clip(cmd_ang_vel, min=min_w, max=max_w)

                        # Set Command
                        self._commands[active_ids, 2] = cmd_ang_vel

        # Enforce standing (zero velocity)
        standing_env_ids = torch.nonzero(self._is_standing_env, as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self._commands[standing_env_ids, :] = 0.0

    def _update_metrics(self):
        """Update tracking metrics."""
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self.step_dt

        # We need to iterate robots to get actual velocities
        for robot_name, robot in self.robots.items():
            env_ids = self.robot_env_ids[robot_name]
            lin_vel_error = torch.norm(
                self._commands[env_ids, :2] - robot.data.root_lin_vel_b[:, :2],
                dim=-1
            )
            ang_vel_error = torch.abs(
                self._commands[env_ids, 2] - robot.data.root_ang_vel_b[:, 2]
            )

            self._metrics["error_vel_xy"][env_ids] += lin_vel_error / max_command_step
            self._metrics["error_vel_yaw"][env_ids] += ang_vel_error / max_command_step

    def _apply_action(self):
        """Apply the processed joint position targets to each robot."""
        for robot_name, robot in self.robots.items():
            env_ids = self.robot_env_ids[robot_name]
            local_indices = torch.arange(len(env_ids), device=self.device)
            robot.set_joint_position_target(
                self.processed_actions[env_ids],
                env_ids=local_indices
            )

    def _get_observations(self) -> dict:
        """Get observations from all robots and combine them."""
        obs_buf = torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device)

        for robot_name, robot in self.robots.items():
            env_ids = self.robot_env_ids[robot_name]
            lin_vel = robot.data.root_lin_vel_b.clone()  # (num_robot_envs, 3)
            ang_vel = robot.data.root_ang_vel_b.clone()  # (num_robot_envs, 3)
            proj_gravity = robot.data.projected_gravity_b.clone()  # (num_robot_envs, 3)
            commands = self._commands[env_ids].clone()  # (num_robot_envs, 3)

            # Get Physical Data
            raw_joint_pos = (robot.data.joint_pos - robot.data.default_joint_pos)
            raw_joint_vel = robot.data.joint_vel

            # Reorder: Robot (Physical) -> Policy (Virtual)
            # Use obs_indices
            reorder_idx = self.obs_indices[robot_name]

            joint_pos_rel = raw_joint_pos[:, reorder_idx]
            joint_vel = raw_joint_vel[:, reorder_idx]

            actions = self.actions[env_ids].clone()  # Already in Policy Order

            if self.cfg.domain_randomization and self.cfg.observation_noise.enabled:
                noise_cfg = self.cfg.observation_noise

                # Linear velocity noise
                lin_vel += torch.empty_like(lin_vel).uniform_(*noise_cfg.lin_vel_noise)

                # Angular velocity noise
                ang_vel += torch.empty_like(ang_vel).uniform_(*noise_cfg.ang_vel_noise)

                # Projected gravity noise
                proj_gravity += torch.empty_like(proj_gravity).uniform_(*noise_cfg.projected_gravity_noise)

                # Joint position noise
                joint_pos_rel += torch.empty_like(joint_pos_rel).uniform_(*noise_cfg.joint_pos_noise)

                # Joint velocity noise
                joint_vel += torch.empty_like(joint_vel).uniform_(*noise_cfg.joint_vel_noise)

            # Start building the observation list
            robot_obs_list = [lin_vel, ang_vel, proj_gravity, commands, joint_pos_rel, joint_vel, actions]

            if getattr(self.cfg, "include_height_scanners", False):
                height_scanner = self.robot_scanners[robot_name]
                hits_z = height_scanner.data.ray_hits_w[..., 2]

                # Match manager-based mdp.height_scan: sensor_z - hit_z - 0.5
                # The 0.5 offset centres the signal around the nominal standing height.
                # Clip to (-1.0, 1.0) matches the manager-based ObsTerm clip parameter.
                height_measurements = height_scanner.data.pos_w[:, 2].unsqueeze(1) - hits_z - self.cfg.height_scanner_offset

                if self.cfg.domain_randomization and self.cfg.observation_noise.enabled:
                    height_measurements += torch.empty_like(height_measurements).uniform_(*self.cfg.observation_noise.height_measurement_noise)

                height_measurements = torch.clip(height_measurements, -1.0, 1.0)
                robot_obs_list.append(height_measurements)

            robot_obs = torch.cat(robot_obs_list, dim=-1)
            obs_buf[env_ids] = robot_obs

        return {"policy": obs_buf}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Update terrain levels BEFORE resetting state so we can read the final
        # episode velocities while the robots are still in their end-of-episode pose.
        if getattr(self.cfg, "include_height_scanners", False):
            self._update_terrain_curriculum(env_ids)

        # Log episode data
        if len(env_ids) > 0 and hasattr(self, 'reward_manager') and self.reward_manager is not None:
            episode_sums = self.reward_manager._episode_sums
            log_data = {}
            for term_name, term_values in episode_sums.items():
                log_data[f"Episode_Reward/{term_name}"] = torch.mean(term_values[env_ids]).item() / self.cfg.episode_length_s

            for reason, results_tensor in self.termination_results.items():
                count = torch.count_nonzero(results_tensor[env_ids]).item()
                log_data[f"Episode_Termination/{reason}"] = count

            # Add metrics to log_data so they appear in standard logs
            # Naming convention matches ManagerBasedEnv metrics
            log_data["Metrics/base_velocity/error_vel_xy"] = torch.mean(self._metrics["error_vel_xy"][env_ids]).item()
            log_data["Metrics/base_velocity/error_vel_yaw"] = torch.mean(self._metrics["error_vel_yaw"][env_ids]).item()

            if getattr(self.cfg, "include_height_scanners", False) and hasattr(self.scene.terrain, "terrain_levels"):
                terrain_levels = self.scene.terrain.terrain_levels  # shape: (num_envs,)
                log_data["Curriculum/terrain_levels"] = terrain_levels[env_ids].float().mean().item()

            self.extras["log"] = log_data

            # Reset the metrics buffers for these envs
            self._metrics["error_vel_xy"][env_ids] = 0.0
            self._metrics["error_vel_yaw"][env_ids] = 0.0

        # Apply reset randomization (handles all robots)
        if self.cfg.domain_randomization:
            self._apply_reset_randomization(env_ids)
        else:
            # Default reset without randomization
            for robot_name, robot in self.robots.items():
                robot_env_ids_all = self.robot_env_ids[robot_name]
                # Find which of the env_ids to reset belong to this robot type
                mask = torch.isin(env_ids, robot_env_ids_all)
                robot_global_ids_to_reset = env_ids[mask]

                if len(robot_global_ids_to_reset) > 0:
                    # Convert global env_ids to local indices for this robot
                    # For anymal_d: global [0,1,2,3] -> local [0,1,2,3]
                    # For unitree_a1: global [8,9,10,11] -> local [0,1,2,3]
                    local_indices = torch.searchsorted(robot_env_ids_all, robot_global_ids_to_reset)

                    # Get default state
                    joint_pos = robot.data.default_joint_pos[local_indices]
                    joint_vel = robot.data.default_joint_vel[local_indices]
                    default_root_state = robot.data.default_root_state[local_indices].clone()

                    # Apply env origins using the global IDs
                    default_root_state[:, :3] += self.scene.env_origins[robot_global_ids_to_reset]

                    # Write state using LOCAL indices
                    robot.write_root_state_to_sim(default_root_state, env_ids=local_indices)
                    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=local_indices)

        # Reset actuators
        for robot_name, robot in self.robots.items():
            robot_env_ids_all = self.robot_env_ids[robot_name]
            mask = torch.isin(env_ids, robot_env_ids_all)
            robot_global_ids_to_reset = env_ids[mask]
            if len(robot_global_ids_to_reset) > 0:
                local_indices = torch.searchsorted(robot_env_ids_all, robot_global_ids_to_reset)
                for actuator in robot.actuators.values():
                    actuator.reset(local_indices)

        # Reset action buffers for the originally requested env_ids
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

        # Reset commands by forcing a resample in the next pre_physics_step
        self._next_command_resample[env_ids] = 0.0
        # Initialize standing/heading state to False (will be updated by resample)
        self._is_standing_env[env_ids] = False
        self._is_heading_env[env_ids] = False

        # Reset episode tracking buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # Reset push timers for reset environments
        if self.cfg.domain_randomization:
            self._elapsed_time[env_ids] = 0.0
            self._next_push_time[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self._push_interval_s)

        # Reset reward manager
        if hasattr(self, 'reward_manager') and self.reward_manager is not None:
            self.reward_manager.reset(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization for velocity arrows."""
        if debug_vis:
            # Create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # commanded velocity (green arrow)
                goal_marker_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/velocity_goal"
                )
                goal_marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.goal_vel_visualizer = VisualizationMarkers(goal_marker_cfg)

                # current velocity (blue arrow)
                current_marker_cfg = BLUE_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/velocity_current"
                )
                current_marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.current_vel_visualizer = VisualizationMarkers(current_marker_cfg)

            # Set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization each frame."""
        # Check if any robot is initialized
        if not self.robots or not any(robot.is_initialized for robot in self.robots.values()):
            return

        # Initialize tensors for all environments
        base_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        root_lin_vel_b = torch.zeros(self.num_envs, 2, device=self.device)
        base_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

        # Gather data from each robot type
        for robot_name, robot in self.robots.items():
            if not robot.is_initialized:
                continue

            env_ids = self.robot_env_ids[robot_name]
            base_pos_w[env_ids] = robot.data.root_pos_w.clone()
            root_lin_vel_b[env_ids] = robot.data.root_lin_vel_b[:, :2]
            base_quat_w[env_ids] = robot.data.root_quat_w

        # Raise marker location (above robot base)
        base_pos_w[:, 2] += 0.5  # 0.5m above the base
        # Resolve the scales and quaternions for commanded velocity
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self._commands[:, :2],  # commanded x, y velocities
            base_quat_w
        )
        # Resolve the scales and quaternions for actual velocity
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            root_lin_vel_b,  # actual x, y velocities in base frame
            base_quat_w
        )

        # Display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(
            self, xy_velocity: torch.Tensor, base_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert XY velocity to arrow scale and orientation."""
        # Get default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale

        # Calculate arrow scale based on velocity magnitude
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        # Calculate arrow direction from velocity
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        # Convert from base frame to world frame
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        """Update terrain difficulty levels based on distance walked.

        Mirrors ``mdp.terrain_levels_vel`` from the manager-based curriculum:
        - Move UP if distance walked > terrain_size / 2
        - Move DOWN if distance walked < commanded_vel * episode_length * 0.5
        """
        if self.scene.terrain.cfg.terrain_generator is None:
            return
        if not getattr(self.scene.terrain.cfg.terrain_generator, "curriculum", False):
            return

        # terrain_size is typically 8m for rough terrain
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

            # Compute distance walked from spawn point
            root_pos_2d = robot.data.root_pos_w[local_indices, :2]
            spawn_pos_2d = self.scene.env_origins[robot_global_ids, :2]
            distance = torch.norm(root_pos_2d - spawn_pos_2d, dim=1)

            # Move up if walked more than half the terrain size
            robot_move_up = distance > (terrain_size / 2)

            # Move down if walked less than half the expected distance
            # Expected distance = commanded_vel_magnitude * episode_length * 0.5
            commanded_distance = torch.norm(self._commands[robot_global_ids, :2], dim=1) * self.max_episode_length_s * 0.5
            robot_move_down = distance < commanded_distance
            # Make move_down mutually exclusive with move_up
            robot_move_down = robot_move_down & ~robot_move_up

            move_up[mask] = robot_move_up
            move_down[mask] = robot_move_down

        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)

    def _apply_startup_randomization(self):
        """Apply one-time startup randomization (friction only) using 64 buckets."""
        print("[INFO - DR] Applying startup domain randomization (friction) using 64 buckets...")

        # Get friction ranges - aligned with manager-based default (0.8, 0.8) static, (0.6, 0.6) dynamic
        if hasattr(self.cfg, "friction_range"):
            friction_range = self.cfg.friction_range
        else:
            friction_range = ((0.8, 0.8), (0.6, 0.6))

        # Manager-based implementation uses a fixed restitution range of (0.0, 0.0) by default
        restitution_range = (0.0, 0.0)
        num_buckets = 64

        for robot_name, robot in self.robots.items():
            num_robot_envs = len(self.robot_env_ids[robot_name])

            # Instead of sampling num_envs unique values, we sample 64 unique values
            s_buckets = torch.empty(num_buckets, device="cpu").uniform_(*friction_range[0])
            d_buckets = torch.empty(num_buckets, device="cpu").uniform_(*friction_range[1])
            r_buckets = torch.empty(num_buckets, device="cpu").uniform_(*restitution_range)

            # Assign Environments to Buckets
            # Randomly assign each environment to one of the 64 buckets
            bucket_ids = torch.randint(0, num_buckets, (num_robot_envs,), device="cpu")

            static_friction = s_buckets[bucket_ids]
            dynamic_friction = d_buckets[bucket_ids]
            restitution = r_buckets[bucket_ids]

            # Get current materials to determine shape
            current_materials = robot.root_physx_view.get_material_properties()

            # PhysX material properties shape is usually (num_envs, 3) or (num_envs, num_shapes, 3)
            # 3 corresponds to [static_friction, dynamic_friction, restitution]
            if current_materials.ndim == 2:
                # Shape: (num_envs, 3)
                current_materials[:, 0] = static_friction
                current_materials[:, 1] = dynamic_friction
                current_materials[:, 2] = restitution
            else:
                # Shape: (num_envs, num_shapes, 3)
                # Broadcast the environment property across all shapes in that environment
                current_materials[:, :, 0] = static_friction.unsqueeze(1)
                current_materials[:, :, 1] = dynamic_friction.unsqueeze(1)
                current_materials[:, :, 2] = restitution.unsqueeze(1)

            # Set material properties (indices must be on CPU)
            local_env_ids = torch.arange(num_robot_envs, device="cpu", dtype=torch.int32)
            robot.root_physx_view.set_material_properties(current_materials, indices=local_env_ids)
            print(f"    [{robot_name}] Randomized friction")

    def _apply_interval_randomization(self):
        """Apply periodic randomization (push robots)."""
        if not self.cfg.domain_randomization:
            return

        # Update elapsed time
        self._elapsed_time += self.step_dt

        # Check which environments need a push
        push_mask = self._elapsed_time >= self._next_push_time

        if push_mask.any():
            for robot_name, robot in self.robots.items():
                robot_env_ids = self.robot_env_ids[robot_name]

                # Find which robot envs need pushing
                robot_push_mask = push_mask[robot_env_ids]
                if not robot_push_mask.any():
                    continue

                # Get local indices that need pushing
                local_push_indices = torch.where(robot_push_mask)[0]

                # Generate random velocity push
                num_to_push = len(local_push_indices)
                vel_push = torch.zeros(num_to_push, 6, device=self.device)
                vel_push[:, 0].uniform_(-0.5, 0.5)  # x velocity
                vel_push[:, 1].uniform_(-0.5, 0.5)  # y velocity

                # Get current velocities and add push
                current_vel = robot.data.root_vel_w[local_push_indices].clone()
                current_vel[:, : 2] += vel_push[:, :2]

                # Apply the push
                robot.write_root_velocity_to_sim(current_vel, env_ids=local_push_indices)

            # Reset timers for pushed environments
            push_env_ids = torch.where(push_mask)[0]
            self._elapsed_time[push_env_ids] = 0.0
            self._next_push_time[push_env_ids] = torch.empty(len(push_env_ids), device=self.device).uniform_(*self._push_interval_s)

    def _apply_reset_randomization(self, env_ids: torch.Tensor):
        """Apply randomization on reset (initial pose, velocity, joint positions)."""
        for robot_name, robot in self.robots.items():
            robot_env_ids_all = self.robot_env_ids[robot_name]

            # Find which of the env_ids to reset belong to this robot type
            mask = torch.isin(env_ids, robot_env_ids_all)
            robot_global_ids_to_reset = env_ids[mask]

            if len(robot_global_ids_to_reset) == 0:
                continue

            local_indices = torch.searchsorted(robot_env_ids_all, robot_global_ids_to_reset)
            num_resets = len(local_indices)

            # Manager default: scale (0.5, 1.5), velocity (0.0)
            joint_pos = robot.data.default_joint_pos[local_indices].clone()
            joint_vel = torch.zeros_like(robot.data.default_joint_vel[local_indices])  # Velocity is 0.0

            # Determine mode (retain 'offset' for Spot compatibility, default 'scale' for others)
            pos_mode_attr = f"reset_joint_pos_mode_{robot_name}"
            position_mode = getattr(self.cfg, pos_mode_attr, "scale")

            # Position Range: Manager uses (0.5, 1.5)
            # We use this as default unless specifically overridden for Spot (offset)
            if position_mode == "scale":
                # Dynamically fetch the position range for this specific robot
                pos_range_attr = f"reset_joint_pos_range_{robot_name}"
                position_range = getattr(self.cfg, pos_range_attr, (0.5, 1.5))  # Default fallback
                position_scale = torch.empty(num_resets, robot.num_joints, device=robot.device).uniform_(*position_range)
                joint_pos = joint_pos * position_scale
            elif position_mode == "offset":
                # Fallback for Spot: use configured range or safe default
                pos_range_attr = f"reset_joint_pos_range_{robot_name}"
                position_range = getattr(self.cfg, pos_range_attr, (-0.2, 0.2))
                position_offset = torch.empty(num_resets, robot.num_joints, device=robot.device).uniform_(*position_range)
                joint_pos = joint_pos + position_offset

            # Clamp to joint limits
            joint_pos_limits = robot.data.soft_joint_pos_limits[local_indices]
            joint_pos = joint_pos.clamp(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

            # Base State
            # Manager default:
            #   Pose: x,y (-0.5, 0.5), yaw (-3.14, 3.14). NO Z NOISE.
            #   Velocity: x,y,z (-0.5, 0.5), ang (-0.5, 0.5).

            default_root_state = robot.data.default_root_state[local_indices].clone()

            # Apply env origins
            default_root_state[:, : 3] += self.scene.env_origins[robot_global_ids_to_reset]

            # Position randomization (X, Y only)
            default_root_state[:, 0] += torch.empty(num_resets, device=robot.device).uniform_(-0.5, 0.5)
            default_root_state[:, 1] += torch.empty(num_resets, device=robot.device).uniform_(-0.5, 0.5)
            # Note: No Z randomization added (matches mdp.reset_root_state_uniform behavior)

            # Yaw randomization
            yaw_noise = torch.empty(num_resets, device=robot.device).uniform_(-3.14, 3.14)
            cos_yaw = torch.cos(yaw_noise / 2)
            sin_yaw = torch.sin(yaw_noise / 2)
            yaw_quat = torch.stack([cos_yaw, torch.zeros_like(cos_yaw), torch.zeros_like(cos_yaw), sin_yaw], dim=-1)
            default_quat = default_root_state[:, 3:7].clone()
            default_root_state[:, 3:7] = math_utils.quat_mul(default_quat, yaw_quat)

            # Velocity randomization
            # Dynamically fetch the velocity range for this specific robot
            vel_range_attr = f"reset_base_vel_range_{robot_name}"
            vel_range = getattr(self.cfg, vel_range_attr, (-0.5, 0.5))
            default_root_state[:, 7:10].uniform_(*vel_range)
            default_root_state[:, 10:13].uniform_(*vel_range)

            # Write state to sim
            robot.write_root_state_to_sim(default_root_state, env_ids=local_indices)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=local_indices)

    def _get_robot_body_groups(self, robot_name: str) -> dict:
        """Get body name patterns for different robot parts."""
        body_groups = {
            "anymal_d": {
                "base": ["base"],
                "hips": [".*_HIP"],  # LF_HIP, LH_HIP, RF_HIP, RH_HIP
                "thighs": [".*_THIGH"],  # LF_THIGH, LH_THIGH, RF_THIGH, RH_THIGH
                "shanks": [".*_SHANK"],  # LF_SHANK, LH_SHANK, RF_SHANK, RH_SHANK
                "feet": [".*_FOOT"],  # LF_FOOT, LH_FOOT, RF_FOOT, RH_FOOT
            },
            "anymal_c": {
                "base": ["base"],
                "hips": [".*_HIP"],
                "thighs": [".*_THIGH"],
                "shanks": [".*_SHANK"],
                "feet": [".*_FOOT"],
            },
            "anymal_b": {
                "base": ["base"],
                "hips": [".*_HIP"],
                "thighs": [".*_THIGH"],
                "shanks": [".*_SHANK"],
                "feet": [".*_FOOT"],
            },
            "unitree_a1": {
                "base": ["trunk"],
                "hips": ["[FR][LR]_hip"],  # FL_hip, FR_hip, RL_hip, RR_hip
                "thighs": ["[FR][LR]_thigh"],  # FL_thigh, FR_thigh, RL_thigh, RR_thigh
                "shanks": ["[FR][LR]_calf"],  # FL_calf, FR_calf, RL_calf, RR_calf
                "feet": ["[FR][LR]_foot"],  # FL_foot, FR_foot, RL_foot, RR_foot
            },
            "unitree_go1": {
                "base": ["trunk"],
                "hips": ["[FR][LR]_hip"],
                "thighs": ["[FR][LR]_thigh"],
                "shanks": ["[FR][LR]_calf"],
                "feet": ["[FR][LR]_foot"],
            },
            "unitree_go2": {
                "base": ["base"],
                # Go2 has Head_upper and Head_lower, so we need specific patterns for legs only
                "hips": ["[FR][LR]_hip"],  # FL_hip, FR_hip, RL_hip, RR_hip (excludes Head_*)
                "thighs": ["[FR][LR]_thigh"],  # FL_thigh, FR_thigh, RL_thigh, RR_thigh
                "shanks": ["[FR][LR]_calf"],  # FL_calf, FR_calf, RL_calf, RR_calf
                "feet": ["[FR][LR]_foot"],  # FL_foot, FR_foot, RL_foot, RR_foot
            },
            "unitree_b2": {
                "base": ["base_link"],
                # B2 has many sensor bodies, so we need specific patterns for legs only
                "hips": ["[FR][LR]_hip"],  # FL_hip, FR_hip, RL_hip, RR_hip
                "thighs": ["[FR][LR]_thigh"],  # FL_thigh, FR_thigh, RL_thigh, RR_thigh (excludes *_thigh_protect)
                "shanks": ["[FR][LR]_calf"],  # FL_calf, FR_calf, RL_calf, RR_calf (excludes *_thigh_protect)
                "feet": ["[FR][LR]_foot"],  # FL_foot, FR_foot, RL_foot, RR_foot
            },
            "spot": {
                "base": ["body"],
                # Spot uses different naming:  _hip, _uleg (upper leg), _lleg (lower leg)
                "hips": ["[fh][lr]_hip"],  # fl_hip, fr_hip, hl_hip, hr_hip
                "thighs": ["[fh][lr]_uleg"],  # fl_uleg, fr_uleg, hl_uleg, hr_uleg (upper leg)
                "shanks": ["[fh][lr]_lleg"],  # fl_lleg, fr_lleg, hl_lleg, hr_lleg (lower leg)
                "feet": ["[fh][lr]_foot"],  # fl_foot, fr_foot, hl_foot, hr_foot
            },
        }
        return body_groups.get(robot_name, {})

    def _apply_morphology_randomization(self):
        """Apply morphology randomization (Base Mass and CoM)."""
        print("[INFO - DR] Applying morphology randomization...")

        for robot_name, robot in self.robots.items():
            num_robot_envs = len(self.robot_env_ids[robot_name])
            # Use CPU for local_env_ids during initialization (will be converted to GPU in mass function)
            local_env_ids = torch.arange(num_robot_envs, device="cpu")

            # Determine if this is a large or small robot
            is_large_robot = robot_name in ["anymal_d", "anymal_c", "anymal_b", "unitree_b2", "spot"]
            base_mass_range = self.cfg.base_mass_range_large if is_large_robot else self.cfg.base_mass_range_small
            # Get body groups for this robot
            body_groups = self._get_robot_body_groups(robot_name)

            # Randomize masses for all bodies
            self._randomize_body_masses(
                robot=robot,
                robot_name=robot_name,
                num_robot_envs=num_robot_envs,
                local_env_ids=local_env_ids,
                body_groups=body_groups,
                base_mass_range=base_mass_range
            )

            # Matches manager config: x=(-0.05, 0.05), y=(-0.05, 0.05), z=(-0.01, 0.01)
            # We apply this to the BASE only.
            com_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)}

            self._randomize_body_com(
                robot=robot,
                robot_name=robot_name,
                num_robot_envs=num_robot_envs,
                local_env_ids=local_env_ids,
                body_groups=body_groups,
                com_range=com_range
            )

    def _randomize_body_com(self, robot, robot_name: str, num_robot_envs: int,
                            local_env_ids: torch.Tensor, body_groups: dict,
                            com_range: dict):
        """Randomize Center of Mass of robot base."""

        # Get current CoMs (Work on CPU to match PhysX backend requirements)
        current_coms = robot.root_physx_view.get_coms().clone()
        if current_coms.device.type != 'cpu':
            current_coms = current_coms.cpu()

        # Apply randomization only to the base link
        if "base" in body_groups:
            for body_pattern in body_groups["base"]:
                try:
                    body_ids, _ = robot.find_bodies(body_pattern)
                    for body_idx in body_ids:
                        # Sample offsets
                        offset_x = torch.empty(num_robot_envs, device="cpu").uniform_(*com_range["x"])
                        offset_y = torch.empty(num_robot_envs, device="cpu").uniform_(*com_range["y"])
                        offset_z = torch.empty(num_robot_envs, device="cpu").uniform_(*com_range["z"])

                        # Add offsets to current CoM (x, y, z)
                        current_coms[:num_robot_envs, body_idx, 0] += offset_x
                        current_coms[:num_robot_envs, body_idx, 1] += offset_y
                        current_coms[:num_robot_envs, body_idx, 2] += offset_z
                except Exception as e:
                    print(f"[Warning] Base CoM randomization: Pattern '{body_pattern}' issue in {robot_name}: {e}")

        # Ensure indices are int32 and on CPU
        if local_env_ids.device.type != 'cpu':
            local_env_ids = local_env_ids.cpu()
        local_env_ids = local_env_ids.to(dtype=torch.int32)

        # Write to physics engine
        robot.root_physx_view.set_coms(current_coms, indices=local_env_ids)
        print(f"    [{robot_name}] Randomized base CoM")

    def _randomize_body_masses(self, robot, robot_name: str, num_robot_envs: int,
                               local_env_ids: torch.Tensor, body_groups: dict,
                               base_mass_range):
        """Randomize masses of robot bodies."""

        # Get current masses and defaults (work on CPU for easier manipulation)
        current_masses = robot.root_physx_view.get_masses().clone()
        default_masses = robot.data.default_mass.clone()

        # Ensure we're working on CPU
        if current_masses.device.type != 'cpu':
            current_masses = current_masses.cpu()
        if default_masses.device.type != 'cpu':
            default_masses = default_masses.cpu()

        current_masses = current_masses.clone()
        default_masses = default_masses.clone()

        # Verify shapes match
        assert current_masses.shape[0] == num_robot_envs, f"Shape mismatch: current_masses has {current_masses.shape[0]} envs, expected {num_robot_envs}"

        # Reset to default
        current_masses[:num_robot_envs] = default_masses[:num_robot_envs]

        if "base" in body_groups:
            for body_pattern in body_groups["base"]:
                try:
                    body_ids, _ = robot.find_bodies(body_pattern)
                    for body_idx in body_ids:
                        mass_offset = torch.empty(num_robot_envs, device="cpu").uniform_(*base_mass_range)
                        current_masses[:num_robot_envs, body_idx] += mass_offset
                except Exception as e:
                    print(f"[Warning] Base mass randomization: Pattern '{body_pattern}' issue in {robot_name}: {e}")

        # Ensure masses are positive
        current_masses = torch.clamp(current_masses, min=0.01)

        # Ensure indices are on CPU with correct dtype
        if local_env_ids.device.type != 'cpu':
            local_env_ids = local_env_ids.cpu()
        local_env_ids = local_env_ids.to(dtype=torch.int32)

        # Set masses (PhysX backend expects CPU tensors)
        robot.root_physx_view.set_masses(current_masses, indices=local_env_ids)
        print(f"    [{robot_name}] Randomized body masses")
