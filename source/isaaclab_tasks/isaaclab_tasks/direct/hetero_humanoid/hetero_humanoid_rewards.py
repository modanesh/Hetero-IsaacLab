# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster


def get_robot_env_ids(env, cfg: SceneEntityCfg):
    """Get environment IDs for a specific robot based on asset or sensor configuration."""
    robot_name = cfg.name.replace("_contacts", "").replace("_scanner", "")

    if robot_name not in env.robot_env_ids:
        raise ValueError(f"Unknown asset or sensor: {cfg.name}. Corresponding robot '{robot_name}' not found in env.robot_env_ids.")

    return env.robot_env_ids[robot_name]


def _resolve_joint_ids(asset, cfg: SceneEntityCfg):
    """Resolve joint IDs from config, defaulting to all if None."""
    if cfg.joint_ids is None:
        return slice(None)
    return cfg.joint_ids


def _resolve_body_ids(asset, cfg: SceneEntityCfg):
    """Resolve body IDs from config, resolving regex if indices are missing."""
    if cfg.body_ids is not None:
        return cfg.body_ids
    if cfg.body_names is not None:
        ids, _ = asset.find_bodies(cfg.body_names)
        return ids
    return slice(None)


def track_lin_vel_xy_exp(env, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    commands = env._commands[robot_env_ids]
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    reward[robot_env_ids] = torch.exp(-lin_vel_error / std ** 2)
    return reward


def track_ang_vel_z_exp(env, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    commands = env._commands[robot_env_ids]
    ang_vel_error = torch.square(commands[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward[robot_env_ids] = torch.exp(-ang_vel_error / std ** 2)
    return reward


def joint_torques_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)

    ids = _resolve_joint_ids(asset, asset_cfg)
    torques = asset.data.applied_torque[:, ids]
    reward[robot_env_ids] = torch.sum(torch.square(torques), dim=1)
    return reward


def flat_orientation_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return reward


def feet_air_time_biped(env, sensor_cfg: SceneEntityCfg, command_name: str, threshold: float) -> torch.Tensor:
    """Reward long steps taken by the biped feet (Enforces single-stance walking)."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    feet_ids = _resolve_body_ids(contact_sensor, sensor_cfg)
    robot_env_ids = get_robot_env_ids(env, sensor_cfg)
    
    air_time = contact_sensor.data.current_air_time[:, feet_ids]
    contact_time = contact_sensor.data.current_contact_time[:, feet_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
    reward = torch.zeros(env.num_envs, device=env.device)
    robot_reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    robot_reward = torch.clamp(robot_reward, max=threshold)
    
    is_moving = torch.norm(env._commands[robot_env_ids, :2], dim=1) > 0.1
    robot_reward *= is_moving

    reward[robot_env_ids] = robot_reward
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet sliding when in contact with the ground."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    sensor_feet_ids = _resolve_body_ids(contact_sensor, sensor_cfg)
    
    if sensor_cfg.body_names is not None:
        asset_feet_ids, _ = asset.find_bodies(sensor_cfg.body_names)
    else:
        asset_feet_ids = sensor_feet_ids

    robot_env_ids = get_robot_env_ids(env, sensor_cfg)

    reward = torch.zeros(env.num_envs, device=env.device)
    in_contact = contact_sensor.data.net_forces_w[:, sensor_feet_ids, 2] > 1.0
    feet_vel = torch.norm(asset.data.body_lin_vel_w[:, asset_feet_ids, :2], dim=-1)
    reward[robot_env_ids] = torch.sum(feet_vel * in_contact, dim=1)
    return reward


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.square(asset.data.root_lin_vel_b[:, 2])
    return reward


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize pitch and roll base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return reward


def joint_acc_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)

    ids = _resolve_joint_ids(asset, asset_cfg)
    acc = asset.data.joint_acc[:, ids]
    reward[robot_env_ids] = torch.sum(torch.square(acc), dim=1)
    return reward


def action_rate_l2(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Penalize changes in actions (action rate) across consecutive steps."""
    reward = torch.sum(torch.square(env.actions - env.previous_actions), dim=1)
    if asset_cfg is not None:
        robot_env_ids = get_robot_env_ids(env, asset_cfg)
        mask = torch.zeros(env.num_envs, device=env.device)
        mask[robot_env_ids] = 1.0
        reward = reward * mask
    return reward


def joint_pos_limits(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint positions that move close to their soft joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)

    ids = _resolve_joint_ids(asset, asset_cfg)
    pos = asset.data.joint_pos[:, ids]
    limits = asset.data.soft_joint_pos_limits[:, ids]

    out_of_limits = (pos < limits[..., 0]) | (pos > limits[..., 1])
    reward[robot_env_ids] = torch.sum(out_of_limits.float(), dim=1)
    return reward


def joint_deviation_l1(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from nominal default positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)

    ids = _resolve_joint_ids(asset, asset_cfg)
    diff = torch.abs(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids])
    reward[robot_env_ids] = torch.sum(diff, dim=1)
    return reward


def no_fly(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the robot if all specified feet are off the ground at the same time."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    feet_ids = _resolve_body_ids(contact_sensor, sensor_cfg)
    robot_env_ids = get_robot_env_ids(env, sensor_cfg)
    
    in_contact = contact_sensor.data.net_forces_w[:, feet_ids, 2] > 1.0
    has_contact = torch.any(in_contact, dim=-1)
    
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = (~has_contact).float()
    return reward


def is_terminated(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize termination for a specific robot."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = env.reset_terminated[robot_env_ids].float()
    return reward


def stand_still_joint_deviation(env, command_name: str, command_threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset = env.scene[asset_cfg.name]
    ids = _resolve_joint_ids(asset, asset_cfg)
    
    diff = torch.abs(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids])
    deviation = torch.sum(diff, dim=1)
    
    is_standing = torch.norm(env._commands[robot_env_ids, :2], dim=1) < command_threshold

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = deviation * is_standing
    return reward


def undesired_contacts(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    robot_env_ids = get_robot_env_ids(env, sensor_cfg)
    body_ids = _resolve_body_ids(contact_sensor, sensor_cfg)
    
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold
    
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.sum(is_contact, dim=1).float()[robot_env_ids]
    return reward


def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    robot_env_ids = get_robot_env_ids(env, sensor_cfg)
    body_ids = _resolve_body_ids(contact_sensor, sensor_cfg)
    
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = zero_contact.float()[robot_env_ids]
    return reward
