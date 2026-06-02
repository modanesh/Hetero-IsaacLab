import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster


def get_robot_env_ids(env, cfg: SceneEntityCfg):
    """Get environment IDs for a specific robot based on asset or sensor configuration."""
    # Derive the base robot name from the asset/sensor name
    # e.g., "anymal_d_contacts" -> "anymal_d"
    robot_name = cfg.name.replace("_contacts", "")

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
        # Resolve names to indices
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

    # Sum over joints (dim=1) to get scalar reward per environment
    reward[robot_env_ids] = torch.sum(torch.square(torques), dim=1)
    return reward


def flat_orientation_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return reward


def feet_air_time(env, sensor_cfg: SceneEntityCfg, command_name: str, threshold: float) -> torch.Tensor:
    """Reward long steps taken by the feet."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    feet_ids = _resolve_body_ids(contact_sensor, sensor_cfg)

    robot_env_ids = get_robot_env_ids(env, sensor_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, feet_ids]
    last_air_time = contact_sensor.data.last_air_time[:, feet_ids]
    air_time_reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    is_moving = torch.norm(env._commands[robot_env_ids][:, :2], dim=1) > 0.1
    reward[robot_env_ids] = air_time_reward * is_moving
    return reward


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.square(asset.data.root_lin_vel_b[:, 2])
    return reward


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
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


def action_rate_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.sum(torch.square(env.actions[robot_env_ids] - env.previous_actions[robot_env_ids]), dim=1)
    return reward


def undesired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    robot_env_ids = get_robot_env_ids(env, sensor_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)

    body_ids = _resolve_body_ids(contact_sensor, sensor_cfg)

    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Check max force over history for specified bodies
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold
    reward[robot_env_ids] = torch.sum(is_contact, dim=1, dtype=torch.float32)
    return reward


def base_height_l2(env, asset_cfg: SceneEntityCfg, target_height: float, sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)

    if sensor_cfg is not None:
        # Use height scanner to adjust target height based on terrain
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height

    # Compute the L2 squared penalty
    reward[robot_env_ids] = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return reward


# ============================================================================
# Spot-specific Reward Functions
# ============================================================================

def air_time_reward(
        env,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        mode_time: float,
        velocity_threshold: float,
        command_name: str,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)

    body_ids = _resolve_body_ids(contact_sensor, sensor_cfg)

    current_air_time = contact_sensor.data.current_air_time[:, body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env._commands[robot_env_ids], dim=1).unsqueeze(dim=1).expand(-1, current_air_time.shape[1])
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, current_air_time.shape[1])
    robot_reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0.0),
        stance_cmd_reward,
    )
    reward[robot_env_ids] = torch.sum(robot_reward, dim=1)
    return reward


def base_angular_velocity_reward(env, asset_cfg: SceneEntityCfg, std: float, command_name: str) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    target = env._commands[robot_env_ids, 2]
    ang_vel_error = torch.abs(target - asset.data.root_ang_vel_b[:, 2])
    reward[robot_env_ids] = torch.exp(-ang_vel_error / std)
    return reward


def base_linear_velocity_reward(
        env,
        asset_cfg: SceneEntityCfg,
        std: float,
        command_name: str,
        ramp_at_vel: float = 1.0,
        ramp_rate: float = 0.5,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    target = env._commands[robot_env_ids, :2]
    lin_vel_error = torch.linalg.norm(target - asset.data.root_lin_vel_b[:, :2], dim=1)
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    reward[robot_env_ids] = torch.exp(-lin_vel_error / std) * velocity_scaling_multiple
    return reward


def foot_clearance_reward(
        env,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        std: float,
        tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)

    body_ids = _resolve_body_ids(asset, asset_cfg)

    foot_z_target_error = torch.square(asset.data.body_pos_w[:, body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, body_ids, :2], dim=2))
    robot_reward = foot_z_target_error * foot_velocity_tanh
    reward[robot_env_ids] = torch.exp(-torch.sum(robot_reward, dim=1) / std)
    return reward


def gait_reward(
        env,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        std: float,
        max_err: float,
        velocity_threshold: float,
        command_name: str,
        synced_feet_pair_names: list,
) -> torch.Tensor:
    """Gait enforcing reward for quadrupeds (trotting gait)."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)

    # Resolve names to indices
    # synced_feet_pair_names is e.g. [["fl_foot", "hr_foot"], ["fr_foot", "hl_foot"]]
    pair_0_ids, _ = contact_sensor.find_bodies(synced_feet_pair_names[0])
    pair_1_ids, _ = contact_sensor.find_bodies(synced_feet_pair_names[1])

    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time

    # Sync rewards for pairs that should be synchronized
    def sync_reward_func(foot_0: int, foot_1: int) -> torch.Tensor:
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err ** 2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=max_err ** 2)
        return torch.exp(-(se_air + se_contact) / std)

    # Async rewards for pairs that should be anti-synchronized
    def async_reward_func(foot_0: int, foot_1: int) -> torch.Tensor:
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err ** 2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err ** 2)
        return torch.exp(-(se_act_0 + se_act_1) / std)

    sync_reward_0 = sync_reward_func(pair_0_ids[0], pair_0_ids[1])
    sync_reward_1 = sync_reward_func(pair_1_ids[0], pair_1_ids[1])
    sync_reward = sync_reward_0 * sync_reward_1

    async_reward_0 = async_reward_func(pair_0_ids[0], pair_1_ids[0])
    async_reward_1 = async_reward_func(pair_0_ids[1], pair_1_ids[1])
    async_reward_2 = async_reward_func(pair_0_ids[0], pair_1_ids[1])
    async_reward_3 = async_reward_func(pair_1_ids[0], pair_0_ids[1])
    async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
    cmd = torch.norm(env._commands[robot_env_ids], dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    robot_reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        sync_reward * async_reward,
        torch.zeros_like(sync_reward)
    )
    reward[robot_env_ids] = robot_reward
    return reward


def action_smoothness_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.linalg.norm(env.actions[robot_env_ids] - env.previous_actions[robot_env_ids], dim=1)
    return reward


def air_time_variance_penalty(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground."""
    robot_env_ids = get_robot_env_ids(env, sensor_cfg)
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)

    body_ids = _resolve_body_ids(contact_sensor, sensor_cfg)

    last_air_time = contact_sensor.data.last_air_time[:, body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, body_ids]

    robot_reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward[robot_env_ids] = robot_reward
    return reward


def base_motion_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    robot_reward = 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )
    reward[robot_env_ids] = robot_reward
    return reward


def base_orientation_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return reward


def foot_slip_penalty(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)

    sensor_body_ids = _resolve_body_ids(contact_sensor, sensor_cfg)
    asset_body_ids = _resolve_body_ids(asset, asset_cfg)

    # Note: sensor_body_ids and asset_body_ids should align in order if referring to same feet

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_body_ids], dim=-1), dim=1)[0] > threshold

    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_body_ids, :2], dim=2)
    robot_reward = is_contact * foot_planar_velocity
    reward[robot_env_ids] = torch.sum(robot_reward, dim=1)
    return reward


def joint_acceleration_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[robot_env_ids] = torch.linalg.norm(asset.data.joint_acc, dim=1)
    return reward


def joint_position_penalty(
        env,
        asset_cfg: SceneEntityCfg,
        stand_still_scale: float,
        velocity_threshold: float,
        command_name: str,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    cmd = torch.linalg.norm(env._commands[robot_env_ids], dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    ids = _resolve_joint_ids(asset, asset_cfg)

    robot_reward = torch.linalg.norm(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids], dim=1)
    robot_reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        robot_reward,
        stand_still_scale * robot_reward
    )
    reward[robot_env_ids] = robot_reward
    return reward


def joint_torques_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)

    ids = _resolve_joint_ids(asset, asset_cfg)

    reward[robot_env_ids] = torch.linalg.norm(asset.data.applied_torque[:, ids], dim=1)
    return reward


def joint_velocity_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)

    ids = _resolve_joint_ids(asset, asset_cfg)

    reward[robot_env_ids] = torch.linalg.norm(asset.data.joint_vel[:, ids], dim=1)
    return reward
