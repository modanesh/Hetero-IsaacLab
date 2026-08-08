# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import List, Optional

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks.direct.hetero_humanoid.hetero_humanoid_rewards as custom_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# Humanoid Robot Assets
from isaaclab_assets.robots.cassie import CASSIE_CFG
from isaaclab_assets.robots.agility import DIGIT_V4_CFG, LEG_JOINT_NAMES
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG, H1_MINIMAL_CFG



##
# Custom Configuration Classes for Heterogeneous Setups
##

@configclass
class HeterogeneousRobotCfg(ArticulationCfg):
    """Configuration for a robot asset in a heterogeneous scene."""
    env_ids: Optional[List[int]] = None
    """List of environment IDs this robot is present in."""


@configclass
class HeterogeneousSensorCfg(ContactSensorCfg):
    """Configuration for a sensor asset in a heterogeneous scene."""
    env_ids: Optional[List[int]] = None
    """List of environment IDs this sensor is present in."""


@configclass
class HeterogeneousRayCasterCfg(RayCasterCfg):
    """Configuration for a raycaster asset in a heterogeneous scene."""
    env_ids: Optional[List[int]] = None
    """List of environment IDs this sensor is present in."""


def get_raycaster_cfg(prim_path_pattern: str) -> HeterogeneousRayCasterCfg:
    """Helper to generate standard grid height scanners for each robot."""
    return HeterogeneousRayCasterCfg(
        prim_path=prim_path_pattern,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


##
# Base Scene definition
##

@configclass
class HeterogeneousHumanoidSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with multiple humanoid robots."""
    replicate_physics: bool = False

    # ground terrain for flat env
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Humanoid Robots
    cassie: HeterogeneousRobotCfg = CASSIE_CFG.replace(prim_path="{ENV_REGEX_NS}/cassie")
    cassie.spawn.activate_contact_sensors = True

    digit: HeterogeneousRobotCfg = DIGIT_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/digit")
    digit.spawn.activate_contact_sensors = True

    g1: HeterogeneousRobotCfg = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/g1")
    g1.spawn.activate_contact_sensors = True

    h1: HeterogeneousRobotCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/h1")
    h1.spawn.activate_contact_sensors = True

    # Contact Sensors
    cassie_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/cassie/.*", history_length=3, track_air_time=True)
    digit_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/digit/.*", history_length=3, track_air_time=True)
    g1_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/g1/.*", history_length=3, track_air_time=True)
    h1_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/h1/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Rough Scene definition (Adds Rough Terrain Generator & RayCasters)
##

@configclass
class HeterogeneousHumanoidRoughSceneCfg(HeterogeneousHumanoidSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    cassie_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/cassie/pelvis")
    digit_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/digit/torso_base")
    g1_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/g1/torso_link")
    h1_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/h1/torso_link")



##
# Observation Noise Configuration
##

@configclass
class ObservationNoiseCfg:
    """Configuration for observation noise."""
    enabled: bool = True
    lin_vel_noise: tuple = (-0.1, 0.1)
    ang_vel_noise: tuple = (-0.2, 0.2)
    projected_gravity_noise: tuple = (-0.05, 0.05)
    joint_pos_noise: tuple = (-0.01, 0.01)
    joint_vel_noise: tuple = (-1.5, 1.5)
    height_measurement_noise: tuple = (-0.1, 0.1)


##
# Reward Configuration
##

@configclass
class RewardsCfg:
    # --- CASSIE REWARDS ---
    track_lin_vel_xy_exp_cassie = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("cassie")}
    )
    track_ang_vel_z_exp_cassie = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("cassie")}
    )
    lin_vel_z_l2_cassie = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("cassie")})
    ang_vel_xy_l2_cassie = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("cassie")})
    dof_torques_l2_cassie = RewTerm(func=custom_mdp.joint_torques_l2, weight=-5.0e-6, params={"asset_cfg": SceneEntityCfg("cassie")})
    dof_acc_l2_cassie = RewTerm(func=custom_mdp.joint_acc_l2, weight=-3.75e-7, params={"asset_cfg": SceneEntityCfg("cassie")})
    action_rate_l2_cassie = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.015, params={"asset_cfg": SceneEntityCfg("cassie")})
    feet_air_time_cassie = RewTerm(
        func=custom_mdp.feet_air_time_biped, weight=5.0,
        params={"sensor_cfg": SceneEntityCfg("cassie_contacts", body_names=".*toe"), "command_name": "base_velocity", "threshold": 0.3}
    )
    flat_orientation_l2_cassie = RewTerm(func=custom_mdp.flat_orientation_l2, weight=-2.5, params={"asset_cfg": SceneEntityCfg("cassie")})
    dof_pos_limits_cassie = RewTerm(
        func=custom_mdp.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("cassie", joint_names=["toe_joint_.*"])}
    )
    joint_deviation_hip_cassie = RewTerm(
        func=custom_mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("cassie", joint_names=["hip_rotation_.*"])}
    )
    joint_deviation_toes_cassie = RewTerm(
        func=custom_mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("cassie", joint_names=["toe_joint_.*"])}
    )
    termination_penalty_cassie = RewTerm(
        func=custom_mdp.is_terminated, weight=-200.0,
        params={"asset_cfg": SceneEntityCfg("cassie")}
    )


    # --- DIGIT REWARDS ---
    track_lin_vel_xy_exp_digit = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("digit")}
    )
    track_ang_vel_z_exp_digit = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("digit")}
    )
    lin_vel_z_l2_digit = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("digit")})
    ang_vel_xy_l2_digit = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("digit")})
    dof_torques_l2_digit = RewTerm(func=custom_mdp.joint_torques_l2, weight=-1.0e-6, params={"asset_cfg": SceneEntityCfg("digit")})
    dof_acc_l2_digit = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.0e-7, params={"asset_cfg": SceneEntityCfg("digit", joint_names=[".*_hip_roll", ".*_hip_yaw", ".*_hip_pitch", ".*_knee", ".*_toe_a", ".*_toe_b", ".*_arm_.*"])})
    action_rate_l2_digit = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.008, params={"asset_cfg": SceneEntityCfg("digit")})
    feet_air_time_digit = RewTerm(
        func=custom_mdp.feet_air_time_biped, weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("digit_contacts", body_names=".*_leg_toe_roll"), "command_name": "base_velocity", "threshold": 0.8}
    )
    flat_orientation_l2_digit = RewTerm(func=custom_mdp.flat_orientation_l2, weight=-2.5, params={"asset_cfg": SceneEntityCfg("digit")})
    feet_slide_digit = RewTerm(func=custom_mdp.feet_slide, weight=-0.25, params={"sensor_cfg": SceneEntityCfg("digit_contacts", body_names=".*_leg_toe_roll"), "asset_cfg": SceneEntityCfg("digit", body_names=".*_leg_toe_roll")})
    dof_pos_limits_digit = RewTerm(func=custom_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("digit", joint_names=[".*_leg_toe_roll", ".*_leg_toe_pitch", ".*_tarsus"])})
    stand_still_digit = RewTerm(
        func=custom_mdp.stand_still_joint_deviation, weight=-0.4,
        params={"command_name": "base_velocity", "command_threshold": 0.06, "asset_cfg": SceneEntityCfg("digit", joint_names=LEG_JOINT_NAMES)}
    )
    no_jumps_digit = RewTerm(
        func=custom_mdp.desired_contacts, weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("digit_contacts", body_names=[".*_leg_toe_roll"])}
    )
    undesired_contacts_digit = RewTerm(
        func=custom_mdp.undesired_contacts, weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("digit_contacts", body_names=[".*_rod", ".*_tarsus"]), "threshold": 1.0}
    )
    termination_penalty_digit = RewTerm(
        func=custom_mdp.is_terminated, weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("digit")}
    )
    joint_deviation_hip_roll_digit = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("digit", joint_names=".*_leg_hip_roll")})
    joint_deviation_hip_yaw_digit = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("digit", joint_names=".*_leg_hip_yaw")})
    joint_deviation_knee_digit = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("digit", joint_names=".*_tarsus")})
    joint_deviation_feet_digit = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("digit", joint_names=[".*_toe_a", ".*_toe_b"])})
    joint_deviation_arms_digit = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("digit", joint_names=".*_arm_.*")})


    # --- G1 REWARDS ---
    track_lin_vel_xy_exp_g1 = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("g1")}
    )
    track_ang_vel_z_exp_g1 = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("g1")}
    )
    lin_vel_z_l2_g1 = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-0.2, params={"asset_cfg": SceneEntityCfg("g1")})
    ang_vel_xy_l2_g1 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("g1")})
    dof_torques_l2_g1 = RewTerm(func=custom_mdp.joint_torques_l2, weight=-2.0e-6, params={"asset_cfg": SceneEntityCfg("g1", joint_names=[".*_hip_.*", ".*_knee_joint"])})
    dof_acc_l2_g1 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": SceneEntityCfg("g1", joint_names=[".*_hip_.*", ".*_knee_joint"])})
    action_rate_l2_g1 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.005, params={"asset_cfg": SceneEntityCfg("g1")})
    feet_air_time_g1 = RewTerm(
        func=custom_mdp.feet_air_time_biped, weight=0.75,
        params={"sensor_cfg": SceneEntityCfg("g1_contacts", body_names=".*_ankle_roll_link"), "command_name": "base_velocity", "threshold": 0.4}
    )
    flat_orientation_l2_g1 = RewTerm(func=custom_mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("g1")})
    feet_slide_g1 = RewTerm(func=custom_mdp.feet_slide, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("g1_contacts", body_names=".*_ankle_roll_link"), "asset_cfg": SceneEntityCfg("g1", body_names=".*_ankle_roll_link")})
    dof_pos_limits_g1 = RewTerm(func=custom_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("g1", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])})
    termination_penalty_g1 = RewTerm(func=custom_mdp.is_terminated, weight=-200.0, params={"asset_cfg": SceneEntityCfg("g1")})
    joint_deviation_hip_g1 = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("g1", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])})
    joint_deviation_arms_g1 = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("g1", joint_names=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_pitch_joint", ".*_elbow_roll_joint"])})
    joint_deviation_fingers_g1 = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("g1", joint_names=[".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint", ".*_zero_joint", ".*_one_joint", ".*_two_joint"])})
    joint_deviation_torso_g1 = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("g1", joint_names="torso_joint")})


    # --- H1 REWARDS ---
    track_lin_vel_xy_exp_h1 = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("h1")}
    )
    track_ang_vel_z_exp_h1 = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("h1")}
    )
    ang_vel_xy_l2_h1 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("h1")})
    dof_torques_l2_h1 = RewTerm(func=custom_mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("h1")})
    dof_acc_l2_h1 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-1.25e-7, params={"asset_cfg": SceneEntityCfg("h1")})
    action_rate_l2_h1 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.005, params={"asset_cfg": SceneEntityCfg("h1")})
    feet_air_time_h1 = RewTerm(
        func=custom_mdp.feet_air_time_biped, weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("h1_contacts", body_names=".*ankle_link"), "command_name": "base_velocity", "threshold": 0.6}
    )
    flat_orientation_l2_h1 = RewTerm(func=custom_mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("h1")})
    feet_slide_h1 = RewTerm(func=custom_mdp.feet_slide, weight=-0.25, params={"sensor_cfg": SceneEntityCfg("h1_contacts", body_names=".*ankle_link"), "asset_cfg": SceneEntityCfg("h1", body_names=".*ankle_link")})
    dof_pos_limits_h1 = RewTerm(func=custom_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("h1", joint_names=".*_ankle")})
    termination_penalty_h1 = RewTerm(func=custom_mdp.is_terminated, weight=-200.0, params={"asset_cfg": SceneEntityCfg("h1")})
    joint_deviation_hip_h1 = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("h1", joint_names=[".*_hip_yaw", ".*_hip_roll"])})
    joint_deviation_arms_h1 = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("h1", joint_names=[".*_shoulder_.*", ".*_elbow"])})
    joint_deviation_torso_h1 = RewTerm(func=custom_mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("h1", joint_names="torso")})



@configclass
class HeterogeneousHumanoidVelocityEnvCfg(DirectRLEnvCfg):
    """Base Configuration for heterogeneous-humanoid velocity-tracking environment."""
    humanoids: list[str] = ["cassie", "digit", "g1", "h1"]
    
    episode_length_s = 20.0
    decimation = 4
    # Set to max needed for any humanoid. Can be overridden.
    num_actions = 64
    # 12 base + 3 * num_actions (joint_pos, joint_vel, actions)
    num_observations = 12 + (3 * num_actions)
    num_states = 0

    include_height_scanners: bool = False
    height_scanner_offset: float = 0.8
    contact_threshold: float = 1.0


    action_scale: float = 0.5
    action_scale_cassie: float = 0.5
    action_scale_digit: float = 0.5
    action_scale_g1: float = 0.25
    action_scale_h1: float = 0.5

    action_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(num_actions,))



    observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(num_observations,), dtype=float)
    state_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(num_observations,), dtype=float)

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: HeterogeneousHumanoidSceneCfg = HeterogeneousHumanoidSceneCfg(num_envs=4096, env_spacing=4.0)

    domain_randomization: bool = True
    observation_noise: ObservationNoiseCfg = ObservationNoiseCfg()

    base_mass_range_large: tuple = (-5.0, 5.0)
    base_mass_range_small: tuple = (-2.0, 2.0)

    reset_base_vel_range_cassie: tuple = (0.0, 0.0)
    reset_base_vel_range_digit: tuple = (-0.5, 0.5)
    reset_base_vel_range_g1: tuple = (0.0, 0.0)
    reset_base_vel_range_h1: tuple = (0.0, 0.0)

    reset_joint_pos_mode_cassie: str = "scale"
    reset_joint_pos_range_cassie: tuple = (1.0, 1.0)

    reset_joint_pos_mode_digit: str = "scale"
    reset_joint_pos_range_digit: tuple = (1.0, 1.0)

    reset_joint_pos_mode_g1: str = "scale"
    reset_joint_pos_range_g1: tuple = (1.0, 1.0)

    reset_joint_pos_mode_h1: str = "scale"
    reset_joint_pos_range_h1: tuple = (1.0, 1.0)

    resampling_time_range: tuple = (10.0, 10.0)
    standing_probability: float = 0.1
    heading_control_stiffness: float = 0.5
    heading_mode_probability: float = 0.0

    command_ranges_default: dict = {
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-1.0, 1.0),
    }

    rewards: RewardsCfg = RewardsCfg()


@configclass
class HeterogeneousHumanoidFlatEnvCfg(HeterogeneousHumanoidVelocityEnvCfg):
    pass


@configclass
class HeterogeneousHumanoidFlatEnvCfg_PLAY(HeterogeneousHumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 60.0
        self.domain_randomization = False
        self.observation_noise.enabled = False


@configclass
class HeterogeneousHumanoidRoughEnvCfg(HeterogeneousHumanoidVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene = HeterogeneousHumanoidRoughSceneCfg(num_envs=4096, env_spacing=4.0)
        self.include_height_scanners = True
        self.height_scanner_offset = 0.8
        self.num_observations = 12 + (3 * self.num_actions)
        if getattr(self, "include_height_scanners", False):
            self.num_observations += 187
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.num_observations,), dtype=float
        )


@configclass
class HeterogeneousHumanoidRoughEnvCfg_PLAY(HeterogeneousHumanoidRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 60.0
        self.domain_randomization = False
        self.observation_noise.enabled = False

