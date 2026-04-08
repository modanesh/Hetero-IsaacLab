# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import List, Optional

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks.direct.hetero_quadruped.hetero_quadruped_rewards as custom_mdp
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
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG, ANYMAL_C_CFG, ANYMAL_B_CFG
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG, UNITREE_B2_CFG


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
# Base Scene definition (Blind / No Scanners)
##


@configclass
class HeterogeneousQuadrupedSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with multiple legged robots."""
    replicate_physics: bool = False

    # ground terrain
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

    # Robots
    anymal_d: HeterogeneousRobotCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/anymal_d")
    anymal_d.spawn.activate_contact_sensors = True
    anymal_c: HeterogeneousRobotCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/anymal_c")
    anymal_c.spawn.activate_contact_sensors = True
    anymal_b: HeterogeneousRobotCfg = ANYMAL_B_CFG.replace(prim_path="{ENV_REGEX_NS}/anymal_b")
    anymal_b.spawn.activate_contact_sensors = True

    unitree_a1: HeterogeneousRobotCfg = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/unitree_a1")
    unitree_a1.spawn.activate_contact_sensors = True
    unitree_go1: HeterogeneousRobotCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/unitree_go1")
    unitree_go1.spawn.activate_contact_sensors = True
    unitree_go2: HeterogeneousRobotCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/unitree_go2")
    unitree_go2.spawn.activate_contact_sensors = True
    unitree_b2: HeterogeneousRobotCfg = UNITREE_B2_CFG.replace(prim_path="{ENV_REGEX_NS}/unitree_b2")
    unitree_b2.spawn.activate_contact_sensors = True

    spot: HeterogeneousRobotCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/spot")
    spot.spawn.activate_contact_sensors = True

    # Contact Sensors
    anymal_d_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/anymal_d/.*", history_length=3, track_air_time=True)
    anymal_c_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/anymal_c/.*", history_length=3, track_air_time=True)
    anymal_b_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/anymal_b/.*", history_length=3, track_air_time=True)
    unitree_a1_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/unitree_a1/.*", history_length=3, track_air_time=True)
    unitree_go1_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/unitree_go1/.*", history_length=3, track_air_time=True)
    unitree_go2_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/unitree_go2/.*", history_length=3, track_air_time=True)
    unitree_b2_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/unitree_b2/.*", history_length=3, track_air_time=True)
    spot_contacts: HeterogeneousSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/spot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Rough Scene definition (Adds RayCasters)
##
@configclass
class HeterogeneousQuadrupedRoughSceneCfg(HeterogeneousQuadrupedSceneCfg):
    # Raycasters (height scanners)
    anymal_d_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/anymal_d/base")
    anymal_c_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/anymal_c/base")
    anymal_b_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/anymal_b/base")
    unitree_a1_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/unitree_a1/trunk")
    unitree_go1_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/unitree_go1/trunk")
    unitree_go2_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/unitree_go2/base")
    unitree_b2_scanner = get_raycaster_cfg("{ENV_REGEX_NS}/unitree_b2/base_link")

##
# Observation Noise Configuration
##

@configclass
class ObservationNoiseCfg:
    """Configuration for observation noise (matching manager-based locomotion)."""
    # Enable/disable observation noise
    enabled: bool = True

    # Noise ranges (min, max) for additive uniform noise
    lin_vel_noise: tuple = (-0.1, 0.1)
    ang_vel_noise: tuple = (-0.2, 0.2)
    projected_gravity_noise: tuple = (-0.05, 0.05)
    joint_pos_noise: tuple = (-0.01, 0.01)
    joint_vel_noise: tuple = (-1.5, 1.5)
    # No noise on commands and actions (same as manager-based)
    height_measurement_noise: tuple = (-0.1, 0.1)


##
# Reward configuration
##

@configclass
class RewardsCfg:
    # Anymal-D Specific Rewards
    track_lin_vel_xy_exp_anymal_d = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("anymal_d")}
    )
    track_ang_vel_z_exp_anymal_d = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("anymal_d")}
    )
    dof_torques_l2_anymal_d = RewTerm(
        func=custom_mdp.joint_torques_l2, weight=-2.5e-5, params={"asset_cfg": SceneEntityCfg("anymal_d")}
    )
    flat_orientation_l2_anymal_d = RewTerm(
        func=custom_mdp.flat_orientation_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("anymal_d")}
    )
    feet_air_time_anymal_d = RewTerm(
        func=custom_mdp.feet_air_time, weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("anymal_d_contacts", body_names=".*_FOOT"),
            "command_name": "base_velocity", "threshold": 0.5
        }
    )

    # Anymal-C Specific Rewards
    track_lin_vel_xy_exp_anymal_c = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("anymal_c")}
    )
    track_ang_vel_z_exp_anymal_c = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("anymal_c")}
    )
    dof_torques_l2_anymal_c = RewTerm(
        func=custom_mdp.joint_torques_l2, weight=-2.5e-5, params={"asset_cfg": SceneEntityCfg("anymal_c")}
    )
    flat_orientation_l2_anymal_c = RewTerm(
        func=custom_mdp.flat_orientation_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("anymal_c")}
    )
    feet_air_time_anymal_c = RewTerm(
        func=custom_mdp.feet_air_time, weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("anymal_c_contacts", body_names=".*_FOOT"),
            "command_name": "base_velocity", "threshold": 0.5
        }
    )

    # Anymal-B Specific Rewards
    track_lin_vel_xy_exp_anymal_b = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("anymal_b")}
    )
    track_ang_vel_z_exp_anymal_b = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("anymal_b")}
    )
    dof_torques_l2_anymal_b = RewTerm(
        func=custom_mdp.joint_torques_l2, weight=-2.5e-5, params={"asset_cfg": SceneEntityCfg("anymal_b")}
    )
    flat_orientation_l2_anymal_b = RewTerm(
        func=custom_mdp.flat_orientation_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("anymal_b")}
    )
    feet_air_time_anymal_b = RewTerm(
        func=custom_mdp.feet_air_time, weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("anymal_b_contacts", body_names=".*_FOOT"),
            "command_name": "base_velocity", "threshold": 0.5
        }
    )

    # Unitree-A1 Specific Rewards
    track_lin_vel_xy_exp_unitree_a1 = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_a1")}
    )
    track_ang_vel_z_exp_unitree_a1 = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_a1")}
    )
    dof_torques_l2_unitree_a1 = RewTerm(
        func=custom_mdp.joint_torques_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("unitree_a1")}
    )
    flat_orientation_l2_unitree_a1 = RewTerm(
        func=custom_mdp.flat_orientation_l2, weight=-2.5, params={"asset_cfg": SceneEntityCfg("unitree_a1")}
    )
    feet_air_time_unitree_a1 = RewTerm(
        func=custom_mdp.feet_air_time, weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("unitree_a1_contacts", body_names=".*_foot"),
            "command_name": "base_velocity", "threshold": 0.5
        }
    )

    # Unitree-Go1 Specific Rewards
    track_lin_vel_xy_exp_unitree_go1 = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_go1")}
    )
    track_ang_vel_z_exp_unitree_go1 = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_go1")}
    )
    dof_torques_l2_unitree_go1 = RewTerm(
        func=custom_mdp.joint_torques_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("unitree_go1")}
    )
    flat_orientation_l2_unitree_go1 = RewTerm(
        func=custom_mdp.flat_orientation_l2, weight=-2.5, params={"asset_cfg": SceneEntityCfg("unitree_go1")}
    )
    feet_air_time_unitree_go1 = RewTerm(
        func=custom_mdp.feet_air_time, weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("unitree_go1_contacts", body_names=".*_foot"),
            "command_name": "base_velocity", "threshold": 0.5
        }
    )

    # Unitree-Go2 Specific Rewards
    track_lin_vel_xy_exp_unitree_go2 = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_go2")}
    )
    track_ang_vel_z_exp_unitree_go2 = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_go2")}
    )
    dof_torques_l2_unitree_go2 = RewTerm(
        func=custom_mdp.joint_torques_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("unitree_go2")}
    )
    flat_orientation_l2_unitree_go2 = RewTerm(
        func=custom_mdp.flat_orientation_l2, weight=-2.5, params={"asset_cfg": SceneEntityCfg("unitree_go2")}
    )
    feet_air_time_unitree_go2 = RewTerm(
        func=custom_mdp.feet_air_time, weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("unitree_go2_contacts", body_names=".*_foot"),
            "command_name": "base_velocity", "threshold": 0.5
        }
    )

    # Unitree-B2 Specific Rewards
    track_lin_vel_xy_exp_unitree_b2 = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_b2")}
    )
    track_ang_vel_z_exp_unitree_b2 = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp, weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_b2")}
    )
    dof_torques_l2_unitree_b2 = RewTerm(
        func=custom_mdp.joint_torques_l2, weight=-2.0e-5, params={"asset_cfg": SceneEntityCfg("unitree_b2")}
    )
    flat_orientation_l2_unitree_b2 = RewTerm(
        func=custom_mdp.flat_orientation_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("unitree_b2")}
    )
    feet_air_time_unitree_b2 = RewTerm(
        func=custom_mdp.feet_air_time, weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("unitree_b2_contacts", body_names=".*_calf"),
            "command_name": "base_velocity", "threshold": 0.5
        }
    )

    # Spot Specific Rewards
    air_time_spot = RewTerm(
        func=custom_mdp.air_time_reward, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("spot"), "sensor_cfg": SceneEntityCfg("spot_contacts", body_names=[".*_foot"]), "mode_time": 0.3,
                "velocity_threshold": 0.5, "command_name": "base_velocity"}
    )
    base_angular_velocity_spot = RewTerm(
        func=custom_mdp.base_angular_velocity_reward, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("spot"), "std": 2.0, "command_name": "base_velocity"}
    )
    base_linear_velocity_spot = RewTerm(
        func=custom_mdp.base_linear_velocity_reward, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("spot"), "std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "command_name": "base_velocity"}
    )
    foot_clearance_spot = RewTerm(
        func=custom_mdp.foot_clearance_reward, weight=0.5,
        params={"asset_cfg": SceneEntityCfg("spot", body_names=[".*_foot"]), "std": 0.05, "tanh_mult": 2.0, "target_height": 0.1}
    )
    gait_spot = RewTerm(
        func=custom_mdp.gait_reward, weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("spot"), "sensor_cfg": SceneEntityCfg("spot_contacts"),
            "std": 0.1, "max_err": 0.2, "velocity_threshold": 0.5, "command_name": "base_velocity",
            "synced_feet_pair_names": [["fl_foot", "hr_foot"], ["fr_foot", "hl_foot"]]
        }
    )

    # Common penalties
    lin_vel_z_l2_anymal_d = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("anymal_d")})
    ang_vel_xy_l2_anymal_d = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("anymal_d")})
    dof_acc_l2_anymal_d = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("anymal_d")})
    action_rate_l2_anymal_d = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("anymal_d")})
    undesired_contacts_anymal_d = RewTerm(
        func=custom_mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("anymal_d_contacts", body_names=".*_THIGH"), "threshold": 1.0}
    )

    lin_vel_z_l2_anymal_c = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("anymal_c")})
    ang_vel_xy_l2_anymal_c = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("anymal_c")})
    dof_acc_l2_anymal_c = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("anymal_c")})
    action_rate_l2_anymal_c = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("anymal_c")})
    undesired_contacts_anymal_c = RewTerm(
        func=custom_mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("anymal_c_contacts", body_names=".*_THIGH"), "threshold": 1.0}
    )

    lin_vel_z_l2_anymal_b = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("anymal_b")})
    ang_vel_xy_l2_anymal_b = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("anymal_b")})
    dof_acc_l2_anymal_b = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("anymal_b")})
    action_rate_l2_anymal_b = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("anymal_b")})
    undesired_contacts_anymal_b = RewTerm(
        func=custom_mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("anymal_b_contacts", body_names=".*_THIGH"), "threshold": 1.0}
    )

    lin_vel_z_l2_unitree_a1 = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("unitree_a1")})
    ang_vel_xy_l2_unitree_a1 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("unitree_a1")})
    dof_acc_l2_unitree_a1 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("unitree_a1")})
    action_rate_l2_unitree_a1 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("unitree_a1")})
    undesired_contacts_unitree_a1: Optional[RewTerm] = None  # Disabled for unitree_a1

    lin_vel_z_l2_unitree_go1 = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("unitree_go1")})
    ang_vel_xy_l2_unitree_go1 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("unitree_go1")})
    dof_acc_l2_unitree_go1 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("unitree_go1")})
    action_rate_l2_unitree_go1 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("unitree_go1")})
    undesired_contacts_unitree_go1: Optional[RewTerm] = None  # Disabled for unitree_go1

    lin_vel_z_l2_unitree_go2 = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
    ang_vel_xy_l2_unitree_go2 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
    dof_acc_l2_unitree_go2 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
    action_rate_l2_unitree_go2 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
    undesired_contacts_unitree_go2: Optional[RewTerm] = None  # Disabled for unitree_go2

    lin_vel_z_l2_unitree_b2 = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("unitree_b2")})
    ang_vel_xy_l2_unitree_b2 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("unitree_b2")})
    dof_acc_l2_unitree_b2 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("unitree_b2")})
    action_rate_l2_unitree_b2 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("unitree_b2")})
    undesired_contacts_unitree_b2: Optional[RewTerm] = None  # Disabled for unitree_b2

    action_smoothness_spot = RewTerm(func=custom_mdp.action_smoothness_penalty, weight=-1.0, params={"asset_cfg": SceneEntityCfg("spot")})
    air_time_variance_spot = RewTerm(
        func=custom_mdp.air_time_variance_penalty, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("spot_contacts", body_names=[".*_foot"])}
    )
    base_motion_spot = RewTerm(func=custom_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("spot")})
    base_orientation_spot = RewTerm(func=custom_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("spot")})
    foot_slip_spot = RewTerm(
        func=custom_mdp.foot_slip_penalty, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("spot", body_names=[".*_foot"]), "sensor_cfg": SceneEntityCfg("spot_contacts", body_names=[".*_foot"]),
                "threshold": 1.0}
    )
    joint_acc_spot = RewTerm(func=custom_mdp.joint_acceleration_penalty, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("spot")})
    joint_pos_spot = RewTerm(
        func=custom_mdp.joint_position_penalty, weight=-0.7,
        params={"asset_cfg": SceneEntityCfg("spot"), "stand_still_scale": 5.0, "velocity_threshold": 0.5, "command_name": "base_velocity"}
    )
    joint_torques_spot = RewTerm(func=custom_mdp.joint_torques_penalty, weight=-0.0005, params={"asset_cfg": SceneEntityCfg("spot")})
    joint_vel_spot = RewTerm(func=custom_mdp.joint_velocity_penalty, weight=-0.01, params={"asset_cfg": SceneEntityCfg("spot")})
    undesired_contacts_spot = RewTerm(
        func=custom_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("spot_contacts", body_names=[".*_lleg", ".*_uleg"]),
            "threshold": 1.0
        }
    )


##
# Environment configuration
##


@configclass
class HeterogeneousQuadrupedVelocityEnvCfg(DirectRLEnvCfg):
    """Base Configuration for heterogeneous-quadruped velocity-tracking environment (Blind)."""

    # Basic environment settings
    episode_length_s = 20.0
    decimation = 4
    num_actions = 12
    num_observations = 48
    num_states = 0

    include_height_scanners: bool = False
    height_scanner_offset: float = 0.5

    action_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(num_actions,))
    observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(num_observations,), dtype=float)
    state_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(num_observations,), dtype=float)

    action_scale_anymal_d: float = 0.5
    action_scale_anymal_c: float = 0.5
    action_scale_anymal_b: float = 0.5
    action_scale_unitree_a1: float = 0.25
    action_scale_unitree_go1: float = 0.25
    action_scale_unitree_go2: float = 0.25
    action_scale_unitree_b2: float = 0.5
    action_scale_spot: float = 0.2

    # Base Scene defaults to Blind
    scene: HeterogeneousQuadrupedSceneCfg = HeterogeneousQuadrupedSceneCfg(num_envs=4096, env_spacing=4.0)

    contact_threshold = 1.0  # used to terminate episodes

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Instantiate the rewards class
    rewards: RewardsCfg = RewardsCfg()

    events = None  # Disable EventManager, we'll handle randomization manually

    # Domain randomization configuration
    domain_randomization: bool = True

    # Mass randomization (additive, in kg)
    base_mass_range_large: tuple = (-5.0, 5.0)  # Anymal, B2, Spot
    base_mass_range_small: tuple = (-1.0, 3.0)  # A1, Go1, Go2

    # Observation noise configuration
    observation_noise: ObservationNoiseCfg = ObservationNoiseCfg()

    # Command settings (Matching UniformVelocityCommandCfg from manager-based)
    resampling_time_range: tuple = (10.0, 10.0)  # Min/Max seconds before resampling
    rel_standing_envs: float = 0.02  # Percentage of standing environments
    rel_heading_envs: float = 1.0  # Percentage of environments tracking heading
    heading_control_stiffness: float = 0.5  # Stiffness for heading control
    heading_command: bool = True  # Enable heading command mode

    command_ranges_default = {
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_z": (-1.0, 1.0),
        "heading": (-math.pi, math.pi),
    }

    # Per-robot reset randomization configurations
    # Format: (min, max) for position scaling or offset
    # "scale" = multiply default by factor, "offset" = add to default

    # ANYmal robots - use scaling (robust to wide variations)
    reset_joint_pos_range_anymal_d: tuple = (0.8, 1.2)
    reset_joint_pos_mode_anymal_d: str = "scale"
    reset_base_vel_range_anymal_d: tuple = (-0.5, 0.5)

    reset_joint_pos_range_anymal_c: tuple = (0.8, 1.2)
    reset_joint_pos_mode_anymal_c: str = "scale"
    reset_base_vel_range_anymal_c: tuple = (-0.5, 0.5)

    reset_joint_pos_range_anymal_b: tuple = (0.8, 1.2)
    reset_joint_pos_mode_anymal_b: str = "scale"
    reset_base_vel_range_anymal_b: tuple = (-0.5, 0.5)

    # Unitree small robots - NO randomization (sensitive to initial pose)
    reset_joint_pos_range_unitree_a1: tuple = (1.0, 1.0)
    reset_joint_pos_mode_unitree_a1: str = "scale"
    reset_base_vel_range_unitree_a1: tuple = (0.0, 0.0)

    reset_joint_pos_range_unitree_go1: tuple = (1.0, 1.0)
    reset_joint_pos_mode_unitree_go1: str = "scale"
    reset_base_vel_range_unitree_go1: tuple = (0.0, 0.0)

    reset_joint_pos_range_unitree_go2: tuple = (1.0, 1.0)
    reset_joint_pos_mode_unitree_go2: str = "scale"
    reset_base_vel_range_unitree_go2: tuple = (0.0, 0.0)

    # Unitree B2 - larger, can handle scaling
    reset_joint_pos_range_unitree_b2: tuple = (0.8, 1.2)
    reset_joint_pos_mode_unitree_b2: str = "scale"
    reset_base_vel_range_unitree_b2: tuple = (-0.5, 0.5)

    # Spot - use offset-based (complex kinematics)
    reset_joint_pos_range_spot: tuple = (-0.05, 0.05)
    reset_joint_pos_mode_spot: str = "offset"
    reset_base_vel_range_spot: tuple = (-0.5, 0.5)

    # Joint velocity randomization (usually disabled)
    reset_joint_vel_range: tuple = (0.0, 0.0)

    def __post_init__(self):
        """Post initialization."""
        # Set all simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15


@configclass
class HeterogeneousQuadrupedFlatEnvCfg(HeterogeneousQuadrupedVelocityEnvCfg):
    """Configuration for flat terrain (Blind / 48 Obs)."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


@configclass
class HeterogeneousQuadrupedFlatEnvCfg_PLAY(HeterogeneousQuadrupedFlatEnvCfg):
    """Configuration for play/evaluation (Blind / 48 Obs)."""

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50


@configclass
class HeterogeneousQuadrupedRoughEnvCfg(HeterogeneousQuadrupedVelocityEnvCfg):
    """Configuration for rough terrain (Vision / 235 Obs)."""

    num_actions = 12
    num_observations = 235
    include_height_scanners = True

    action_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(num_actions,))
    observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(num_observations,), dtype=float)
    state_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(num_observations,), dtype=float)

    # Override the base scene with the Rough Scene that contains RayCasters
    scene: HeterogeneousQuadrupedRoughSceneCfg = HeterogeneousQuadrupedRoughSceneCfg(num_envs=4096, env_spacing=4.0)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Override the reward configs for rough terrain
        self.rewards.flat_orientation_l2_anymal_d.weight = 0.0
        self.rewards.dof_torques_l2_anymal_d.weight = -1.0e-5
        self.rewards.feet_air_time_anymal_d.weight = 0.125

        self.rewards.flat_orientation_l2_anymal_c.weight = 0.0
        self.rewards.dof_torques_l2_anymal_c.weight = -1.0e-5
        self.rewards.feet_air_time_anymal_c.weight = 0.125

        self.rewards.flat_orientation_l2_anymal_b.weight = 0.0
        self.rewards.dof_torques_l2_anymal_b.weight = -1.0e-5
        self.rewards.feet_air_time_anymal_b.weight = 0.125

        self.rewards.flat_orientation_l2_unitree_a1.weight = 0.0
        self.rewards.feet_air_time_unitree_a1.weight = 0.01

        self.rewards.flat_orientation_l2_unitree_go1.weight = 0.0
        self.rewards.feet_air_time_unitree_go1.weight = 0.01

        self.rewards.flat_orientation_l2_unitree_go2.weight = 0.0
        self.rewards.feet_air_time_unitree_go2.weight = 0.01

        self.rewards.flat_orientation_l2_unitree_b2.weight = 0.0
        self.rewards.dof_torques_l2_unitree_b2.weight = -1.0e-5

        # Note: spot is automatically excluded from rough terrain in the env __init__
        # (HeterogeneousQuadrupedRoughSceneCfg has no spot_scanner).
        # _filter_rewards() removes all spot reward terms automatically.

@configclass
class HeterogeneousQuadrupedRoughEnvCfg_PLAY(HeterogeneousQuadrupedRoughEnvCfg):
    """Configuration for play/evaluation on rough terrain (Vision / 235 Obs)."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
