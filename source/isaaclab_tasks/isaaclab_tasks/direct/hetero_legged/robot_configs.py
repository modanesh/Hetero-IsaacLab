# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint and termination settings of every robot in hetero_legged, and the feet of each robot.

The humanoids' settings are those of hetero_humanoid. The quadrupeds control all 12 leg joints and end an
episode when the base touches the ground, as in hetero_quadruped.
"""

from __future__ import annotations

from isaaclab_tasks.direct.hetero_humanoid.robot_configs import ROBOT_CONFIGS as HUMANOID_CONFIGS
from isaaclab_tasks.direct.hetero_humanoid.robot_configs import RobotJointConfig

_ANYMAL_JOINTS = [f"{leg}_{joint}" for joint in ("HAA", "HFE", "KFE") for leg in ("LF", "LH", "RF", "RH")]
_UNITREE_JOINTS = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]

QUADRUPED_CONFIGS: dict[str, RobotJointConfig] = {
    "anymal_d": RobotJointConfig(canonical_joints=_ANYMAL_JOINTS, base_link="base", action_scale=0.5),
    "anymal_c": RobotJointConfig(canonical_joints=_ANYMAL_JOINTS, base_link="base", action_scale=0.5),
    "anymal_b": RobotJointConfig(canonical_joints=_ANYMAL_JOINTS, base_link="base", action_scale=0.5),
    "unitree_a1": RobotJointConfig(canonical_joints=_UNITREE_JOINTS, base_link="trunk", action_scale=0.25),
    "unitree_go1": RobotJointConfig(canonical_joints=_UNITREE_JOINTS, base_link="trunk", action_scale=0.25),
    "unitree_go2": RobotJointConfig(canonical_joints=_UNITREE_JOINTS, base_link="base", action_scale=0.25),
    "unitree_b2": RobotJointConfig(canonical_joints=_UNITREE_JOINTS, base_link="base_link", action_scale=0.5),
}

ROBOT_CONFIGS: dict[str, RobotJointConfig] = {**QUADRUPED_CONFIGS, **HUMANOID_CONFIGS}

# the body at the end of each leg (regex over body names), used for the robot description
FEET: dict[str, str] = {
    "anymal_d": ".*_FOOT",
    "anymal_c": ".*_FOOT",
    "anymal_b": ".*_FOOT",
    "unitree_a1": ".*_foot",
    "unitree_go1": ".*_foot",
    "unitree_go2": ".*_foot",
    "unitree_b2": ".*_foot",
    "cassie": ".*toe",
    "digit": ".*_leg_toe_roll",
    "g1": ".*_ankle_roll_link",
    "h1": ".*ankle_link",
    "h2": ".*_ankle_roll_link",
}
