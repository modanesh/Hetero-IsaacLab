# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized robot joint configuration for heterogeneous humanoid environments.

Each robot has a canonical 12-DOF biped leg ordering used for action/observation
alignment across different morphologies. Extra DOFs (arms, torso, fingers) are
appended after the canonical legs and discovered automatically at runtime.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RobotJointConfig:
    """Configuration for a single robot's joint layout and properties."""

    # --- Canonical 50-DOF semantic joint alignment ---
    # We use a canonical slot ordering to align identical joints across different robots.
    # 0-11: Legs
    # 12-14: Torso (Yaw, Pitch, Roll)
    # 15-19: Left Arm (Shoulder Pitch, Roll, Yaw, Elbow Pitch, Roll)
    # 20-24: Right Arm (Shoulder Pitch, Roll, Yaw, Elbow Pitch, Roll)
    # 25+: Unmapped joints (Fingers, etc) will be appended dynamically
    canonical_joints: List[str | None] = field(default_factory=list)

    # --- Kinematic type ---
    # "bird" = Cassie/Digit (no sign flip needed)
    # "humanoid" = G1/H1 (pitch & knee signs flipped to align with bird-leg convention)
    kinematic_type: str = "bird"

    # --- Termination ---
    base_link: str = "torso"
    orientation_limit: float | None = None  # radians; None = no orientation termination

    # --- Action scale override (if different from default) ---
    action_scale: float = 0.5


# =============================================================================
# Per-Robot Configurations
# =============================================================================

ROBOT_CONFIGS: Dict[str, RobotJointConfig] = {
    "cassie": RobotJointConfig(
        canonical_joints=[
            # 0-11: Legs
            "hip_flexion_left",
            "hip_abduction_left",
            "hip_rotation_left",
            "thigh_joint_left",
            "ankle_joint_left",
            "toe_joint_left",
            "hip_flexion_right",
            "hip_abduction_right",
            "hip_rotation_right",
            "thigh_joint_right",
            "ankle_joint_right",
            "toe_joint_right",
            # 12-14: Torso
            None,
            None,
            None,
            # 15-19: Left Arm
            None,
            None,
            None,
            None,
            None,
            # 20-24: Right Arm
            None,
            None,
            None,
            None,
            None,
        ],
        kinematic_type="bird",
        base_link="pelvis",
        orientation_limit=None,
        action_scale=0.5,
    ),
    "digit": RobotJointConfig(
        canonical_joints=[
            # 0-11: Legs
            "left_leg_hip_pitch",
            "left_leg_hip_roll",
            "left_leg_hip_yaw",
            "left_leg_knee",
            "left_leg_toe_a",
            "left_leg_toe_b",
            "right_leg_hip_pitch",
            "right_leg_hip_roll",
            "right_leg_hip_yaw",
            "right_leg_knee",
            "right_leg_toe_a",
            "right_leg_toe_b",
            # 12-14: Torso
            None,
            None,
            None,
            # 15-19: Left Arm
            "left_arm_shoulder_pitch",
            "left_arm_shoulder_roll",
            "left_arm_shoulder_yaw",
            "left_arm_elbow",
            None,
            # 20-24: Right Arm
            "right_arm_shoulder_pitch",
            "right_arm_shoulder_roll",
            "right_arm_shoulder_yaw",
            "right_arm_elbow",
            None,
        ],
        kinematic_type="bird",
        base_link="torso_base",
        orientation_limit=0.7,
        action_scale=0.5,
    ),
    "g1": RobotJointConfig(
        canonical_joints=[
            # 0-11: Legs
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # 12-14: Torso
            "torso_joint",
            None,
            None,
            # 15-19: Left Arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_roll_joint",
            # 20-24: Right Arm
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
            # 25-31: Left Hand
            "left_zero_joint",
            "left_one_joint",
            "left_two_joint",
            "left_three_joint",
            "left_four_joint",
            "left_five_joint",
            "left_six_joint",
            # 32-38: Right Hand
            "right_zero_joint",
            "right_one_joint",
            "right_two_joint",
            "right_three_joint",
            "right_four_joint",
            "right_five_joint",
            "right_six_joint",
        ],
        kinematic_type="humanoid",
        base_link="torso_link",
        orientation_limit=None,
        action_scale=0.25,
    ),
    "h1": RobotJointConfig(
        canonical_joints=[
            # 0-11: Legs
            "left_hip_pitch",
            "left_hip_roll",
            "left_hip_yaw",
            "left_knee",
            "left_ankle",
            None,
            "right_hip_pitch",
            "right_hip_roll",
            "right_hip_yaw",
            "right_knee",
            "right_ankle",
            None,
            # 12-14: Torso
            "torso",
            None,
            None,
            # 15-19: Left Arm
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            None,
            # 20-24: Right Arm
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
            None,
        ],
        kinematic_type="humanoid",
        base_link="torso_link",
        orientation_limit=None,
        action_scale=0.5,
    ),
}
