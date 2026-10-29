# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadrupeds and humanoids together on rough terrain, in one simulation.

All robots with a terrain scanner: the 7 of hetero_quadruped (Spot has none) and the 5 of hetero_humanoid. Each
robot keeps the scene entries, reward terms and per-robot settings it has in its own family's rough-terrain
config; settings shared by the whole env (terrain, simulation, episode length, command defaults, observation noise)
are hetero_humanoid's, which match hetero_quadruped's.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.direct.hetero_humanoid.hetero_humanoid_velocity_env_cfg import HeterogeneousHumanoidRoughEnvCfg
from isaaclab_tasks.direct.hetero_quadruped.hetero_quadruped_velocity_env_cfg import (
    HeterogeneousQuadrupedRoughEnvCfg,
)

QUADRUPEDS = ["anymal_d", "anymal_c", "anymal_b", "unitree_a1", "unitree_go1", "unitree_go2", "unitree_b2"]
HUMANOIDS = ["cassie", "digit", "g1", "h1", "h2"]


@configclass
class HeteroLeggedRoughEnvCfg(HeterogeneousHumanoidRoughEnvCfg):
    robots: list[str] = QUADRUPEDS + HUMANOIDS

    # per-robot scan pattern, {robot_name: (length_x, width_y, resolution)} in metres; robots not listed keep
    # their family's 1.6 m x 1.0 m grid at 0.1 m
    scan_patterns: dict = {}

    def __post_init__(self):
        super().__post_init__()
        quad = HeterogeneousQuadrupedRoughEnvCfg()

        for name in QUADRUPEDS:
            for entry in (name, f"{name}_contacts", f"{name}_scanner"):
                setattr(self.scene, entry, getattr(quad.scene, entry))
            for setting in ("action_scale", "reset_joint_pos_range", "reset_joint_pos_mode", "reset_base_vel_range"):
                setattr(self, f"{setting}_{name}", getattr(quad, f"{setting}_{name}"))
            # hetero_quadruped's mass groups, base CoM randomization and standing share
            large = name in ("anymal_d", "anymal_c", "anymal_b", "unitree_b2")
            setattr(self, f"base_mass_range_{name}", quad.base_mass_range_large if large else quad.base_mass_range_small)
            setattr(self, f"randomize_base_com_{name}", True)
            setattr(self, f"standing_probability_{name}", quad.rel_standing_envs)

        for term, value in vars(quad.rewards).items():
            if any(term.endswith(f"_{name}") for name in QUADRUPEDS):
                setattr(self.rewards, term, value)

        self.sim.physx.gpu_max_rigid_patch_count = quad.sim.physx.gpu_max_rigid_patch_count
