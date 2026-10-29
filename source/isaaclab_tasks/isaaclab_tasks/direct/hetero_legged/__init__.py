# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadrupeds and humanoids trained together on rough terrain."""

import gymnasium as gym

from .hetero_legged_env_cfg import HeteroLeggedRoughEnvCfg

gym.register(
    id="Isaac-Velocity-Rough-HeteroLegged-v0",
    entry_point="isaaclab_tasks.direct.hetero_legged.hetero_legged_env:HeteroLeggedVelocityEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": HeteroLeggedRoughEnvCfg},
)
