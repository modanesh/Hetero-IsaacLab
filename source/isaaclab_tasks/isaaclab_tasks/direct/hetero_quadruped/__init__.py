# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .hetero_quadruped_velocity_env_cfg import HeterogeneousQuadrupedFlatEnvCfg, HeterogeneousQuadrupedFlatEnvCfg_PLAY, HeterogeneousQuadrupedRoughEnvCfg, HeterogeneousQuadrupedRoughEnvCfg_PLAY

gym.register(
    id="Isaac-Velocity-Flat-HeteroQuadruped-v0",
    entry_point="isaaclab_tasks.direct.hetero_quadruped.hetero_quadruped_env:HeterogeneousQuadrupedVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousQuadrupedFlatEnvCfg,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hetero_quadruped.agents.rsl_rl_ppo_cfg:HeterogeneousQuadrupedFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-HeteroQuadruped-Play-v0",
    entry_point="isaaclab_tasks.direct.hetero_quadruped.hetero_quadruped_env:HeterogeneousQuadrupedVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousQuadrupedFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hetero_quadruped.agents.rsl_rl_ppo_cfg:HeterogeneousQuadrupedFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-HeteroQuadruped-v0",
    entry_point="isaaclab_tasks.direct.hetero_quadruped.hetero_quadruped_env:HeterogeneousQuadrupedVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousQuadrupedRoughEnvCfg,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hetero_quadruped.agents.rsl_rl_ppo_cfg:HeterogeneousQuadrupedRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-HeteroQuadruped-Play-v0",
    entry_point="isaaclab_tasks.direct.hetero_quadruped.hetero_quadruped_env:HeterogeneousQuadrupedVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousQuadrupedRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hetero_quadruped.agents.rsl_rl_ppo_cfg:HeterogeneousQuadrupedRoughPPORunnerCfg",
    },
)