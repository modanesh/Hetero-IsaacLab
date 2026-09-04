# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .hetero_humanoid_velocity_env_cfg import (
    HeterogeneousHumanoidFlatEnvCfg,
    HeterogeneousHumanoidFlatEnvCfg_PLAY,
    HeterogeneousHumanoidRoughEnvCfg,
    HeterogeneousHumanoidRoughEnvCfg_PLAY,
)

gym.register(
    id="Isaac-Velocity-Flat-HeteroHumanoid-v0",
    entry_point="isaaclab_tasks.direct.hetero_humanoid.hetero_humanoid_env:HeterogeneousHumanoidVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousHumanoidFlatEnvCfg,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hetero_humanoid.agents.rsl_rl_ppo_cfg:HeterogeneousHumanoidFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-HeteroHumanoid-Play-v0",
    entry_point="isaaclab_tasks.direct.hetero_humanoid.hetero_humanoid_env:HeterogeneousHumanoidVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousHumanoidFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hetero_humanoid.agents.rsl_rl_ppo_cfg:HeterogeneousHumanoidFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-HeteroHumanoid-v0",
    entry_point="isaaclab_tasks.direct.hetero_humanoid.hetero_humanoid_env:HeterogeneousHumanoidVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousHumanoidRoughEnvCfg,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hetero_humanoid.agents.rsl_rl_ppo_cfg:HeterogeneousHumanoidRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-HeteroHumanoid-Play-v0",
    entry_point="isaaclab_tasks.direct.hetero_humanoid.hetero_humanoid_env:HeterogeneousHumanoidVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousHumanoidRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hetero_humanoid.agents.rsl_rl_ppo_cfg:HeterogeneousHumanoidRoughPPORunnerCfg",
    },
)
