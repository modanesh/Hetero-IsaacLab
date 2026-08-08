# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlMLPModelCfg


@configclass
class HeterogeneousHumanoidRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "heterogeneous_humanoid_rough"
    empirical_normalization = False
    actor = RslRlMLPModelCfg(
        class_name="isaaclab_tasks.direct.hetero_humanoid.masked_mlp_model:MaskedMLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="GaussianDistribution",
            init_std=1.0,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeterogeneousHumanoidFlatPPORunnerCfg(HeterogeneousHumanoidRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.experiment_name = "heterogeneous_humanoid_flat"
        self.actor.hidden_dims = [512, 256, 128]
        self.critic.hidden_dims = [512, 256, 128]

