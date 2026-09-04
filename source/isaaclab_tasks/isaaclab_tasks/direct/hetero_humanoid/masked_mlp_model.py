# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from rsl_rl.models.mlp_model import MLPModel


class MaskedMLPModel(MLPModel):
    """Custom MLP Model that intercepts continuous action masks from observations
    and strictly zeroes out padded dummy dimensions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_mask = None
        self.mask_dim = args[3] if len(args) > 3 else kwargs.get("output_dim", 64)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        # The environment returns the action mask as a separate key in the TensorDict
        if hasattr(obs, "keys") and "action_mask" in obs.keys():
            self.current_mask = obs["action_mask"].clone()
        else:
            # Fallback if action_mask isn't provided (e.g. inference time)
            self.current_mask = None

        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)

        if self.current_mask is not None:
            mlp_output = mlp_output * self.current_mask

        if self.distribution is not None:
            self.distribution.update(mlp_output)
            if stochastic_output:
                actions = self.distribution.sample()
            else:
                actions = self.distribution.deterministic_output(mlp_output)
        else:
            actions = mlp_output

        if self.current_mask is not None:
            actions = actions * self.current_mask

        return actions

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        # Access the internal torch.distributions.Normal object directly to get unsummed log probs
        unsummed_log_prob = self.distribution._distribution.log_prob(outputs)

        if self.current_mask is not None:
            unsummed_log_prob = unsummed_log_prob * self.current_mask

        return unsummed_log_prob.sum(dim=-1)

    @property
    def output_entropy(self) -> torch.Tensor:
        # Access the internal torch.distributions.Normal object directly to get unsummed entropy
        unsummed_entropy = self.distribution._distribution.entropy()

        if self.current_mask is not None:
            unsummed_entropy = unsummed_entropy * self.current_mask

        return unsummed_entropy.sum(dim=-1)
