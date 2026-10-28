# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL actor with independent camera/proprioception paths for the two players."""

import torch
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn


class _PlayerLocalLinear(nn.Module):
    """Keep checkpoint-shaped parameters while excluding the other player's path.

    Sliced matrix products also prevent an invalid value in the other player's
    input from propagating through a nominal zero connection (zero times NaN).
    The fixed nonpersistent mask is recreated on construction and follows device
    transfers/export copies. Parameter names remain ``weight`` and ``bias``.
    """

    __constants__ = ["half_input", "half_output"]

    def __init__(self, layer: nn.Linear):
        super().__init__()
        if layer.in_features % 2 or layer.out_features % 2 or layer.bias is None:
            raise ValueError("A local player layer requires even input/output widths and a bias.")
        self.half_input = layer.in_features // 2
        self.half_output = layer.out_features // 2
        self.weight = layer.weight
        self.bias = layer.bias
        mask = torch.zeros_like(layer.weight)
        mask[: self.half_output, : self.half_input] = 1
        mask[self.half_output :, self.half_input :] = 1
        self.register_buffer("_player_mask", mask, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self._player_mask
        first = F.linear(
            value[..., : self.half_input],
            weight[: self.half_output, : self.half_input],
            self.bias[: self.half_output],
        )
        second = F.linear(
            value[..., self.half_input :],
            weight[self.half_output :, self.half_input :],
            self.bias[self.half_output :],
        )
        return torch.cat((first, second), dim=-1)


class LocalPassActor(MLPModel):
    """Two trainable 89→512→256→128→14 policies in checkpoint-compatible tensors.

    Player A's 14 means depend only on its 89 camera/own-odometry inputs, and the
    same holds for player B. The complete MLP retains 178→1024→512→256→28 shapes
    and existing RSL parameter keys, so an explicitly projected 178-input actor
    can load strictly. A legacy centralized policy must first drop cross-player
    weights; its original behavior is therefore not claimed to be preserved.

    RSL's featurewise empirical normalizer, diagonal action distribution, PPO
    log-probabilities, and export wrappers remain unchanged. All connections
    inside each player's path, all servo outputs, and their distribution scales
    remain trainable. The critic may still use privileged global training state.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ):
        if obs_set != "actor" or output_dim != 28:
            raise ValueError("LocalPassActor is an actor for two independent 14-servo action blocks.")
        if tuple(hidden_dims) != (1024, 512, 256):
            raise ValueError("LocalPassActor requires checkpoint-compatible hidden widths [1024, 512, 256].")
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        if self.obs_dim != 178:
            raise ValueError("LocalPassActor requires exactly two 89-value player observation blocks (178 total).")
        if not isinstance(self.mlp[-1], nn.Linear) or self.mlp[-1].out_features != 28:
            raise ValueError("LocalPassActor requires a state-independent distribution with 28 MLP outputs.")
        for index, layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear):
                self.mlp[index] = _PlayerLocalLinear(layer)
