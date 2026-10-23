# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint transfer must preserve real RSL-RL inference before fine-tuning."""

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from scripts.tools.prepare_microduck_ball_checkpoint import prepare_checkpoint


def _model(width: int, players: int = 1, critic: bool = False) -> MLPModel:
    return MLPModel(
        TensorDict({"policy": torch.zeros(3, width)}, batch_size=[3]),
        {"actor": ["policy"], "critic": ["policy"]},
        "critic" if critic else "actor",
        1 if critic else 14 * players,
        hidden_dims=[512, 256, 128] if critic else [512 * players, 256 * players, 128 * players],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None
        if critic
        else {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    )


@pytest.mark.parametrize("source_width,players", [(61, 1), (71, 2)])
def test_transfer_preserves_policy_and_can_learn_ball_inputs(source_width, players):
    """Padding and duplication preserve actions, and PPO can update formerly unused ball inputs."""
    torch.manual_seed(7)
    source_model = _model(source_width).eval()
    # Non-identity statistics expose accidental loss or reordering of the source normalizer.
    with torch.no_grad():
        source_model.obs_normalizer._mean.uniform_(-0.2, 0.2)
        source_model.obs_normalizer._std.uniform_(0.5, 1.5)
        source_model.obs_normalizer._var.copy_(source_model.obs_normalizer._std.square())
        source_model.obs_normalizer.count.fill_(12345)
    result = prepare_checkpoint({"actor_state_dict": source_model.state_dict(), "iter": 4000}, players)
    target = _model(71 * players, players).eval()
    target.load_state_dict(result["actor_state_dict"], strict=True)
    critic = _model(71 * players, critic=True)
    critic.load_state_dict(result["critic_state_dict"], strict=True)
    blocks = torch.randn(3, players, 71)
    expected = torch.cat(
        [source_model(TensorDict({"policy": blocks[:, i, :source_width]}, batch_size=[3])) for i in range(players)],
        dim=-1,
    )
    actual = target(TensorDict({"policy": blocks.flatten(1)}, batch_size=[3]))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert result["iter"] == 0 and "optimizer_state_dict" not in result
    assert critic.obs_normalizer.count.item() == 0
    assert torch.isfinite(critic(TensorDict({"policy": blocks.flatten(1)}, batch_size=[3]))).all()

    # Every joint and the newly introduced connections must be able to learn.
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)
    before = actual.detach().clone()
    actual.square().sum().backward()
    new_connections = (
        target.mlp[0].weight.grad[:, source_width:71] if source_width < 71 else target.mlp[0].weight.grad[:512, 71:]
    )
    assert new_connections.abs().sum() > 0
    assert all(parameter.requires_grad for parameter in target.parameters())
    optimizer.step()
    after = target(TensorDict({"policy": blocks.flatten(1)}, batch_size=[3]))
    assert not torch.allclose(before, after)


def test_transfer_rejects_already_centralized_actor():
    """A 28-action pair must not be silently interpreted as a single robot on a later transfer."""
    source = {"actor_state_dict": _model(142, players=2).state_dict()}
    with pytest.raises(ValueError, match="single MicroDuck actor"):
        prepare_checkpoint(source, players=2)
