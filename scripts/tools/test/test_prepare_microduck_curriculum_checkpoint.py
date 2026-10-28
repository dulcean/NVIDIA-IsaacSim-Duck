# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum transfer preserves actual RSL inference while exposing new state inputs."""

import copy

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from scripts.tools.prepare_microduck_curriculum_checkpoint import prepare_curriculum_checkpoint


def _model(width: int, players: int = 1, critic: bool = False) -> MLPModel:
    return MLPModel(
        TensorDict({"policy": torch.zeros(5, width)}, batch_size=[5]),
        {"actor": ["policy"], "critic": ["policy"]},
        "critic" if critic else "actor",
        1 if critic else 14 * players,
        hidden_dims=[512, 256, 128] if critic else [512 * players, 256 * players, 128 * players],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None
        if critic
        else {"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
    )


@pytest.mark.parametrize("width,players", [(61, 1), (71, 1), (142, 2)])
def test_transfer_preserves_legacy_actions_for_arbitrary_new_state(width, players):
    """Real centralized cross-weights and nonzero normalized placeholders must retain their contribution."""
    torch.manual_seed(13)
    source = _model(width, players).eval()
    with torch.no_grad():
        source.obs_normalizer._mean.uniform_(-0.7, 0.7)
        source.obs_normalizer._std.uniform_(0.2, 1.3)
        source.obs_normalizer._var.copy_(source.obs_normalizer._std.square())
        source.obs_normalizer.count.fill_(400_000_000)
    original = copy.deepcopy(source.state_dict())
    prepared = prepare_curriculum_checkpoint(
        {"actor_state_dict": original, "iter": 40250, "optimizer_state_dict": {"stale": True}}, initial_std=0.08
    )
    target = _model(142, 2).eval()
    target.load_state_dict(prepared["actor_state_dict"], strict=True)
    critic = _model(142, critic=True)
    critic.load_state_dict(prepared["critic_state_dict"], strict=True)
    legacy_blocks = torch.randn(5, 2, 71)
    legacy_blocks[:, :, 48:61] = 0.0
    if players == 1:
        expected = torch.cat(
            [source(TensorDict({"policy": legacy_blocks[:, i, :width]}, batch_size=[5])) for i in range(2)], -1
        )
    else:
        expected = source(TensorDict({"policy": legacy_blocks.flatten(1)}, batch_size=[5]))
    curriculum_blocks = legacy_blocks.clone()
    curriculum_blocks[:, :, 48:61] = torch.randn(5, 2, 13) * 3
    actual = target(TensorDict({"policy": curriculum_blocks.flatten(1)}, batch_size=[5]))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(original[name], value, rtol=0, atol=0)
    assert prepared["iter"] == 0 and "optimizer_state_dict" not in prepared
    assert critic.obs_normalizer.count.item() == 0
    assert target.obs_normalizer.count.item() == 400_000_000
    torch.testing.assert_close(target.distribution.std_param, torch.full((28,), 0.08))
    assert torch.isfinite(critic(TensorDict({"policy": curriculum_blocks.flatten(1)}, batch_size=[5]))).all()
    actual.square().sum().backward()
    assert target.mlp[0].weight.grad[:, 48:61].abs().sum() > 0
    assert target.mlp[0].weight.grad[:, 119:132].abs().sum() > 0
    assert all(parameter.requires_grad for parameter in target.parameters())
    with pytest.raises(ValueError, match="already a curriculum checkpoint"):
        prepare_curriculum_checkpoint(prepared)


def test_rejects_corrupt_checkpoint_before_policy_can_be_started():
    """Malformed observations or nonfinite source weights cannot silently enter a training job."""
    source = {"actor_state_dict": _model(142, 2).state_dict()}
    source["actor_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Invalid actor tensor mlp.0.weight"):
        prepare_curriculum_checkpoint(source)
