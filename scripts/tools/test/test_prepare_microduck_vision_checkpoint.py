# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual-task migration must preserve real actor inference and independent critic dimensions."""

import copy

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from scripts.tools.prepare_microduck_curriculum_checkpoint import CURRICULUM_PROFILE
from scripts.tools.prepare_microduck_vision_checkpoint import (
    LAST_SEEN_BEARING_COLUMNS,
    LEGACY_VISION_PROFILE,
    VISION_PROFILE,
    VISION_STATE_COLUMNS,
    prepare_vision_checkpoint,
)


def _model(width: int, players: int = 2, critic: bool = False) -> MLPModel:
    group = "critic" if critic else "policy"
    return MLPModel(
        TensorDict({group: torch.zeros(4, width)}, batch_size=[4]),
        {"actor": ["policy"], "critic": ["critic"]},
        "critic" if critic else "actor",
        1 if critic else 14 * players,
        hidden_dims=[512, 256, 128] if critic else [512 * players, 256 * players, 128 * players],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None
        if critic
        else {"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
    )


@pytest.mark.parametrize(
    "width,players,profile",
    [(61, 1, None), (71, 1, None), (142, 2, None), (142, 2, CURRICULUM_PROFILE), (146, 2, LEGACY_VISION_PROFILE)],
)
def test_vision_transfer_preserves_actions_and_keeps_state_inputs_trainable(width, players, profile):
    """Reindexing preserves dense cross-weights and old features despite arbitrary new visual history."""
    torch.manual_seed(23)
    source = _model(width, players).eval()
    with torch.no_grad():
        source.obs_normalizer._mean.uniform_(-0.5, 0.5)
        source.obs_normalizer._std.uniform_(0.2, 1.4)
        source.obs_normalizer._var.copy_(source.obs_normalizer._std.square())
        source.obs_normalizer.count.fill_(1_838_856_192)
    original = copy.deepcopy(source.state_dict())
    checkpoint = {"actor_state_dict": original, "iter": 40250, "optimizer_state_dict": {"stale": True}}
    if profile:
        checkpoint["infos"] = {"microduck_curriculum": {"profile": profile, "curriculum_state": {"stage": 2}}}
    prepared = prepare_vision_checkpoint(checkpoint, initial_std=0.09)
    target = _model(152).eval()
    target.load_state_dict(prepared["actor_state_dict"], strict=True)
    critic = _model(142, critic=True)
    critic.load_state_dict(prepared["critic_state_dict"], strict=True)

    source_block_width = 73 if profile == LEGACY_VISION_PROFILE else 71
    legacy = torch.randn(4, 2, source_block_width)
    if profile is None:
        legacy[:, :, 48:61] = 0
    if players == 1:
        expected = torch.cat(
            [source(TensorDict({"policy": legacy[:, player, :width]}, batch_size=[4])) for player in range(2)], -1
        )
    else:
        expected = source(TensorDict({"policy": legacy.flatten(1)}, batch_size=[4]))
    visual = torch.cat((legacy, torch.randn(4, 2, 76 - source_block_width) * 5), -1)
    actual = target(TensorDict({"policy": visual.flatten(1)}, batch_size=[4]))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(original[name], value, rtol=0, atol=0)
    assert prepared["iter"] == 0 and "optimizer_state_dict" not in prepared
    metadata = prepared["infos"]["microduck_curriculum"]
    assert metadata["profile"] == VISION_PROFILE and "curriculum_state" not in metadata
    assert metadata["source_iteration"] == 40250
    assert metadata["source_curriculum_stage"] == (2 if profile else None)
    assert critic.obs_normalizer.count.item() == 0
    assert target.obs_normalizer.count.item() == 1_838_856_192
    privileged = legacy[:, :, :71].reshape(4, 142)
    assert torch.isfinite(critic(TensorDict({"critic": privileged}, batch_size=[4]))).all()
    torch.testing.assert_close(target.distribution.std_param, torch.full((28,), 0.09))

    actual.square().sum().backward()
    new_columns = LAST_SEEN_BEARING_COLUMNS if profile == LEGACY_VISION_PROFILE else VISION_STATE_COLUMNS
    assert target.mlp[0].weight.grad[:, new_columns].abs().sum() > 0
    assert all(parameter.requires_grad for parameter in target.parameters())
    if profile:
        torch.testing.assert_close(target.mlp[0].weight[:, 48:61], source.mlp[0].weight[:, 48:61])
        torch.testing.assert_close(
            target.mlp[0].weight[:, 124:137], source.mlp[0].weight[:, source_block_width + 48 : source_block_width + 61]
        )


@pytest.mark.parametrize(
    "damage", ["duplicate_conversion", "nonfinite_actor", "malformed_actor", "unknown_profile", "corrupt_vision_v1"]
)
def test_vision_transfer_rejects_incompatible_or_corrupt_sources(damage):
    """Invalid shape/profile or NaN state must fail before a long visual training job starts."""
    source = {"actor_state_dict": _model(142).state_dict()}
    if damage == "duplicate_conversion":
        source = prepare_vision_checkpoint(source)
        message = "already a vision checkpoint"
    elif damage == "nonfinite_actor":
        source["actor_state_dict"]["obs_normalizer._std"][0, 0] = float("nan")
        message = "Invalid actor tensor"
    elif damage == "malformed_actor":
        source["actor_state_dict"]["mlp.0.weight"] = source["actor_state_dict"]["mlp.0.weight"][:, :-1]
        message = "Expected a MicroDuck actor"
    elif damage == "corrupt_vision_v1":
        source["actor_state_dict"] = _model(146).state_dict()
        source["actor_state_dict"]["mlp.0.weight"][0, 145] = float("nan")
        source["infos"] = {"microduck_curriculum": {"profile": LEGACY_VISION_PROFILE}}
        message = "Invalid actor tensor"
    else:
        source["infos"] = {"microduck_curriculum": {"profile": "unknown_visual_task_v2"}}
        message = "Unsupported source checkpoint profile"
    with pytest.raises(ValueError, match=message):
        prepare_vision_checkpoint(source)
