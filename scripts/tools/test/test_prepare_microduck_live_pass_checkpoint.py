# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local visual warm starts retain learned connections without cross-player state or stale progress."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from scripts.tools.prepare_microduck_live_pass_checkpoint import LIVE_PASS_PROFILE, prepare_live_pass_checkpoint


def _model(width, *, critic=False, isolated=True):
    group = "critic" if critic else "policy"
    model_type = MLPModel
    if width == 178 and not critic and isolated:
        path = (
            Path(__file__).resolve().parents[3]
            / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/live_pass_actor.py"
        )
        spec = importlib.util.spec_from_file_location("isaaclab_tasks.contrib.microduck.live_pass_actor", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model_type = module.LocalPassActor
    return model_type(
        TensorDict({group: torch.zeros(4, width)}, batch_size=[4]),
        {"actor": ["policy"], "critic": ["critic"]},
        "critic" if critic else "actor",
        1 if critic else 28,
        hidden_dims=[512, 256, 128] if critic else [1024, 512, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None
        if critic
        else {"class_name": "GaussianDistribution", "init_std": 0.14, "std_type": "scalar"},
    )


def _checkpoint(actor):
    return {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": {"stale": torch.tensor(123.0)},
        "optimizer_state_dict": {"stale": True},
        "iter": 4999,
        "infos": {
            "microduck_curriculum": {
                "profile": "microduck_volley_vision_v2",
                "vision_contract": {"version": 3, "actor_width": 152, "camera": "physical_rgb"},
                "curriculum_state": {"stage": 2},
                "certified": True,
                "task_settings": {"version": 1, "ball_solref": [0.02, 0.5], "continuous_launch_shaping": True},
                "player_symmetry": True,
            }
        },
    }


def test_live_pass_transfer_preserves_local_rgb_connections_and_removes_privileged_cross_player_paths():
    """A local actor matches independent source blocks, rather than a centralized policy's actions."""
    torch.manual_seed(2810)
    source = _model(152).eval()
    with torch.no_grad():
        source.obs_normalizer._mean.uniform_(-0.4, 0.4)
        source.obs_normalizer._std.uniform_(0.3, 1.5)
        source.obs_normalizer._var.copy_(source.obs_normalizer._std.square())
        source.obs_normalizer.count.fill_(123456)
        for layer in source.mlp:
            if isinstance(layer, torch.nn.Linear):
                rows, columns = layer.weight.shape
                layer.weight[: rows // 2, columns // 2 :] = 0.25
                layer.weight[rows // 2 :, : columns // 2] = -0.25
    checkpoint = _checkpoint(source)
    original = copy.deepcopy(checkpoint)
    prepared = prepare_live_pass_checkpoint(checkpoint)
    target = _model(178).eval()
    target.load_state_dict(prepared["actor_state_dict"], strict=True)
    legacy_observation = torch.randn(4, 2, 76)
    live_observation = torch.cat((legacy_observation, torch.randn(4, 2, 13) * 3), -1)
    normalized = source.obs_normalizer(legacy_observation.flatten(1))
    local_outputs = []
    for player in range(2):
        local = normalized[:, player * 76 : (player + 1) * 76]
        for layer in source.mlp:
            if isinstance(layer, torch.nn.Linear):
                rows, columns = layer.weight.shape
                inputs = slice(player * columns // 2, (player + 1) * columns // 2)
                outputs = slice(player * rows // 2, (player + 1) * rows // 2)
                local = torch.nn.functional.linear(local, layer.weight[outputs, inputs], layer.bias[outputs])
            else:
                local = layer(local)
        local_outputs.append(local)
    expected = torch.cat(local_outputs, dim=-1)
    actual = target(TensorDict({"policy": live_observation.flatten(1)}, batch_size=[4]))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    # The serialized checkpoint itself must contain the projection; a reader
    # without LocalPassActor's runtime masks must see the same isolated policy.
    serialized_actor = _model(178, isolated=False).eval()
    serialized_actor.load_state_dict(prepared["actor_state_dict"], strict=True)
    torch.testing.assert_close(
        serialized_actor(TensorDict({"policy": live_observation.flatten(1)}, batch_size=[4])),
        expected,
        atol=3e-6,
        rtol=3e-5,
    )
    actual.square().sum().backward()
    assert target.mlp[0].weight.grad[:, 76:89].abs().sum() > 0
    assert target.mlp[0].weight.grad[:, 165:178].abs().sum() > 0
    assert all(parameter.requires_grad for parameter in target.parameters())
    for name, value in checkpoint["actor_state_dict"].items():
        torch.testing.assert_close(value, original["actor_state_dict"][name], rtol=0, atol=0)
    torch.testing.assert_close(target.distribution.std_param, source.distribution.std_param, rtol=0, atol=0)
    critic = _model(142, critic=True).eval()
    critic.load_state_dict(prepared["critic_state_dict"], strict=True)
    assert critic.obs_normalizer.count.item() == 0
    assert torch.isfinite(critic(TensorDict({"critic": torch.randn(4, 142)}, batch_size=[4]))).all()
    assert prepared["iter"] == 0 and "optimizer_state_dict" not in prepared
    metadata = prepared["infos"]["microduck_curriculum"]
    assert metadata["profile"] == LIVE_PASS_PROFILE and metadata["source_iteration"] == 4999
    assert metadata["source_policy_actions_preserved"] is False
    assert not {"curriculum_state", "certified", "certification"}.intersection(metadata)
    assert metadata["source_vision_contract"] == original["infos"]["microduck_curriculum"]["vision_contract"]
    metadata["source_vision_contract"]["version"] = -1
    assert checkpoint["infos"]["microduck_curriculum"]["vision_contract"]["version"] == 3


@pytest.mark.parametrize("damage", ["profile", "camera_contract", "nonfinite", "shape"])
def test_live_pass_transfer_rejects_incompatible_visual_sources(damage):
    """Invalid actor or camera provenance must fail before allocating a long GPU run."""
    source = _checkpoint(_model(152))
    metadata = source["infos"]["microduck_curriculum"]
    if damage == "profile":
        metadata["profile"] = LIVE_PASS_PROFILE
        message = "requires a trained vision-v2 checkpoint"
    elif damage == "camera_contract":
        del metadata["vision_contract"]
        message = "camera contract"
    elif damage == "nonfinite":
        source["actor_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
        message = "Invalid visual actor tensor"
    else:
        source["actor_state_dict"]["obs_normalizer._mean"] = torch.zeros(1, 153)
        message = "Invalid visual actor tensor"
    with pytest.raises(ValueError, match=message):
        prepare_live_pass_checkpoint(source)
