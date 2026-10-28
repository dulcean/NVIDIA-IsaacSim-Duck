# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local information flow and genuine RSL PPO/export contracts for camera passes."""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

_ROOT = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck"
_PACKAGE = "microduck_local_actor_tests"
_namespace = ModuleType(_PACKAGE)
_namespace.__path__ = [str(_ROOT)]
sys.modules[_PACKAGE] = _namespace
local = importlib.import_module(f"{_PACKAGE}.live_pass_actor")
_GROUPS = {"actor": ["policy"], "critic": ["critic"]}
_DISTRIBUTION = {"class_name": "GaussianDistribution", "init_std": 0.15, "std_type": "scalar"}


def actor(obs, model=local.LocalPassActor):
    return model(
        obs,
        _GROUPS,
        "actor",
        28,
        hidden_dims=[1024, 512, 256],
        obs_normalization=True,
        distribution_cfg=_DISTRIBUTION,
    )


def test_actor_strictly_loads_rsl_tensors_and_keeps_each_players_camera_path_private_in_exports():
    """Other-player inputs cannot affect servo means through dense layers, normalization, or export."""
    torch.manual_seed(2810)
    obs = TensorDict({"policy": torch.randn(4, 178)}, batch_size=[4])
    source = actor(obs, MLPModel)
    model = actor(obs).eval()
    model.load_state_dict(source.state_dict(), strict=True)
    model.update_normalization(obs)
    initial = model(obs)
    for player in range(2):
        changed = obs.clone()
        changed["policy"][:, player * 89 : (player + 1) * 89] = torch.randn(4, 89) * 100
        output = model(changed)
        other = slice((1 - player) * 14, (2 - player) * 14)
        torch.testing.assert_close(output[:, other], initial[:, other], rtol=0, atol=0)
        assert not torch.allclose(
            output[:, player * 14 : (player + 1) * 14], initial[:, player * 14 : (player + 1) * 14]
        )
        changed["policy"][:, player * 89 : (player + 1) * 89] = float("nan")
        torch.testing.assert_close(model(changed)[:, other], initial[:, other], rtol=0, atol=0)

    exported = model.as_jit()
    scripted = torch.jit.script(exported)
    compiled = torch.compile(exported, backend="eager", fullgraph=True)
    torch.testing.assert_close(scripted(obs["policy"]), initial)
    torch.testing.assert_close(compiled(obs["policy"]), initial)
    changed = obs["policy"].clone()
    changed[:, 89:] += 20
    torch.testing.assert_close(scripted(changed)[:, :14], initial[:, :14], rtol=0, atol=0)
    torch.testing.assert_close(compiled(changed)[:, :14], initial[:, :14], rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["critic-role", "action-width", "observation-width", "hidden-width"])
def test_local_actor_rejects_incompatible_role_or_checkpoint_dimensions(damage):
    obs = TensorDict({"policy": torch.zeros(1, 178), "critic": torch.zeros(1, 142)}, batch_size=[1])
    arguments = {"obs_set": "actor", "output_dim": 28, "hidden_dims": [1024, 512, 256]}
    if damage == "critic-role":
        arguments["obs_set"] = "critic"
    elif damage == "action-width":
        arguments["output_dim"] = 14
    elif damage == "observation-width":
        obs["policy"] = torch.zeros(1, 152)
    else:
        arguments["hidden_dims"] = [512, 256, 128]
    with pytest.raises(ValueError, match="LocalPassActor"):
        local.LocalPassActor(obs, _GROUPS, **arguments)
