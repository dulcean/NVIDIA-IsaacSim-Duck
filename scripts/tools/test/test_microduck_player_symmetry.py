# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Player-local frame exchange and genuine RSL-RL learning integration, without physics."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

_PATH = (
    Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/player_symmetry.py"
)
_SPEC = importlib.util.spec_from_file_location("isaaclab_tasks.contrib.microduck.player_symmetry", _PATH)
symmetry = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = symmetry
_SPEC.loader.exec_module(symmetry)
_LIVE_SPEC = importlib.util.spec_from_file_location(
    "isaaclab_tasks.contrib.microduck.live_pass_player_symmetry", _PATH.with_name("live_pass_player_symmetry.py")
)
live_symmetry = importlib.util.module_from_spec(_LIVE_SPEC)
sys.modules[_LIVE_SPEC.name] = live_symmetry
_LIVE_SPEC.loader.exec_module(live_symmetry)


def test_exchange_preserves_local_signs_visual_history_and_critic_isolation():
    """A pi court rotation exchanges identities, including actions/history, without mirroring any local axis."""
    obs = TensorDict(
        {"policy": torch.arange(304).reshape(2, 152).float(), "critic": torch.arange(284).reshape(2, 142).float()},
        batch_size=[2],
    )
    before = obs.clone()
    actions = torch.arange(56).reshape(2, 28).float()
    augmented, output = symmetry.exchange_players(None, obs=obs, actions=actions)
    assert augmented.batch_size == torch.Size([4])
    torch.testing.assert_close(augmented[:2], before)
    # Ball channels, visibility, age and last-seen bearing retain their local signs.
    torch.testing.assert_close(augmented["policy"][2:, 61:76], before["policy"][:, 137:152])
    torch.testing.assert_close(augmented["policy"][2:, 137:152], before["policy"][:, 61:76])
    torch.testing.assert_close(augmented["critic"][2:, :71], before["critic"][:, 71:])
    torch.testing.assert_close(output[2:, :14], actions[:, 14:])
    torch.testing.assert_close(output[2:, 14:], actions[:, :14])
    restored, restored_actions = symmetry.exchange_players(None, obs=augmented[2:], actions=output[2:])
    torch.testing.assert_close(restored[2:], before)
    torch.testing.assert_close(restored_actions[2:], actions)
    obs["critic"].fill_(float("nan"))
    changed, _ = symmetry.exchange_players(None, obs=obs)
    torch.testing.assert_close(changed["policy"], augmented["policy"])
    torch.testing.assert_close(before["policy"], obs["policy"])
    with pytest.raises(ValueError, match="152"):
        symmetry.exchange_players(None, obs=TensorDict({"policy": torch.zeros(2, 146)}, batch_size=[2]))
    with pytest.raises(ValueError, match="14-servo"):
        symmetry.exchange_players(None, actions=torch.zeros(2, 14))


@pytest.mark.parametrize(
    "policy_width,exchange", [(152, symmetry.exchange_players), (178, live_symmetry.exchange_players)]
)
def test_actual_rsl_ppo_updates_both_players_with_exchange_and_consistency_loss(policy_width, exchange):
    """Exercise real rollout storage, augmentation, Gaussian log probabilities and PPO backward together."""
    torch.manual_seed(9)
    obs = TensorDict({"policy": torch.randn(4, policy_width), "critic": torch.randn(4, 142)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor_type, hidden_dims = MLPModel, [32, 16]
    if policy_width == 178:
        spec = importlib.util.spec_from_file_location(
            "isaaclab_tasks.contrib.microduck.live_pass_actor", _PATH.with_name("live_pass_actor.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        actor_type, hidden_dims = module.LocalPassActor, [1024, 512, 256]
    actor = actor_type(
        obs,
        groups,
        "actor",
        28,
        hidden_dims=hidden_dims,
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.15, "std_type": "scalar"},
    )
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[32, 16], obs_normalization=True)
    storage = RolloutStorage("rl", 4, 4, obs, [28])
    algorithm = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=2,
        symmetry_cfg={
            "env": None,
            "data_augmentation_func": exchange,
            "use_data_augmentation": True,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 1.0,
        },
    )
    if policy_width == 178:
        # Match the actual warm-start adapter: added partner/planner channels
        # initially have no influence, but must acquire weights through PPO.
        added = torch.tensor((*range(76, 89), *range(165, 178)))
        with torch.no_grad():
            actor.mlp[0].weight.index_fill_(1, added, 0.0)
    before = actor.mlp[-1].weight.detach().clone()
    with torch.inference_mode():
        for _ in range(4):
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.tensor([0.0, 3.0, 0.0, 3.0]), torch.zeros(4), {})
        algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert all(math.isfinite(value) for value in losses.values())
    assert losses["symmetry"] > 0
    assert torch.isfinite(actor(obs)).all()
    assert ((actor.mlp[-1].weight - before).abs().sum(-1) > 0).all()
    if policy_width == 178:
        assert (actor.mlp[0].weight[:, 76:89].abs().sum(0) > 0).all()
        assert (actor.mlp[0].weight[:, 165:178].abs().sum(0) > 0).all()
