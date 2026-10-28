# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical routine evidence and actual RSL walking-to-ballet actor migration, without a simulator."""

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

_PATH = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/ballet.py"
_SPEC = importlib.util.spec_from_file_location("microduck_ballet_tests", _PATH)
ballet = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ballet
_SPEC.loader.exec_module(ballet)


def frame(yaw=0.0, *, both=False, pitch=0.0, head_z=0.0):
    def scalar(value):
        return torch.tensor([value], dtype=torch.float32)

    return ballet.BalletFrame(
        yaw=scalar(yaw),
        pitch=scalar(pitch),
        head_z=scalar(head_z),
        up=scalar(math.cos(pitch)),
        height=scalar(0.125),
        planar_speed=scalar(0),
        tumbling=scalar(0),
        yaw_rate=scalar(0),
        clearance=torch.tensor([[0.0, 0.0 if both else 0.04]]),
        forces=torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0 if both else 0.0]]]),
        nonfoot_contact=torch.tensor([False]),
        displacement=scalar(0),
        anchor_drift=scalar(0),
        com_error=scalar(0),
        finite=torch.tensor([True]),
    )


def start_turn(goal=2 * math.pi, yaw=0.0):
    routine = ballet.BalletRoutine(1, "cpu")
    routine.reset(torch.tensor([0]), torch.tensor([yaw]), torch.tensor([goal]))
    for _ in range(18):
        routine.update(frame(yaw), 0.02)
    assert int(routine.phase[0]) == 1
    return routine


def test_complete_turn_requires_net_root_rotation_then_a_grounded_body_and_head_bow_then_recovery():
    """The +/-pi crossing counts correctly; head motion, a partial turn, and half a bow cannot finish."""
    routine = start_turn(yaw=3.0)
    total_yaw = 3.0
    for _ in range(40):
        routine.update(frame(total_yaw, head_z=-0.6), 0.02)
    assert int(routine.phase[0]) == 1 and routine.progress[0] == 0
    for index in range(263):
        total_yaw += 0.024
        wrapped = math.atan2(math.sin(total_yaw), math.cos(total_yaw))
        routine.update(frame(wrapped), 0.02)
        if index < 250:
            assert int(routine.phase[0]) == 1
    assert int(routine.phase[0]) == 2 and not bool(routine.success[0])
    for _ in range(11):
        routine.update(frame(wrapped, both=True), 0.02)
    assert int(routine.phase[0]) == 3
    for pitch, head_z in ((0.0, -0.45), (0.30, 0.0)):
        for _ in range(20):
            routine.update(frame(wrapped, both=True, pitch=pitch, head_z=head_z), 0.02)
        assert int(routine.phase[0]) == 3
    for _ in range(18):
        routine.update(frame(wrapped, both=True, pitch=0.30, head_z=-0.45), 0.02)
    assert int(routine.phase[0]) == 4 and not bool(routine.success[0])
    for _ in range(26):
        routine.update(frame(wrapped, both=True), 0.02)
    assert bool(routine.success[0]) and not bool(routine.failed[0])


@pytest.mark.parametrize("loss", ("toe_touch", "flight", "wrong_foot", "low_clearance", "tilt", "head_contact"))
def test_a_single_physics_sample_without_valid_one_foot_support_erases_and_repays_partial_turn(loss):
    """A 5ms touch between policy actions cannot be hidden in a nominally one-legged turn."""
    routine = start_turn()
    for index in range(40):
        routine.update(frame((index + 1) * 0.005), 0.005)
    previous = routine.progress.clone()
    sample = frame(0.205)
    if loss == "toe_touch":
        sample.forces[0, 1, 2] = 0.5
    elif loss == "flight":
        sample.forces.zero_()
    elif loss == "wrong_foot":
        sample.forces[0, :, 2] = torch.tensor((0.0, 2.0))
    elif loss == "low_clearance":
        sample.clearance[0, 1] = 0.005
    elif loss == "tilt":
        sample.up[:] = 0.8
    else:
        sample.nonfoot_contact[:] = True
    routine.begin_control_step()
    routine.update(sample, 0.005)
    assert routine.progress[0] == 0 and int(routine.phase[0]) == 1
    torch.testing.assert_close(ballet.rotation_potential(previous, routine.progress, routine.failed), -previous)
    assert not bool(routine.success[0])


def test_wiggling_wrong_way_rotation_and_zero_turn_lessons_cannot_farm_spin_credit():
    routine = start_turn()
    for index in range(200):
        routine.update(frame(0.4 * math.sin(index * 0.2)), 0.02)
    assert not bool(routine.failed[0]) and int(routine.phase[0]) == 1
    assert abs(float(routine.progress[0])) < 0.5
    for index in range(100):
        routine.update(frame(-0.4 - 0.04 * index), 0.02)
    assert int(routine.phase[0]) == 1 and not bool(routine.success[0])
    balance = start_turn(goal=0.0)
    for index in range(25):
        balance.update(frame(index * 0.02), 0.02)
    assert float(balance.peak[0]) <= 0.03501


def test_a_turn_followed_by_a_fall_repays_the_whole_rotation_potential_only_once():
    routine = start_turn(goal=math.pi / 4)
    for index in range(40):
        routine.update(frame((index + 1) * 0.02), 0.02)
    assert int(routine.phase[0]) == 2
    banked = routine.progress.clone()
    sample = frame(0.8, both=True)
    sample.height[:] = 0.04
    routine.begin_control_step()
    routine.update(sample, 0.005)
    torch.testing.assert_close(ballet.rotation_potential(banked, routine.progress, routine.failed), -banked)
    assert bool(routine.failed[0]) and not bool(routine.success[0])
    routine.begin_control_step()
    routine.update(sample, 0.005)
    assert routine.progress[0] == 0


def test_discounted_rotation_shaping_has_zero_return_for_complete_and_abandoned_routines():
    """The actual PPO discount cancels transient spin potential even when its loss happens much later."""
    for path in ((0.0, 0.2, 0.8, 1.5, 0.0), (0.0, 1.0, 3.0, 6.28, 6.28, 6.28)):
        total = 0.0
        for index, (previous, current) in enumerate(zip(path, path[1:])):
            terminal = torch.tensor(index == len(path) - 2)
            value = ballet.rotation_potential(torch.tensor(previous), torch.tensor(current), terminal)
            total += ballet.REWARD_DISCOUNT**index * float(value)
        assert total == pytest.approx(0.0, abs=1e-6)


def test_curriculum_requires_consecutive_complete_routines_and_resumes_levels_without_inventing_a_streak():
    curriculum = ballet.BalletCurriculum(2, "cpu")
    ids = torch.tensor((0, 1))
    curriculum.record(ids, torch.tensor((True, False)))
    assert curriculum.stage.tolist() == [0, 0]
    curriculum.record(ids, torch.tensor((False, True)))
    curriculum.record(ids, torch.tensor((True, True)))
    assert curriculum.stage.tolist() == [0, 1]
    saved = curriculum.state_dict()
    restored = ballet.BalletCurriculum(4, "cpu")
    restored.load_state_dict(saved)
    assert restored.stage.tolist() == [0, 1, 0, 1] and restored.streak.tolist() == [0, 0, 0, 0]
    before = restored.stage.clone()
    with pytest.raises(ValueError, match="contract"):
        restored.load_state_dict({**saved, "angles": [0, 1, 2, 3, 4]})
    torch.testing.assert_close(restored.stage, before)


def test_inference_rollout_can_be_reset_outside_inference_mode_without_changing_other_worlds():
    """The simulator preflight and RSL wrapper cross inference-mode boundaries at reset."""
    routine = ballet.BalletRoutine(2, "cpu")
    ids = torch.tensor([0, 1])
    routine.reset(ids, torch.zeros(2), torch.full((2,), 2 * math.pi))
    sample = frame()
    for name in sample.__dataclass_fields__:
        value = getattr(sample, name)
        setattr(sample, name, value.expand(2, *value.shape[1:]).clone())
    with torch.inference_mode():
        for _ in range(30):
            routine.update(sample, 0.02)
    other = {name: value[1].clone() for name, value in vars(routine).items() if isinstance(value, torch.Tensor)}
    routine.reset(torch.tensor([0]), torch.tensor([1.0]), torch.tensor([math.pi]))
    for name, value in other.items():
        torch.testing.assert_close(getattr(routine, name)[1], value)
    assert int(routine.phase[0]) == 0 and routine.progress[0] == 0


def test_walking_actor_strictly_loads_into_real_ballet_model_and_keeps_actions_before_fine_tuning():
    """Actual RSL normalization, layers, servo order and appended zero weights preserve the source policy."""
    torch.manual_seed(42)
    groups = {"actor": ["policy"], "critic": ["policy"]}
    config = dict(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"},
    )
    old_obs = TensorDict({"policy": torch.randn(8, 61)}, batch_size=[8])
    new_obs = TensorDict({"policy": torch.cat((old_obs["policy"], torch.randn(8, 28)), -1)}, batch_size=[8])
    old = MLPModel(old_obs, groups, "actor", 14, **config).eval()
    old.update_normalization(old_obs)
    source = {key: value.clone() for key, value in old.state_dict().items()}
    transferred = ballet.walking_actor_warm_start(source)
    new = MLPModel(new_obs, groups, "actor", 14, **config).eval()
    new.load_state_dict(transferred, strict=True)
    torch.testing.assert_close(new(new_obs), old(old_obs), rtol=1e-5, atol=1e-6)
    assert all(parameter.requires_grad for parameter in new.parameters())
    for key, value in old.state_dict().items():
        torch.testing.assert_close(source[key], value)
    critic = MLPModel(new_obs, groups, "critic", 1, hidden_dims=[512, 256, 128], obs_normalization=True)
    algorithm = PPO(
        new,
        critic,
        RolloutStorage("rl", 8, 4, new_obs, [14]),
        num_learning_epochs=1,
        num_mini_batches=2,
        gamma=ballet.REWARD_DISCOUNT,
    )
    previous = new.mlp[-1].weight.detach().clone()
    with torch.inference_mode():
        for _ in range(4):
            algorithm.act(new_obs)
            algorithm.process_env_step(new_obs, torch.arange(8).float(), torch.zeros(8), {})
        algorithm.compute_returns(new_obs)
    losses = algorithm.update()
    assert all(math.isfinite(value) for value in losses.values())
    assert (new.mlp[0].weight[:, 61:].abs().sum(0) > 0).all()
    assert ((new.mlp[-1].weight - previous).abs().sum(1) > 0).all()
    damaged = dict(source)
    damaged["mlp.0.weight"] = torch.zeros(512, 71)
    with pytest.raises(ValueError, match="61-input"):
        ballet.walking_actor_warm_start(damaged)
