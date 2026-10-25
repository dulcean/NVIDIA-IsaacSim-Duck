# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts for volley turn order and robot collisions."""

import importlib.util
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/ball_rules.py"
_SPEC = importlib.util.spec_from_file_location("microduck_ball_rules", _PATH)
rules = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rules)


def test_alternating_onset_accepts_reply_without_flight_time_gate():
    """A reply from the other robot is legal; wrong turns, wrong surfaces, and persistent touches are not."""
    valid, surface, turn = rules.classify_head_contact(
        torch.tensor([True, True, True, False, True]),
        torch.tensor([True, True, False, True, True]),
        torch.tensor([1, 0, 1, 1, 1]),
        torch.tensor([0, 1, 0, 0, 0]),
        torch.tensor([False, False, False, False, True]),
        player=1,
        num_players=2,
    )
    assert valid.tolist() == [True, False, False, False, False]
    assert surface.tolist() == [False, False, True, False, False]
    assert turn.tolist() == [False, True, False, False, False]


def test_heading_still_allows_only_one_strike():
    """Removing the pair's flight-time gate must not turn single-robot heading into juggling."""
    valid, surface, turn = rules.classify_head_contact(
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([-1, 0]),
        torch.zeros(2, dtype=torch.bool),
        player=0,
        num_players=1,
    )
    assert valid.tolist() == [True, False]
    assert surface.tolist() == [False, False]
    assert turn.tolist() == [False, True]


def test_robot_contacts_do_not_cancel_or_count_subthreshold_noise():
    """Opposing body contacts still end a rally, while tiny numerical forces do not."""
    forces = torch.zeros(4, 2, 2, 3)
    forces[0, 0, 0, 0] = 1.0
    forces[0, 1, 1, 0] = -1.0
    forces[2, 0, 0, 2] = 0.005
    forces[3, 0, 0, 2] = float("nan")
    assert rules.robot_collision_mask(forces, 0.01).tolist() == [True, False, False, True]


def test_balance_detects_either_partner_tumbling_but_allows_yaw():
    """One standing partner must not mask a collapsing torso; turning in place remains allowed."""
    gravity = torch.tensor(
        [
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            [[0.8660254, 0.0, -0.5], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        ]
    )
    angular = torch.zeros(3, 2, 3)
    angular[0, :, 2] = 5.0
    angular[1, 0, 1] = 3.0
    angular[2, 1, 0] = 2.0
    tilt, tumbling = rules.balance_errors(gravity, angular)
    torch.testing.assert_close(tilt, torch.tensor([0.0, 0.5, 2.0]))
    torch.testing.assert_close(tumbling, torch.tensor([0.0, 9.0, 4.0]))


def test_recovery_conserves_strike_credit_and_pays_only_once():
    """Two recovered strikes retain the original seven-point budget, including immediate credit."""
    state = rules.RecoveryRewards(1, 2, "cpu")
    credit = torch.tensor([[0.8, 5.8]])
    hits = torch.ones_like(credit)
    stable = torch.ones_like(credit, dtype=torch.bool)
    ended = torch.tensor([False])
    paid, recovered = state.step(credit, hits, stable, ended, 0.02, 0.12, 0.10)
    assert paid.sum() == 0 and recovered.sum() == 0
    for _ in range(5):
        paid, _ = state.step(credit * 0, hits * 0, stable, ended, 0.02, 0.12, 0.10)
        assert paid.sum() == 0
    paid, recovered = state.step(credit * 0, hits * 0, stable, ended, 0.02, 0.12, 0.10)
    torch.testing.assert_close(paid.sum() + 0.4, torch.tensor(7.0))
    assert recovered.sum() == 2
    for _ in range(10):
        paid, recovered = state.step(credit * 0, hits * 0, stable, ended, 0.02, 0.12, 0.10)
        assert paid.sum() == 0 and recovered.sum() == 0


def test_terminal_step_cancels_credit_even_when_recovery_is_ready():
    """Neither a terminal strike nor a failure at payout time may cash unpaid bonuses."""
    state = rules.RecoveryRewards(2, 2, "cpu")
    credit = torch.tensor([[0.8, 5.8], [0.8, 5.8]])
    hits = torch.ones_like(credit)
    stable = torch.ones_like(credit, dtype=torch.bool)
    ongoing = torch.zeros(2, dtype=torch.bool)
    state.step(credit, hits, stable, ongoing, 0.02, 0.12, 0.10)
    for _ in range(5):
        state.step(credit * 0, hits * 0, stable, ongoing, 0.02, 0.12, 0.10)
    paid, recovered = state.step(credit * 0, hits * 0, stable, torch.tensor([True, False]), 0.02, 0.12, 0.10)
    assert paid[0].sum() == 0 and recovered[0].sum() == 0
    torch.testing.assert_close(paid[1].sum(), torch.tensor(6.6))
    # Two immediate hit bonuses followed by a fall cannot profit from aim or return credit.
    assert 0.4 + paid[0].sum() - 2.0 < 0
    paid, _ = state.step(credit, hits, stable, ~ongoing, 0.02, 0.12, 0.10)
    assert paid.sum() == 0 and state.pending.sum() == 0


def test_recovery_requires_continuous_stability_for_each_player_and_resets_per_world():
    """A balanced partner cannot unlock another player's credit; resetting one world cannot erase another."""
    state = rules.RecoveryRewards(2, 2, "cpu")
    credit = torch.ones(2, 2)
    hits = torch.ones_like(credit)
    stable = torch.ones_like(credit, dtype=torch.bool)
    ongoing = torch.zeros(2, dtype=torch.bool)
    state.step(credit, hits, stable, ongoing, 0.02, 0.12, 0.10)
    for _ in range(4):
        state.step(credit * 0, hits * 0, stable, ongoing, 0.02, 0.12, 0.10)
    stable[:, 1] = False
    state.step(credit * 0, hits * 0, stable, ongoing, 0.02, 0.12, 0.10)
    stable[:, 1] = True
    paid, recovered = state.step(credit * 0, hits * 0, stable, ongoing, 0.02, 0.12, 0.10)
    torch.testing.assert_close(paid, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert recovered.sum() == 2
    state.reset(torch.tensor([0]))
    for _ in range(4):
        paid, _ = state.step(credit * 0, hits * 0, stable, ongoing, 0.02, 0.12, 0.10)
    torch.testing.assert_close(paid, torch.tensor([[0.0, 0.0], [0.0, 1.0]]))


def test_another_own_strike_restarts_recovery_without_losing_credit():
    """Rapid repeated exchanges bank credit but cannot use stability preceding the newest own hit."""
    state = rules.RecoveryRewards(1, 1, "cpu")
    credit = torch.ones(1, 1)
    stable = torch.ones(1, 1, dtype=torch.bool)
    ended = torch.tensor([False])
    state.step(credit, credit, stable, ended, 0.02, 0.12, 0.10)
    for _ in range(5):
        state.step(credit * 0, credit * 0, stable, ended, 0.02, 0.12, 0.10)
    paid, _ = state.step(credit, credit, stable, ended, 0.02, 0.12, 0.10)
    assert paid.sum() == 0
    for _ in range(5):
        paid, _ = state.step(credit * 0, credit * 0, stable, ended, 0.02, 0.12, 0.10)
        assert paid.sum() == 0
    paid, recovered = state.step(credit * 0, credit * 0, stable, ended, 0.02, 0.12, 0.10)
    assert paid.sum() == 2 and recovered.sum() == 2


def test_miss_continues_simulation_until_deadline_without_restarting_clock():
    """A missed rally remains observable for 0.6 s; unrelated worlds keep playing."""
    state = rules.PostRallyRecovery(2, "cpu")
    miss = torch.tensor([True, False])
    no = torch.zeros(2, dtype=torch.bool)
    stable = ~no
    done, started, recovered = state.step(miss, no, no, stable, 0.2, 0.6, 0.2)
    assert started.tolist() == [True, False]
    assert not done.any() and not recovered.any()
    for _ in range(2):
        done, started, recovered = state.step(miss, no, no, stable, 0.2, 0.6, 0.2)
        assert not done.any() and not started.any() and not recovered.any()
    done, started, recovered = state.step(miss, no, no, stable, 0.2, 0.6, 0.2)
    assert done.tolist() == [True, False]
    assert recovered.tolist() == [True, False]
    assert not started.any()


def test_post_rally_checks_final_hold_and_safety_before_paying():
    """Early stability cannot hide later collapse; terminal safety and timeout override recovery."""
    state = rules.PostRallyRecovery(4, "cpu")
    yes = torch.ones(4, dtype=torch.bool)
    no = ~yes
    state.step(yes, no, no, yes, 0.2, 0.6, 0.4)
    state.step(no, no, no, yes, 0.2, 0.6, 0.4)
    stable = torch.tensor([True, False, True, True])
    done, _, recovered = state.step(no, no, no, stable, 0.2, 0.6, 0.4)
    assert not done.any() and not recovered.any()
    done, _, recovered = state.step(
        no,
        torch.tensor([False, False, False, True]),
        torch.tensor([False, False, True, False]),
        yes,
        0.2,
        0.6,
        0.4,
    )
    assert done.all()
    assert recovered.tolist() == [True, False, False, False]


def test_post_rally_fall_ends_immediately_and_subset_reset_preserves_other_world():
    """A collapse during the grace period must not wait for its timer or disappear into a reset."""
    state = rules.PostRallyRecovery(2, "cpu")
    yes = torch.ones(2, dtype=torch.bool)
    no = ~yes
    state.step(yes, no, no, yes, 0.2, 0.6, 0.2)
    done, _, recovered = state.step(no, torch.tensor([True, False]), no, yes, 0.2, 0.6, 0.2)
    assert done.tolist() == [True, False] and not recovered.any()
    state.reset(torch.tensor([0]))
    for _ in range(2):
        done, _, recovered = state.step(no, no, no, yes, 0.2, 0.6, 0.2)
    assert done.tolist() == [False, True]
    assert recovered.tolist() == [False, True]
