# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent SI-unit and feasibility contracts for camera-guided pass references."""

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/passing.py"
_SPEC = importlib.util.spec_from_file_location("microduck_passing", _PATH)
passing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(passing)


def test_incoming_prediction_uses_gravity_and_moving_head_height_before_planning_a_pass():
    """A 300ms falling arc meets the translating head at x=-.17 and z=.33m."""
    # Independently constructed trajectory: z(.3)=.6+.5715*.3-4.905*.3**2=.33.
    contact = passing.predict_descending_contact(
        torch.tensor([[-0.20, 0.0, 0.60]]),
        torch.tensor([[0.10, 0.0, 0.5715]]),
        torch.tensor([[-0.20, 0.0, 0.30]]),
        torch.tensor([[0.10, 0.0, 0.10]]),
    )
    assert contact.valid.tolist() == [True]
    torch.testing.assert_close(contact.time, torch.tensor([0.3]))
    torch.testing.assert_close(contact.position, torch.tensor([[-0.17, 0.0, 0.33]]))
    torch.testing.assert_close(contact.velocity, torch.tensor([[0.10, 0.0, -2.3715]]))

    # The partner has also moved during the incoming flight; passing uses that
    # future reception reference, rather than its position when first observed.
    partner = torch.tensor([[0.16, 0.0, 0.30]])
    partner_velocity = torch.tensor([[0.10, 0.0, 0.0]])
    plan = passing.plan_ballistic_pass(
        contact.position,
        contact.velocity,
        partner + partner_velocity * contact.time.unsqueeze(-1),
        partner_velocity,
        min_flight=0.3,
        max_flight=0.3,
    )
    assert plan.valid.tolist() == [True]
    torch.testing.assert_close(plan.intercept_position, torch.tensor([[0.22, 0.0, 0.30]]))
    torch.testing.assert_close(plan.outgoing_velocity, torch.tensor([[1.3, 0.0, 1.3715]]))


def test_pass_exactly_intercepts_a_moving_partner_descending_and_reports_ball_momentum_change():
    """A fixed 300ms arc gives an independently known moving-target solution."""
    incoming = torch.tensor([[0.0, 0.0, -1.0]])
    plan = passing.plan_ballistic_pass(
        torch.tensor([[-0.15, 0.0, 0.30]]),
        incoming,
        torch.tensor([[0.15, 0.04, 0.28]]),
        torch.tensor([[0.10, -0.20, 0.05]]),
        min_flight=0.3,
        max_flight=0.3,
        ball_mass=0.005,
    )
    assert plan.valid.tolist() == [True]
    torch.testing.assert_close(plan.flight_time, torch.tensor([0.3]))
    torch.testing.assert_close(plan.intercept_position, torch.tensor([[0.18, -0.02, 0.295]]))
    torch.testing.assert_close(plan.outgoing_velocity, torch.tensor([[1.1, -1 / 15, 1.454833333]]))
    torch.testing.assert_close(plan.delta_velocity, torch.tensor([[1.1, -1 / 15, 2.454833333]]))
    torch.testing.assert_close(plan.impulse, torch.tensor([[0.0055, -1 / 3000, 0.01227416667]]))
    # Independent endpoint integration checks both horizontal motion and gravity.
    endpoint = torch.tensor([[-0.15, 0.0, 0.30]]) + 0.3 * plan.outgoing_velocity
    endpoint[:, 2] -= 0.44145
    torch.testing.assert_close(endpoint, plan.intercept_position)
    assert (plan.outgoing_velocity[:, 2] - 2.943 < 0).all()


def test_candidate_choice_is_a_modest_feasible_arc_in_both_world_directions():
    """Translated courts and player exchange preserve the physical passing solution."""
    position = torch.tensor([[[-0.175, 0.0, 0.30], [10.175, 0.0, 0.30]]])
    partner = torch.tensor([[[0.175, 0.0, 0.30], [9.825, 0.0, 0.30]]])
    incoming = torch.tensor([[[0.0, 0.0, -2.0], [0.0, 0.0, -2.0]]])
    plan = passing.plan_ballistic_pass(
        position, incoming, partner, torch.zeros_like(partner), center_x=torch.tensor([[0, 10]])
    )
    assert plan.valid.all()
    torch.testing.assert_close(plan.flight_time[..., 0], plan.flight_time[..., 1])
    torch.testing.assert_close(plan.outgoing_velocity[..., 0, 0], -plan.outgoing_velocity[..., 1, 0])
    assert (plan.flight_time >= 0.2).all() and (plan.flight_time <= 0.45).all()
    assert (plan.outgoing_velocity.norm(dim=-1) < 2).all()
    crossing_time = 0.175 / plan.outgoing_velocity[..., 0, 0]
    crossing_height = 0.3 + plan.outgoing_velocity[..., 0, 2] * crossing_time - 4.905 * crossing_time.square()
    assert (crossing_height >= 0.2).all()


@pytest.mark.parametrize(
    "ball, incoming, partner, options",
    [
        ([-0.15, 0.0, float("nan")], [0, 0, -1], [0.15, 0, 0.3], {}),
        ([-0.15, 0.0, 0.3], [0, 0, -1], [-0.10, 0, 0.3], {}),
        ([-0.15, 0.0, 0.1], [0, 0, -1], [0.15, 0, 0.1], {"min_flight": 0.1, "max_flight": 0.1}),
        ([-0.15, 0.0, 0.3], [0, 0, -1], [0.15, 0, 2.0], {}),
        ([-0.15, 0.0, 0.3], [0, 0, -3], [0.15, 0, 0.3], {"max_delta_speed": 0.1}),
        (
            [-0.15, 0.0, 0.3],
            [0, 0, -1],
            [0.15, 0, 1.0],
            {"min_flight": 0.2, "max_flight": 0.2, "max_speed": 10.0},
        ),
    ],
    ids=[
        "missing-rgb",
        "same-half",
        "low-center-crossing",
        "speed-unreachable",
        "impulse-unreachable",
        "ascending-arrival",
    ],
)
def test_impossible_arcs_are_explicitly_invalid_and_cannot_earn_release_credit(ball, incoming, partner, options):
    plan = passing.plan_ballistic_pass(
        torch.tensor([ball], dtype=torch.float32),
        torch.tensor([incoming], dtype=torch.float32),
        torch.tensor([partner], dtype=torch.float32),
        torch.zeros(1, 3),
        **options,
    )
    assert not plan.valid.any()
    for value in plan[:-1]:
        assert torch.isfinite(value).all() and not value.any()
    assert not passing.release_velocity_score(torch.zeros(1, 3), plan).any()


def test_release_score_improves_with_useful_impulse_but_rejects_overspeed_and_lateral_shots():
    """Weak actual shots receive distinct feedback; the optimum is the targeted velocity."""
    plan = passing.plan_ballistic_pass(
        torch.tensor([[-0.15, 0.0, 0.30]] * 7),
        torch.tensor([[0.0, 0.0, -2.0]] * 7),
        torch.tensor([[0.15, 0.0, 0.30]] * 7),
        torch.zeros(7, 3),
        min_flight=0.3,
        max_flight=0.3,
    )
    actual = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0, 0.2], [0.6, 0, 0.8], [1, 0, 1.4715], [4, 0, 4], [0, 3, 1.4715], [0, 0, 0]]
    )
    actual[-1, 0] = float("nan")
    score = passing.release_velocity_score(actual, plan)
    assert 0 < score[0] < score[1] < score[2] < score[3]
    assert score[3].item() == pytest.approx(1.0)
    assert score[4] < score[0] and score[5] < score[0]
    assert score[6] == 0
    # Same vector error in opposite directions has the same bounded score.
    actual[:2] = plan.outgoing_velocity[:2]
    actual[0, 0] += 0.75
    actual[1, 0] -= 0.75
    torch.testing.assert_close(passing.release_velocity_score(actual, plan)[:2], torch.tensor([0.5, 0.5]))


def test_incoming_prediction_rejects_passed_balls_and_out_of_reach_or_missing_estimates():
    contact = passing.predict_descending_contact(
        torch.tensor([[-0.15, 0.0, 0.2], [0.5, 0.0, 0.6], [-0.15, 0.0, float("nan")]]),
        torch.tensor([[0.0, 0.0, -1.0]] * 3),
        torch.tensor([[-0.15, 0.0, 0.3]] * 3),
        torch.zeros(3, 3),
    )
    assert not contact.valid.any()
    assert torch.isfinite(contact.position).all() and not contact.position.any()
    assert torch.isfinite(contact.velocity).all() and not contact.velocity.any()
    assert torch.isfinite(contact.time).all() and not contact.time.any()
