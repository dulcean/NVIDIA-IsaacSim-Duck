# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic contracts for volley trajectory diagnostics; no simulator or policy required."""

import pytest

from scripts.tools.trace_microduck_volley import predicted_miss, summarize


def _row(time, event="physics", player=None, episode=0):
    row = {
        "episode": episode,
        "time_s": time,
        "event": event,
        "strike": 0,
        "candidate_hit_a": int(player == "a"),
        "candidate_hit_b": int(player == "b"),
        "physical_head_onsets": int(player is not None),
        "ball_x": 0.0,
        "ball_y": 0.0,
        "ball_z": 0.3,
        "ball_v_x": 1.75,
        "ball_v_y": 0.0,
        "ball_v_z": 0.981,
        "home_head_a_x": 0.0,
        "home_head_a_y": 0.0,
        "home_head_a_z": 0.3,
        "home_head_b_x": 0.35,
        "home_head_b_y": 0.0,
        "home_head_b_z": 0.3,
        "root_a_x": -0.175,
        "root_b_x": 0.175,
        "tilt_a_deg": 0.0,
        "tilt_b_deg": 0.0,
    }
    for reason in (
        "fell",
        "robot_collision",
        "crossed_center",
        "ball_outside",
        "ball_ground",
        "invalid_contact",
        "timeout",
    ):
        row[reason] = False
    return row


def test_trajectory_report_distinguishes_accurate_first_and_wrong_direction_second_strike():
    """A known ballistic arc reaches B; the same velocity away from A misses A by 70 cm."""
    first = _row(0, "impact", "a")
    first["strike"] = 1
    after_first = _row(0.02)
    after_first.update(ball_x=0.035, ball_z=0.317658, ball_v_z=0.7848)
    second = _row(0.10, "impact", "b")
    second.update(strike=2, tilt_a_deg=30.0)
    after_second = _row(0.12)
    after_second.update(ball_x=0.385, ball_z=0.317658, ball_v_z=0.7848, tilt_a_deg=45.0, root_a_x=-0.10)
    terminal = _row(0.14, "terminal")
    terminal.update(crossed_center=True, tilt_a_deg=45.0, root_a_x=-0.10)
    report = summarize([first, after_first, second, after_second, terminal])
    assert report["completed_episodes"] == 1
    assert report["by_strike"]["first_strike"]["mean_predicted_home_head_miss_m"] == pytest.approx(0, abs=1e-6)
    assert report["by_strike"]["second_strike"]["mean_predicted_home_head_miss_m"] == pytest.approx(0.7)
    assert report["strikes"][0]["max_tilt_next_250ms_deg"] == 45
    assert report["strikes"][0]["inward_root_drift_next_250ms_m"] == pytest.approx(0.075)
    assert report["terminal_reason_fractions"]["crossed_center"] == 1


def test_prediction_does_not_borrow_samples_after_a_reset_or_another_head_contact():
    """Neither reset ball state nor a later impact is a valid outgoing-velocity sample."""
    first = _row(0, "impact", "a")
    first["strike"] = 1
    terminal = _row(0.01, "terminal")
    reset = _row(0.02, episode=1)
    report = summarize([first, terminal, reset])
    assert report["strikes"][0]["predicted_home_head_miss_m"] is None
    assert report["strikes"][0]["velocity_sample_delay_s"] is None
    contact = _row(0.01, "impact")
    contact["physical_head_onsets"] = 1  # Invalid head contact also invalidates a free-flight estimate.
    report = summarize([first, contact, _row(0.02)])
    assert report["strikes"][0]["predicted_home_head_miss_m"] is None


def test_unreachable_receiver_height_has_no_invented_ballistic_solution():
    row = _row(0)
    row.update(ball_z=0.1, ball_v_z=0.0)
    assert predicted_miss(row, "b") is None
