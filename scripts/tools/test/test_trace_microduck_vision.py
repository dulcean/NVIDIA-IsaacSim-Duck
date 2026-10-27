# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured-sample diagnostic boundaries for the actual visual curriculum targets."""

import pytest

from scripts.tools.trace_microduck_vision import flight_diagnostics, summarize_flights


def row(time, *, world=0, episode=1, event="physics", hit=False, contact=False):
    sample = {
        "world": world,
        "episode": episode,
        "event": event,
        "time_s": time,
        "candidate_hit_a": hit,
        "candidate_hit_b": False,
        "contact_a": contact,
        "contact_b": False,
        "raw_head_onsets": int(hit),
        "ball_x": 1.75 * time,
        "ball_y": 0.0,
        "ball_z": 0.3 + 0.981 * time - 0.5 * 9.81 * time**2,
        "ball_v_x": 1.75,
        "ball_v_y": 0.0,
        "ball_v_z": 0.981 - 9.81 * time,
        "target_a_x": 0.35,
        "target_a_y": 0.0,
        "target_a_z": 0.3,
        # The parked robot's physical head must not replace the virtual disk.
        "home_head_b_x": 0.8,
        "home_head_b_y": 0.0,
        "home_head_b_z": 0.3,
    }
    return sample


def test_visual_flight_uses_actual_target_and_waits_for_separation():
    """A known physical arc hits the virtual disk, despite a farther parked head and lingering contact."""
    rows = [row(0, hit=True, contact=True), row(0.02, contact=True), row(0.025)]
    report = summarize_flights(rows, [])
    strike = report["strikes"][0]
    assert strike["sample_delay_s"] == pytest.approx(0.025)
    assert strike["flight"]["predicted_miss_m"] == pytest.approx(0, abs=1e-9)
    assert strike["flight"]["toward_target_speed_m_s"] == pytest.approx(1.75)
    assert report["by_player"]["a"]["quality_gate_passes"] == 1
    assert report["by_player"]["b"]["mean_predicted_miss_m"] is None


@pytest.mark.parametrize("boundary", ["reset", "next_contact", "other_world", "terminal"])
def test_visual_outgoing_samples_do_not_cross_physical_boundaries(boundary):
    later = row(0.025)
    rows = [row(0, hit=True)]
    if boundary == "reset":
        later["episode"] = 2
    elif boundary == "other_world":
        later["world"] = 1
    elif boundary == "next_contact":
        interruption = row(0.01)
        interruption["raw_head_onsets"] = 1  # Even an invalid head contact changes the trajectory.
        rows.append(interruption)
    else:
        rows.append(row(0.01, event="terminal"))
    rows.append(later)
    report = summarize_flights(rows, [])
    assert report["strikes"][0]["flight"] is None


def test_visual_flight_distinguishes_wrong_direction_unreachable_and_elapsed_horizon():
    away = row(0.02)
    away["ball_v_x"] = -1.75
    result = flight_diagnostics(away, "a")
    assert result["toward_target_speed_m_s"] < 0 and result["predicted_miss_m"] > 0.6
    low = row(0)
    low.update(ball_z=0.1, ball_v_z=0.0)
    result = flight_diagnostics(low, "a")
    assert not result["reachable_plane"] and result["predicted_miss_m"] is None
    short = row(0)
    short.update(ball_x=0.035, ball_z=0.314715, ball_v_z=0.63765, ball_v_x=0.3 / 0.17)
    result = flight_diagnostics(short, "a", elapsed_since_hit=0.02)
    assert result["quality_gate"] and not result["legacy_quality_gate"]
