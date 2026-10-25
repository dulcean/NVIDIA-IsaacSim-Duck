# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent analytic checks for the measured ball-physics report."""

import math

import pytest

from scripts.tools.check_microduck_ball_physics import analyze_trace


def _analytic_drop(gravity=9.81, restitution=0.6):
    height, radius = 0.5, 0.025
    impact_time = math.sqrt(2 * (height - radius) / gravity)
    impact_speed = gravity * impact_time
    result = []
    for step in range(301):
        time = step * 0.005
        if time <= impact_time:
            z, vz = height - gravity * time**2 / 2, -gravity * time
        else:
            flight = time - impact_time
            z = radius + restitution * impact_speed * flight - gravity * flight**2 / 2
            vz = restitution * impact_speed - gravity * flight
            if z < radius:
                break
        result.append((time, z, vz))
    return result


def test_report_recovers_known_gravity_and_passive_rebound():
    report = analyze_trace(_analytic_drop(), 0.025)
    assert report["measured_acceleration_m_s2"] == pytest.approx(-9.81, abs=1e-8)
    assert report["estimated_restitution"] == pytest.approx(0.6, abs=0.03)
    assert report["first_rebound_height_above_floor_m"] == pytest.approx(0.171, abs=0.001)
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    "gravity,restitution,bad_check",
    [
        (4.0, 0.6, "gravity_within_5_percent"),
        (9.81, 1.2, "no_passive_energy_gain"),
    ],
)
def test_report_rejects_wrong_gravity_or_energy_creating_bounce(gravity, restitution, bad_check):
    report = analyze_trace(_analytic_drop(gravity, restitution), 0.025)
    assert not report["checks"][bad_check]
