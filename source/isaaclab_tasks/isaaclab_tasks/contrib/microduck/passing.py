# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched ballistic references for physical, camera-guided MicroDuck passes.

Positions and velocities share one Cartesian frame, in metres and metres/second;
positive Z points upward. The helpers neither read simulator state nor apply an
impulse. Actor callers supply camera estimates; reward callers may supply measured
physical states. Targets are ball-centre reception points, including the ball
radius and head-surface offset before calling these helpers.
"""

from typing import NamedTuple

import torch


class ContactPrediction(NamedTuple):
    """Ball state at its descending crossing of the moving own-head height."""

    position: torch.Tensor
    velocity: torch.Tensor
    time: torch.Tensor
    valid: torch.Tensor


class PassPlan(NamedTuple):
    """One feasible flight reference; invalid entries contain finite zero values.

    ``impulse`` is the necessary *ball* momentum change in N s, not an actuator
    command. Head speed, surface orientation, compliance, and balance still
    determine whether the robot can produce it through a real collision.
    """

    outgoing_velocity: torch.Tensor
    delta_velocity: torch.Tensor
    impulse: torch.Tensor
    flight_time: torch.Tensor
    intercept_position: torch.Tensor
    valid: torch.Tensor


def _vectors(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Broadcast floating Cartesian vectors, preserving their shared device."""
    if any(value.ndim == 0 or value.shape[-1] != 3 or not value.is_floating_point() for value in values):
        raise ValueError("Expected floating Cartesian vectors with trailing dimension 3.")
    if any(value.device != values[0].device for value in values):
        raise ValueError("All Cartesian vectors must be on the same device.")
    return torch.broadcast_tensors(*values)


def predict_descending_contact(
    ball_position: torch.Tensor,
    ball_velocity: torch.Tensor,
    head_position: torch.Tensor,
    head_velocity: torch.Tensor,
    *,
    max_time: float = 0.6,
    max_horizontal_error: float = 0.18,
    gravity: float = 9.81,
) -> ContactPrediction:
    """Predict the incoming ball at a reachable descending head-height crossing.

    The head reception point follows constant velocity; the ball follows gravity.
    The horizontal reach bound permits ordinary repositioning, but rejects a
    mathematically valid crossing far outside the robot's reachable workspace.
    A zero-time descending crossing is valid, allowing the same reference at
    contact. Missing/stale camera measurements must additionally be masked by the
    caller; finite values alone cannot establish that an estimate is observed.
    """
    if max_time <= 0 or max_horizontal_error <= 0 or gravity <= 0:
        raise ValueError("Prediction horizon, horizontal reach, and gravity must be positive.")
    ball_position, ball_velocity, head_position, head_velocity = _vectors(
        ball_position, ball_velocity, head_position, head_velocity
    )
    finite = torch.stack(
        [torch.isfinite(value).all(dim=-1) for value in (ball_position, ball_velocity, head_position, head_velocity)]
    ).all(dim=0)
    ball_position, ball_velocity, head_position, head_velocity = (
        torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        for value in (ball_position, ball_velocity, head_position, head_velocity)
    )
    relative_height = ball_position[..., 2] - head_position[..., 2]
    relative_speed = ball_velocity[..., 2] - head_velocity[..., 2]
    discriminant = relative_speed.square() + 2 * gravity * relative_height
    time = (relative_speed + discriminant.clamp_min(0).sqrt()) / gravity
    position = ball_position + ball_velocity * time.unsqueeze(-1)
    position = position.clone()
    position[..., 2] -= 0.5 * gravity * time.square()
    velocity = ball_velocity.clone()
    velocity[..., 2] -= gravity * time
    head_at_contact = head_position + head_velocity * time.unsqueeze(-1)
    horizontal_error = (position[..., :2] - head_at_contact[..., :2]).norm(dim=-1)
    valid = (
        finite
        & (discriminant > 0)
        & (time >= 0)
        & (time <= max_time)
        & (horizontal_error <= max_horizontal_error)
        & (velocity[..., 2] < 0)
        & (velocity[..., 2] - head_velocity[..., 2] < 0)
    )
    return ContactPrediction(
        torch.where(valid.unsqueeze(-1), position, 0.0),
        torch.where(valid.unsqueeze(-1), velocity, 0.0),
        torch.where(valid, time, 0.0),
        valid,
    )


def plan_ballistic_pass(
    ball_position: torch.Tensor,
    ball_velocity: torch.Tensor,
    partner_position: torch.Tensor,
    partner_velocity: torch.Tensor,
    *,
    min_flight: float = 0.20,
    max_flight: float = 0.45,
    num_candidates: int = 33,
    max_speed: float = 3.5,
    max_delta_speed: float = 8.0,
    min_descent_speed: float = 0.10,
    min_crossing_height: float = 0.20,
    center_x: torch.Tensor | float = 0.0,
    gravity: float = 9.81,
    ball_mass: float = 0.005,
) -> PassPlan:
    """Choose a modest descending arc to the predicted moving partner reception point.

    A target does not define one unique ideal arc. Candidate flight times span
    the configured horizon. Each candidate exactly intercepts the partner's
    linear prediction, and must cross the court's X centre plane above the
    required height, descend at reception, and satisfy outgoing/change-of-speed
    bounds. Among feasible arcs, prefer low outgoing kinetic energy and modest
    change of velocity; a small time cost breaks near ties in favour of shorter
    flights. These are kinematic bounds, not a guarantee of actuator feasibility.

    ``ball_velocity`` is the anticipated *incoming* contact velocity. Advance
    the partner estimate to that contact time before calling this helper. The
    same helper can score an observed physical release using the remaining
    flight horizon. All leading dimensions broadcast, including ``center_x``.
    """
    if not (0 < min_flight <= max_flight) or num_candidates < 1:
        raise ValueError("Flight times must be positive and ordered, with at least one candidate.")
    if min(max_speed, max_delta_speed, gravity, ball_mass) <= 0 or min_descent_speed < 0:
        raise ValueError("Speed bounds, gravity, and mass must be positive; descent speed must be nonnegative.")
    ball_position, ball_velocity, partner_position, partner_velocity = _vectors(
        ball_position, ball_velocity, partner_position, partner_velocity
    )
    center = torch.as_tensor(center_x, device=ball_position.device, dtype=ball_position.dtype)
    center = torch.broadcast_to(center, ball_position.shape[:-1])
    finite = torch.stack(
        [
            torch.isfinite(value).all(dim=-1)
            for value in (ball_position, ball_velocity, partner_position, partner_velocity)
        ]
    ).all(dim=0) & torch.isfinite(center)
    ball_position, ball_velocity, partner_position, partner_velocity = (
        torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        for value in (ball_position, ball_velocity, partner_position, partner_velocity)
    )
    times = torch.linspace(
        min_flight, max_flight, num_candidates, device=ball_position.device, dtype=ball_position.dtype
    )
    target = partner_position.unsqueeze(-2) + partner_velocity.unsqueeze(-2) * times.unsqueeze(-1)
    outgoing = (target - ball_position.unsqueeze(-2)) / times.unsqueeze(-1)
    outgoing[..., 2] += 0.5 * gravity * times
    delta = outgoing - ball_velocity.unsqueeze(-2)
    speed = outgoing.norm(dim=-1)
    delta_speed = delta.norm(dim=-1)
    reception_speed_z = outgoing[..., 2] - gravity * times
    # Dividing a zero horizontal speed by a safe sentinel preserves finite output;
    # the explicit crossing-time and speed checks still reject that candidate.
    vx = outgoing[..., 0]
    crossing = (center - ball_position[..., 0]).unsqueeze(-1) / torch.where(vx.abs() > 1e-8, vx, 1.0)
    crossing_height = (
        ball_position[..., 2].unsqueeze(-1) + outgoing[..., 2] * crossing - 0.5 * gravity * crossing.square()
    )
    feasible = (
        finite.unsqueeze(-1)
        & torch.isfinite(outgoing).all(dim=-1)
        & (speed <= max_speed)
        & (delta_speed <= max_delta_speed)
        & (reception_speed_z <= -min_descent_speed)
        & (reception_speed_z - partner_velocity[..., 2].unsqueeze(-1) <= -min_descent_speed)
        & (vx.abs() > 1e-8)
        & (crossing > 0)
        & (crossing < times)
        & (crossing_height >= min_crossing_height)
    )
    cost = (speed / max_speed).square() + 0.15 * (delta_speed / max_delta_speed).square()
    cost += 0.05 * (times / max_flight).square()
    chosen = torch.where(feasible, cost, float("inf")).argmin(dim=-1)
    valid = feasible.any(dim=-1)
    index = chosen.unsqueeze(-1).unsqueeze(-1).expand(*chosen.shape, 1, 3)
    selected_velocity = outgoing.gather(-2, index).squeeze(-2)
    selected_delta = delta.gather(-2, index).squeeze(-2)
    selected_target = target.gather(-2, index).squeeze(-2)
    selected_time = times[chosen]
    selected_velocity = torch.where(valid.unsqueeze(-1), selected_velocity, 0.0)
    selected_delta = torch.where(valid.unsqueeze(-1), selected_delta, 0.0)
    return PassPlan(
        selected_velocity,
        selected_delta,
        ball_mass * selected_delta,
        torch.where(valid, selected_time, 0.0),
        torch.where(valid.unsqueeze(-1), selected_target, 0.0),
        valid,
    )


def release_velocity_score(
    actual_velocity: torch.Tensor, plan: PassPlan, *, velocity_tolerance: float = 0.75
) -> torch.Tensor:
    """Score actual post-separation velocity against the feasible pass, in [0, 1].

    A Cauchy error in m/s provides feedback even for weak shots that never reach
    the reception height. Excess speed and lateral/vertical error reduce the
    same score; speed by itself earns no reward. One tolerance-sized vector
    error scores 0.5, an exact arc scores 1, and invalid plans/nonfinite velocity
    score 0. This is a training reference, not a physical success criterion.
    """
    if velocity_tolerance <= 0:
        raise ValueError("Velocity tolerance must be positive, in metres/second.")
    actual_velocity, desired = _vectors(actual_velocity, plan.outgoing_velocity)
    error = ((actual_velocity - desired) / velocity_tolerance).square().sum(dim=-1)
    score = 1 / (1 + error)
    return torch.where(plan.valid & torch.isfinite(actual_velocity).all(dim=-1), score, 0.0)
