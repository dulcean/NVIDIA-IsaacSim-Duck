# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ballistic incoming serves, independent of the simulator and policy state."""

import torch


def frontal_serve_source_offset(
    receiver: torch.Tensor,
    partner: torch.Tensor,
    lateral_fraction: torch.Tensor,
    distance: float,
    *,
    distance_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Return a reset-only frontal offset, optionally bounding total XY launch distance.

    The ordinary offset adds ``distance`` beyond the sampled fraction of the
    head-to-head gap. Limits bound their *sum* from the receiver, keeping a
    wider live layout physically reachable without putting near serves above
    the camera's field of view. No in-flight ball state is modified.
    """
    direction = partner - receiver
    direction = torch.cat((direction[:, :2], torch.zeros_like(direction[:, 2:3])), dim=-1)
    unit = torch.nn.functional.normalize(direction, dim=-1)
    if distance_limits is None:
        return unit * distance
    low, high = distance_limits
    if not 0 < low <= high:
        raise ValueError("Frontal serve distance limits must be positive and ordered.")
    sampled_distance = direction.norm(dim=-1) * lateral_fraction
    total_distance = (sampled_distance + distance).clamp(low, high)
    return unit * (total_distance - sampled_distance)[:, None]


def descending_target_crossing(
    previous: torch.Tensor, current: torch.Tensor, target: torch.Tensor, radius: float
) -> torch.Tensor:
    """Detect an observed descending segment crossing a horizontal target disk.

    Positions have shape ``(N, 3)`` in one common frame. The ball need not be
    inside the disk at either sampled endpoint: interpolate its XY position at
    the target's height. Both segment endpoints are included. Nonfinite input,
    upward/level segments, and crossings outside the sampled segment fail.
    Callers latch a successful crossing so a boundary endpoint cannot pay twice.
    This measures the observed path; it does not predict a future ballistic hit.
    """
    if radius < 0:
        raise ValueError("The target radius cannot be negative.")
    if (
        previous.ndim != 2
        or previous.shape[-1] != 3
        or current.shape != previous.shape
        or target.shape != previous.shape
    ):
        raise ValueError("Previous, current and target positions must have matching (N, 3) shapes.")
    delta = current - previous
    descending = delta[:, 2] < 0
    safe_delta_z = torch.where(descending, delta[:, 2], torch.ones_like(delta[:, 2]))
    alpha = (target[:, 2] - previous[:, 2]) / safe_delta_z
    crossing_xy = previous[:, :2] + delta[:, :2] * alpha.unsqueeze(-1)
    distance_squared = (crossing_xy - target[:, :2]).square().sum(dim=-1)
    finite = torch.isfinite(torch.cat((previous, current, target), dim=-1)).all(dim=-1)
    return finite & descending & (alpha >= 0) & (alpha <= 1) & (distance_squared <= radius**2)


def incoming_ballistic_serve(
    receiver: torch.Tensor,
    partner: torch.Tensor,
    flight_time: torch.Tensor,
    height: torch.Tensor,
    lateral_fraction: torch.Tensor,
    arrival_offset: torch.Tensor,
    *,
    source_offset: torch.Tensor | None = None,
    gravity: float = 9.81,
    min_center_height: float = 0.05,
    min_arrival_down_speed: float = 0.5,
    max_arrival_speed: float = 3.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct an incoming trajectory that descends onto a chosen strike point.

    All positions use the same frame, with +Z up and the ground at Z=0. Inputs
    have shape ``(N, 3)`` for positions/offsets and ``(N,)`` for scalar samples.
    The caller supplies samples: this function has no random generator or
    curriculum state. A lateral fraction of zero starts vertically above the
    receiver; one starts horizontally above the partner. ``height`` is the
    source's height above the arrival target, before applying ``source_offset``.

    Only the reset position and velocity are constructed. Gravity, collisions,
    and the robot's movement determine the subsequent physical flight. The
    arrival target is a point for the *ball center*, including any desired
    clearance from the forehead surface in ``arrival_offset``.

    Returns initial position, initial linear velocity, arrival target, and a
    per-world validity mask. Invalid samples must be rejected by the caller;
    the mask rejects nonfinite values, nonpositive time, out-of-range lateral
    fractions, below-floor endpoints, excessive arrival speed, and lateral
    throws moving away from the receiver. Because a gravity-only vertical
    trajectory is concave, endpoints above the floor imply the whole arc is
    above it. This does not check the intervening robot geometry.
    """
    if gravity <= 0 or min_arrival_down_speed < 0 or max_arrival_speed <= 0:
        raise ValueError("Gravity and maximum speed must be positive; minimum downward speed cannot be negative.")
    if min_center_height < 0:
        raise ValueError("The minimum ball-center height cannot be negative.")
    if receiver.ndim != 2 or receiver.shape[-1] != 3 or partner.shape != receiver.shape:
        raise ValueError("Receiver and partner must have matching (N, 3) shapes.")
    if arrival_offset.shape != receiver.shape or (source_offset is not None and source_offset.shape != receiver.shape):
        raise ValueError("Arrival and source offsets must have the receiver's (N, 3) shape.")
    batch_shape = receiver.shape[:-1]
    if any(value.shape != batch_shape for value in (flight_time, height, lateral_fraction)):
        raise ValueError("Flight time, source height and lateral fraction must each have shape (N,).")

    target = receiver + arrival_offset
    source = receiver + lateral_fraction.unsqueeze(-1) * (partner - receiver)
    source = torch.cat((source[:, :2], (target[:, 2] + height).unsqueeze(-1)), dim=-1)
    if source_offset is not None:
        source = source + source_offset
    # Invalid zero/negative times must not cause a divide-by-zero in a reset batch.
    time = flight_time.clamp_min(1e-6).unsqueeze(-1)
    acceleration = receiver.new_tensor((0.0, 0.0, -gravity))
    velocity = (target - source) / time - 0.5 * acceleration * time
    arrival_velocity = velocity + acceleration * time

    finite = torch.isfinite(torch.cat((source, velocity, target, arrival_velocity), dim=-1)).all(dim=-1)
    toward_receiver = receiver[:, :2] - partner[:, :2]
    lateral = lateral_fraction > 1e-6
    moving_toward = (velocity[:, :2] * toward_receiver).sum(dim=-1) > 0
    valid = (
        finite
        & (flight_time > 0)
        & torch.isfinite(height)
        & (height >= 0)
        & (lateral_fraction >= 0)
        & (lateral_fraction <= 1)
        & (source[:, 2] >= min_center_height)
        & (target[:, 2] >= min_center_height)
        & (arrival_velocity[:, 2] <= -min_arrival_down_speed)
        & (arrival_velocity.norm(dim=-1) <= max_arrival_speed)
        & (~lateral | moving_toward)
    )
    return source, velocity, target, valid
