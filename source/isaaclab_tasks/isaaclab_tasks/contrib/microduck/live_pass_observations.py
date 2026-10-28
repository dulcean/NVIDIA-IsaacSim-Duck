# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera-only ball/partner pass features, isolated from simulator target truth."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .ball_vision import VisualBallTracks, visual_actor_observations
from .passing import plan_ballistic_pass, predict_descending_contact


def _inverse_rotate(vector: torch.Tensor, quat_xyzw: torch.Tensor) -> torch.Tensor:
    inverse = -quat_xyzw[..., :3]
    first = 2 * torch.cross(inverse, vector, dim=-1)
    return vector + quat_xyzw[..., 3:] * first + torch.cross(inverse, first, dim=-1)


def _recent_track(track: VisualBallTracks, max_age: float) -> torch.Tensor:
    return (
        track.valid
        & torch.isfinite(track.position_w).all(-1)
        & torch.isfinite(track.velocity_w).all(-1)
        & torch.isfinite(track.age)
        & (track.age >= 0)
        & (track.age <= max_age + 1e-6)
    )


def live_pass_actor_observations(
    privileged: torch.Tensor,
    ball_tracks: list[VisualBallTracks] | tuple[VisualBallTracks, VisualBallTracks],
    partner_tracks: list[VisualBallTracks] | tuple[VisualBallTracks, VisualBallTracks],
    root_positions_w: torch.Tensor,
    root_quats_xyzw: torch.Tensor,
    root_velocities_w: torch.Tensor,
    own_head_positions_w: torch.Tensor,
    own_head_velocities_w: torch.Tensor,
    *,
    env_origins: torch.Tensor,
    ball_radius: float = 0.025,
    max_ball_age: float = 0.15,
    max_partner_age: float = 0.75,
) -> torch.Tensor:
    """Build 178 actor inputs from RGB tracks plus each player's own odometry.

    The input has two legacy 71-value player blocks. The existing visual helper
    replaces privileged ball channels and adds ball visibility, age, and search
    history. This helper additionally replaces target channels 67:70 using only
    the observing camera's partner estimate. It never indexes the other robot's
    kinematics and never restores target/ball truth when measurements are absent.

    Each 89-value output block retains the 76 visual features and appends partner
    relative velocity (3), valid (1), normalized age (1), desired outgoing ball
    velocity relative to the observing root (3), necessary delta-v (3), normalized
    time to own contact (1), and plan valid (1). Vectors use the observing root's
    frame. Delta-v is a vector difference and therefore has no translational
    velocity subtraction. Unknown targets, partner velocity and plans are zero;
    an unknown/expired partner's normalized age is one. Existing ball search
    history remains available even after its current position estimate expires.

    The partner track describes its RGB forehead point. A world-up ball radius
    makes this a ball-centre reception point. The same offset is applied to the
    own forehead kinematics before predicting the incoming descending contact.
    All planning uses the anticipated contact state, rather than treating a ball
    still in flight as if it had already been struck. ``env_origins`` supplies
    only the known court centre plane, not another robot's position.
    """
    positive = (ball_radius, max_ball_age, max_partner_age)
    if not all(math.isfinite(value) and value > 0 for value in positive):
        raise ValueError("Ball radius and both visual lifetimes must be finite and positive.")
    visual = visual_actor_observations(
        privileged, ball_tracks, root_positions_w, root_quats_xyzw, root_velocities_w, max_age=max_ball_age
    )
    count = privileged.shape[0]
    if len(partner_tracks) != 2:
        raise ValueError("Live passes require one partner estimate from each player's camera.")
    if own_head_positions_w.shape != (count, 2, 3) or own_head_velocities_w.shape != (count, 2, 3):
        raise ValueError("Own forehead positions/velocities must have shape (batch, 2, 3).")
    if env_origins.shape != (count, 3):
        raise ValueError("Court origins must have shape (batch, 3).")
    blocks = visual.view(count, 2, 76)
    actor = []
    contact_horizon = 0.6
    up = root_positions_w.new_tensor((0.0, 0.0, ball_radius))
    for player, (ball, partner) in enumerate(zip(ball_tracks, partner_tracks)):
        position = root_positions_w[:, player]
        velocity = root_velocities_w[:, player]
        raw_quat = root_quats_xyzw[:, player]
        root_valid = torch.isfinite(position).all(-1) & torch.isfinite(velocity).all(-1)
        root_valid &= torch.isfinite(raw_quat).all(-1) & (raw_quat.norm(dim=-1) > 1e-6)
        quat = F.normalize(torch.nan_to_num(raw_quat), dim=-1)
        partner_valid = _recent_track(partner, max_partner_age) & root_valid
        partner_point = partner.position_w + up
        target = _inverse_rotate(partner_point - position, quat)
        partner_velocity = _inverse_rotate(partner.velocity_w - velocity, quat)
        target = torch.where(partner_valid[:, None], target, 0.0)
        partner_velocity = torch.where(partner_valid[:, None], partner_velocity, 0.0)
        partner_age = torch.where(partner_valid, partner.age / max_partner_age, 1.0).clamp(0, 1)
        block = blocks[:, player].clone()
        block[:, 67:70] = target

        contact = predict_descending_contact(
            ball.position_w,
            ball.velocity_w,
            own_head_positions_w[:, player] + up,
            own_head_velocities_w[:, player],
            max_time=contact_horizon,
        )
        plan = plan_ballistic_pass(
            contact.position,
            contact.velocity,
            partner_point + partner.velocity_w * contact.time[:, None],
            partner.velocity_w,
            center_x=env_origins[:, 0],
        )
        plan_valid = partner_valid & _recent_track(ball, max_ball_age) & root_valid
        plan_valid &= contact.valid & plan.valid & (block[:, 70] > 0.5)
        desired = _inverse_rotate(plan.outgoing_velocity - velocity, quat)
        delta_velocity = _inverse_rotate(plan.delta_velocity, quat)
        desired = torch.where(plan_valid[:, None], desired, 0.0)
        delta_velocity = torch.where(plan_valid[:, None], delta_velocity, 0.0)
        time_to_contact = torch.where(plan_valid, contact.time / contact_horizon, 0.0).clamp(0, 1)
        actor.append(
            torch.cat(
                (
                    block,
                    partner_velocity,
                    partner_valid.float()[:, None],
                    partner_age[:, None],
                    desired,
                    delta_velocity,
                    time_to_contact[:, None],
                    plan_valid.float()[:, None],
                ),
                dim=-1,
            )
        )
    return torch.cat(actor, dim=-1)
