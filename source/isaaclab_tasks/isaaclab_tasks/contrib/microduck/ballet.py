# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured one-foot turns followed by a grounded bow; no simulator imports."""

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch

BALLET_PROFILE = "microduck_ballet_v1"
TURN_ANGLES = (0.0, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi)
PHASE_NAMES = ("lift", "turn", "land", "bow", "recover", "complete")
OBSERVATION_WIDTH = 89
REWARD_DISCOUNT = 0.997


def rotation_potential(previous, current, terminal):
    """Discount-consistent shaping: partial turns cannot profit from reset/fall cycles."""
    return REWARD_DISCOUNT * torch.where(terminal, 0.0, current) - previous


def walking_actor_warm_start(source, initial_std=0.15):
    """Keep the real 61-input walking actor and add initially inactive ballet features.

    Only the actor transfers. A new task needs its own critic, optimizer and
    physical curriculum. All original and appended connections remain trainable.
    """
    expected = ((512, 61), (256, 512), (128, 256), (14, 128))
    if not math.isfinite(initial_std) or initial_std <= 0:
        raise ValueError("Initial action standard deviation must be finite and positive.")
    if any(
        f"mlp.{2 * i}.weight" not in source or tuple(source[f"mlp.{2 * i}.weight"].shape) != shape
        for i, shape in enumerate(expected)
    ):
        raise ValueError("Use a single-robot 61-input MicroDuck walking checkpoint with 512/256/128 hidden layers.")
    actor = OrderedDict((key, value.clone()) for key, value in source.items())
    if any(not bool(torch.isfinite(value).all()) for value in actor.values()):
        raise ValueError("Walking checkpoint contains nonfinite actor parameters.")
    for key, identity in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        value = source[f"obs_normalizer.{key}"]
        if value.shape != (1, 61):
            raise ValueError("Walking observation normalization must have 61 channels.")
        result = value.new_full((1, OBSERVATION_WIDTH), identity)
        result[:, :61] = value
        actor[f"obs_normalizer.{key}"] = result
    weights = source["mlp.0.weight"]
    expanded = weights.new_zeros((512, OBSERVATION_WIDTH))
    expanded[:, :61] = weights
    actor["mlp.0.weight"] = expanded
    actor["distribution.std_param"] = source["distribution.std_param"].new_full((14,), initial_std)
    # Keep the measured mean/variance as a prior, without letting hundreds of
    # millions of standing samples prevent new ballet channels from adapting.
    actor["obs_normalizer.count"] = source["obs_normalizer.count"].clamp_max(4096).clone()
    return actor


@dataclass
class BalletFrame:
    """Physical evidence, feet ordered support then free, with floor-only forces in world coordinates."""

    yaw: torch.Tensor
    pitch: torch.Tensor
    head_z: torch.Tensor
    up: torch.Tensor
    height: torch.Tensor
    planar_speed: torch.Tensor
    tumbling: torch.Tensor
    yaw_rate: torch.Tensor
    clearance: torch.Tensor
    forces: torch.Tensor
    nonfoot_contact: torch.Tensor
    displacement: torch.Tensor
    anchor_drift: torch.Tensor
    com_error: torch.Tensor
    finite: torch.Tensor

    def support(self):
        """Require upward ground force and a genuinely clear free foot, not an airborne spin."""
        contact = self.forces.norm(dim=-1) > 0.04
        grounded = self.forces[..., 2] > 0.05
        single = grounded[:, 0] & ~contact[:, 1] & (self.clearance[:, 1] >= 0.018)
        single &= (self.up >= 0.90) & (self.height >= 0.09)
        single &= (self.planar_speed <= 0.30) & (self.tumbling <= 1.5)
        single &= (self.anchor_drift <= 0.06) & (self.com_error <= 0.045) & ~self.nonfoot_contact
        both = grounded.all(-1) & ~self.nonfoot_contact
        return single, both


class BalletRoutine:
    """Count contiguous, signed physical yaw at physics cadence before allowing a bow.

    A free-foot touch, flight, or excessive tilt resets the current turn and
    pays back its progress shaping. Alternating partial turns cannot accumulate
    into a full revolution. Net root yaw, including +/-pi crossings, is used;
    rotating just the head or oscillating the torso cannot pass.
    """

    def __init__(self, count, device, *, direction=1.0):
        if count < 1 or direction not in (-1.0, 1.0):
            raise ValueError("Ballet needs a positive batch and a signed unit turn direction.")
        self.direction = direction
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        self.goal = torch.zeros(count, device=device)
        self.age = torch.zeros_like(self.goal)
        self.elapsed = torch.zeros_like(self.goal)
        self.hold = torch.zeros_like(self.goal)
        self.progress = torch.zeros_like(self.goal)
        self.peak = torch.zeros_like(self.goal)
        self.last_yaw = torch.zeros_like(self.goal)
        self.bad_time = torch.zeros_like(self.goal)
        self.failed = torch.zeros(count, dtype=torch.bool, device=device)
        self.success = torch.zeros_like(self.failed)
        self.transitions = torch.zeros_like(self.goal)

    def reset(self, indices, yaw, goal):
        for value in (
            self.age,
            self.elapsed,
            self.hold,
            self.progress,
            self.peak,
            self.bad_time,
            self.failed,
            self.success,
            self.transitions,
        ):
            value[indices] = 0
        self.phase[indices] = 0
        self.last_yaw[indices] = yaw
        self.goal[indices] = goal

    def begin_control_step(self):
        self.transitions.zero_()

    def update(self, frame: BalletFrame, dt: float):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Ballet physics time must be finite and positive.")
        self.elapsed += dt
        self.age += dt
        # atan2(sin,cos) unwraps local increments without confusing pi with a full turn.
        delta = torch.atan2(torch.sin(frame.yaw - self.last_yaw), torch.cos(frame.yaw - self.last_yaw))
        self.last_yaw.copy_(frame.yaw)
        finite = frame.finite & torch.isfinite(frame.yaw) & torch.isfinite(delta)
        self.bad_time.copy_(torch.where(frame.nonfoot_contact, self.bad_time + dt, 0.0))
        self.failed |= ~finite | (frame.up < 0.5) | (frame.height < 0.065) | (frame.displacement > 0.22)
        self.failed |= (self.elapsed > 0.15) & (self.bad_time >= 0.04)
        single, both = frame.support()
        single &= finite & ~self.failed & (delta.abs() <= 6.0 * dt)
        turning = (self.phase == 1) & ~self.failed
        next_progress = torch.where(single, self.progress + self.direction * delta, 0.0)
        next_progress = torch.minimum(next_progress.clamp_min(-2 * math.pi), self.goal + 0.035)
        self.progress.copy_(torch.where(turning, next_progress, self.progress))
        self.peak.copy_(torch.maximum(self.peak, self.progress))
        quiet = (frame.planar_speed < 0.12) & (frame.yaw_rate.abs() < 0.35) & (frame.tumbling < 0.6)
        bowed = (
            both & quiet & (frame.up >= 0.88) & (frame.pitch >= 0.18) & (frame.pitch <= 0.45) & (frame.head_z <= -0.25)
        )
        recovered = both & quiet & (frame.up > math.cos(0.12)) & (frame.head_z.abs() < 0.18)
        eligible = torch.where(self.phase <= 1, single, torch.where(self.phase == 3, bowed, both & quiet))
        eligible = torch.where(self.phase == 4, recovered, eligible) & ~self.failed
        self.hold.copy_(torch.where(eligible, self.hold + dt, 0.0))
        durations = self.goal.new_tensor((0.35, 0.50, 0.20, 0.35, 0.50, float("inf")))[self.phase]
        advance = (self.hold + 1e-6 >= durations) & (self.phase < 5)
        advance &= (self.phase != 1) | (self.progress >= self.goal - 0.035)
        self.phase += advance.long()
        self.transitions += advance
        self.age.copy_(torch.where(advance, 0.0, self.age))
        self.hold.copy_(torch.where(advance, 0.0, self.hold))
        limits = self.goal.new_tensor((5.0, 12.0, 2.0, 3.0, 3.0, float("inf")))[self.phase]
        self.failed |= self.age > limits
        # Rotation shaping is a refundable potential until the entire bow/recovery
        # succeeds. A turn followed by a deliberate fall cannot bank spin credit.
        self.progress.masked_fill_(self.failed, 0.0)
        self.success.copy_((self.phase == 5) & ~self.failed & recovered)


class BalletCurriculum:
    """Advance each training world after two completed routines; persist levels with weights."""

    def __init__(self, count, device, start_stage=0):
        if start_stage not in range(len(TURN_ANGLES)):
            raise ValueError("Unknown ballet stage.")
        self.stage = torch.full((count,), start_stage, dtype=torch.long, device=device)
        self.streak = torch.zeros_like(self.stage)

    @property
    def angles(self):
        return self.stage.new_tensor(TURN_ANGLES, dtype=torch.float32)[self.stage]

    def record(self, indices, success):
        self.streak[indices] = torch.where(success, self.streak[indices] + 1, 0)
        advance = (self.streak[indices] >= 2) & (self.stage[indices] < len(TURN_ANGLES) - 1)
        self.stage[indices] += advance.long()
        self.streak[indices] = torch.where(advance, 0, self.streak[indices])

    def state_dict(self):
        return {"profile": BALLET_PROFILE, "angles": list(TURN_ANGLES), "stage": self.stage.cpu().tolist()}

    def load_state_dict(self, state):
        if state.get("profile") != BALLET_PROFILE or state.get("angles") != list(TURN_ANGLES):
            raise ValueError("Saved ballet curriculum has a different physical contract.")
        levels = state.get("stage")
        if not isinstance(levels, list) or not levels or any(type(x) is not int or x not in range(5) for x in levels):
            raise ValueError("Saved ballet levels must be nonempty integers from 0 to 4.")
        values = torch.tensor(levels, device=self.stage.device)
        self.stage.copy_(values[torch.arange(len(self.stage), device=self.stage.device) % len(values)])
        # Physical episodes restart on resume; an interrupted attempt cannot extend a pass streak.
        self.streak.zero_()
