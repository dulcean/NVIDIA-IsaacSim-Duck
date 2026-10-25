# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact rules for short-range MicroDuck rallies."""

import torch


def classify_head_contact(
    onset: torch.Tensor,
    front_contact: torch.Tensor,
    expected_player: torch.Tensor,
    last_hitter: torch.Tensor,
    invalid: torch.Tensor,
    player: int,
    num_players: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify new touches without imposing a minimum flight time between different players.

    Contact onsets and turn order reject persistent touches and double hits. A legal
    reply from the other robot may arrive sooner than the aiming horizon.
    """
    candidate = onset & ~invalid
    right_turn = expected_player == player
    if num_players == 1:
        right_turn &= last_hitter < 0
    wrong_surface = candidate & ~front_contact
    wrong_turn = candidate & front_contact & ~right_turn
    valid = candidate & front_contact & right_turn
    return valid, wrong_surface, wrong_turn


def robot_collision_mask(forces: torch.Tensor, threshold: float) -> torch.Tensor:
    """Detect any robot-to-robot contact, without cancelling opposing contact forces."""
    return ((forces.norm(dim=-1) > threshold) | ~torch.isfinite(forces).all(dim=-1)).flatten(1).any(dim=-1)


def balance_errors(gravity_b: torch.Tensor, angular_velocity_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure the least balanced player's torso, allowing yaw rotation and airborne motion.

    Inputs have shape (environments, players, 3). Upright projected gravity is
    (0, 0, -1). Using the worst player prevents one upright robot from hiding
    its partner's collapse. Head orientation and linear velocity are unrestricted.
    """
    tilt = (1.0 + gravity_b[..., 2]).clamp_min(0).amax(dim=1)
    tumbling = angular_velocity_b[..., :2].square().sum(-1).amax(dim=1)
    return tilt, tumbling


class RecoveryRewards:
    """Bank strike credit per player until sustained recovery; terminal steps cancel unpaid credit."""

    def __init__(self, num_envs: int, num_players: int, device: str):
        self.pending = torch.zeros(num_envs, num_players, device=device)
        self.hits = torch.zeros_like(self.pending)
        self.age = torch.zeros_like(self.pending)
        self.stable_time = torch.zeros_like(self.pending)

    def reset(self, env_ids):
        """Clear only the reset environments, including unpaid rewards."""
        for state in (self.pending, self.hits, self.age, self.stable_time):
            state[env_ids] = 0

    def step(
        self,
        credit: torch.Tensor,
        hits: torch.Tensor,
        stable: torch.Tensor,
        ended: torch.Tensor,
        dt: float,
        delay: float,
        hold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate strikes and pay once after recovering from the most recent own strike.

        A new strike restarts the recovery clock but retains unpaid credit. Other
        players' strikes do not restart it. Failure and timeout take priority over
        payout, even when recovery would otherwise complete on that same step.
        """
        new_hit = hits > 0
        self.pending += credit
        self.hits += hits
        active = self.hits > 0
        self.age = torch.where(new_hit | ~active, 0.0, self.age + dt)
        self.stable_time = torch.where(stable & active & ~new_hit, self.stable_time + dt, 0.0)
        ready = active & (self.age + 1e-6 >= delay) & (self.stable_time + 1e-6 >= hold)
        ready &= ~ended.unsqueeze(-1)
        paid = torch.where(ready, self.pending, 0.0)
        recovered = torch.where(ready, self.hits, 0.0)
        clear = ready | ended.unsqueeze(-1)
        for state in (self.pending, self.hits, self.age, self.stable_time):
            state.masked_fill_(clear, 0.0)
        return paid, recovered
