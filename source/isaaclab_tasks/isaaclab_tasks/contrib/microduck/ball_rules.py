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
