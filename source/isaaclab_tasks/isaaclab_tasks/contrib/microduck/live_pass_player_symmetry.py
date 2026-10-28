# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exchange complete 89-value camera/planner blocks while retaining servo order."""

from .player_symmetry import _exchange_players


def exchange_players(env, obs=None, actions=None):
    """Use the same court rotation for 178-value actors and 142-value critics."""
    return _exchange_players(obs, actions, policy_width=89)
