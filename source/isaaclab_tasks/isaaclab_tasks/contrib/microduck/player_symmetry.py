# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exchange complete player-local observations and servo actions for visual PPO.

This is a rotation of the court by pi and a relabeling of its identical robots,
not a left/right reflection of a robot. Joint signs, IMU vectors, visual bearings
and target vectors already use each robot's own frame, so no sign flip is needed.
Only estimated ball features already supplied to the actor are exchanged. No
image or simulator ball truth is introduced into the actor's observations.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


def exchange_players(env, obs=None, actions=None):
    """Append exchanged samples for RSL-RL's augmentation and consistency loss.

    Preserve the original batch first, as required by the actual RSL-RL
    ``Symmetry`` extension. The visual actor has two 76-value local blocks,
    the critic has two 71-value blocks, and actions have two 14-servo blocks.
    The same rotation exchanges the first receiver and the partner implicitly
    through their local turn/phase/state features. The environment is unused.
    """
    augmented_obs = None
    augmented_actions = None
    if obs is not None:
        if len(obs.batch_size) != 1:
            raise ValueError("Player exchange requires a flat PPO observation batch.")
        widths = {"policy": 76, "critic": 71}
        if not obs.keys() or any(name not in widths for name in obs.keys()):
            raise ValueError("Player exchange supports only visual policy and privileged critic groups.")
        exchanged = {}
        for name, value in obs.items():
            half = widths[name]
            if value.ndim != 2 or value.shape[-1] != 2 * half:
                raise ValueError(f"Player exchange requires {2 * half} inputs for {name}.")
            exchanged[name] = torch.cat((value[:, half:], value[:, :half]), dim=-1)
        mirrored = TensorDict(exchanged, batch_size=obs.batch_size)
        augmented_obs = torch.cat((obs, mirrored), dim=0)
    if actions is not None:
        if actions.ndim != 2 or actions.shape[-1] != 28:
            raise ValueError("Player exchange requires two 14-servo action blocks.")
        if obs is not None and actions.shape[0] != obs.batch_size[0]:
            raise ValueError("Player observations and actions must have the same batch size.")
        mirrored = torch.cat((actions[:, 14:], actions[:, :14]), dim=-1)
        augmented_actions = torch.cat((actions, mirrored), dim=0)
    return augmented_obs, augmented_actions
