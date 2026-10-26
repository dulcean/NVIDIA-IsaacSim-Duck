# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode completion requires recovery after a physical volley objective.

No simulator is imported. A goal starts a fixed recovery window; it does not
immediately finish the episode or award success. A missed objective gets the
same observation window but can never become a successful episode. Physical
safety and time limits always override success.
"""

import math

import torch


class VolleyEpisode:
    """Vectorized rally/goal-recovery/miss-recovery state with terminal priority.

    Phase 0 is active play, phase 1 recovery after reaching the objective, and
    phase 2 recovery after a miss. The transition itself starts at elapsed time
    zero; the subsequent full window must elapse before recovery is assessed.
    The caller truncates timeouts separately from the returned termination
    mask. Neither repeated events nor a timeout can restart an ended world.
    """

    def __init__(self, num_envs: int, device: str | torch.device):
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_time = torch.zeros(num_envs, device=device)
        self.stable_time = torch.zeros(num_envs, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.missed = torch.zeros_like(self.success)
        self.unsafe = torch.zeros_like(self.success)
        self._finished = torch.zeros_like(self.success)
        self._done = torch.zeros_like(self.success)

    def step(
        self,
        goal: torch.Tensor,
        miss: torch.Tensor,
        unsafe: torch.Tensor,
        timeout: torch.Tensor,
        stable: torch.Tensor,
        dt: float,
        duration: float = 0.6,
        hold: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return termination, success event, goal-start event and miss-start event.

        Only an uninterrupted stable interval ending at the recovery deadline
        can validate a goal. Stability before the goal, or earlier in the
        recovery window, cannot unlock success. Miss takes priority over goal
        on the same step; safety and timeout take priority over both.
        """
        if not all(math.isfinite(value) for value in (dt, duration, hold)):
            raise ValueError("Episode timing values must be finite.")
        if dt <= 0 or duration <= 0 or hold < 0 or hold > duration:
            raise ValueError("Require dt > 0, duration > 0, and 0 <= hold <= duration.")

        self.unsafe |= unsafe
        eligible = ~self._finished & ~self.unsafe & ~timeout
        playing = eligible & (self.phase == 0)
        started_miss = playing & miss
        started_goal = playing & goal & ~miss
        recovering = eligible & (self.phase != 0)

        # A newly completed goal receives a full recovery window from this
        # boundary, rather than counting the control interval before contact.
        self.phase_time += recovering * dt
        self.stable_time = torch.where(
            recovering, torch.where(stable, self.stable_time + dt, 0.0), self.stable_time
        )
        self.phase[started_goal] = 1
        self.phase[started_miss] = 2
        self.missed |= started_miss

        # A small tolerance removes float32 accumulation errors at exact
        # multiples of the control period without shortening a physical step.
        expired = eligible & (self.phase != 0) & (self.phase_time + 1e-6 >= duration)
        self.success = (
            expired & (self.phase == 1) & stable & (self.stable_time + 1e-6 >= hold) & ~self.missed
        )
        self._done |= self.unsafe | expired
        self._finished |= self._done | timeout
        return self._done.clone(), self.success, started_goal, started_miss

    def reset(self, env_ids: torch.Tensor):
        """Clear only the selected worlds, retaining other worlds' phase and clocks."""
        self.phase[env_ids] = 0
        self.phase_time[env_ids] = 0.0
        self.stable_time[env_ids] = 0.0
        self.success[env_ids] = False
        self.missed[env_ids] = False
        self.unsafe[env_ids] = False
        self._finished[env_ids] = False
        self._done[env_ids] = False
