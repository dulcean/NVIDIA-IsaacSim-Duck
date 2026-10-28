# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live visual partners: graduation requires actual legal head-to-head transfers."""

from dataclasses import replace

from .vision_curriculum import VISION_STAGES as COACH_STAGES
from .vision_curriculum import VisionCurriculumState
from .vision_curriculum import VisionEvaluationReport as VisionEvaluationReport

LIVE_PASS_PROFILE = "microduck_volley_live_pass_v1"
VISION_STAGES = (
    *COACH_STAGES[:2],
    replace(COACH_STAGES[2], name="first_real_pass", paired=True, required_returns=1, objective="rally"),
    *COACH_STAGES[3:],
)


class LivePassCurriculumState(VisionCurriculumState):
    """Keep every physical, visual and recovery gate; a predicted arc never graduates a lesson."""

    stages = VISION_STAGES
