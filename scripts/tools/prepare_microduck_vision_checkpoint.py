# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare a MicroDuck actor for RGB-derived ball estimates and visual history.

The vision actor receives two 76-value blocks; its independent privileged
critic still receives the two original 71-value blocks. This is a policy warm
start, never a continuation of the source objective or optimizer. The original
actor's deterministic actions are preserved when its six ball features match
the estimates, regardless of newly appended visual inputs. Old visual actors
also preserve their trained visibility/age features when adding bearing memory.
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path

import torch

if __package__:
    from .prepare_microduck_curriculum_checkpoint import (
        CURRICULUM_PROFILE,
        _validate_actor,
        prepare_curriculum_checkpoint,
    )
else:
    from prepare_microduck_curriculum_checkpoint import (
        CURRICULUM_PROFILE,
        _validate_actor,
        prepare_curriculum_checkpoint,
    )


VISION_PROFILE = "microduck_volley_vision_v2"
LEGACY_VISION_PROFILE = "microduck_volley_vision_v1"
ACTOR_WIDTH = 152
PER_PLAYER_WIDTH = 76
CRITIC_WIDTH = 142
LEGACY_COLUMNS = tuple(range(71)) + tuple(range(76, 147))
LEGACY_VISION_COLUMNS = tuple(range(73)) + tuple(range(76, 149))
VISION_STATE_COLUMNS = tuple(range(71, 76)) + tuple(range(147, 152))
LAST_SEEN_BEARING_COLUMNS = (73, 74, 75, 149, 150, 151)


def _validate_legacy_vision_actor(actor: dict) -> int:
    """Validate the complete old visual architecture before preserving its learned features."""
    if not isinstance(actor, dict):
        raise ValueError("Expected an RSL-RL actor_state_dict with an MLP actor.")
    widths = [146, 1024, 512, 256, 28]
    expected = {
        "obs_normalizer._mean": (1, 146),
        "obs_normalizer._var": (1, 146),
        "obs_normalizer._std": (1, 146),
        "obs_normalizer.count": (),
        "distribution.std_param": (28,),
    }
    for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        expected[f"mlp.{2 * index}.weight"] = (outputs, inputs)
        expected[f"mlp.{2 * index}.bias"] = (outputs,)
    if set(actor) != set(expected):
        raise ValueError(f"Unsupported actor state keys: {set(actor) ^ set(expected)}")
    for name, shape in expected.items():
        value = actor[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid actor tensor {name}: expected finite values with shape {shape}.")
    if any((actor[f"obs_normalizer.{name}"] < 0).any() for name in ("_std", "_var", "count")):
        raise ValueError("Normalizer variance, standard deviation and count must be nonnegative.")
    return 2


def prepare_vision_checkpoint(source: dict, initial_std: float = 0.15) -> dict:
    """Add visual state/history without changing existing state-to-action weights.

    Accepted sources are tagged 142/28 curriculum actors, or legacy 61/14,
    71/14 and 142/28 actors first migrated through the curriculum adapter.
    Tagged 146/28 vision-v1 actors retain every learned visual/state feature.
    Curriculum state channels are preserved as actual features; they must not
    be folded as if they were the legacy constant-zero command placeholders.
    A fresh critic, fresh optimizer and iteration zero avoid carrying the
    source reward/value objective into the visual task.
    """
    if not isinstance(source, dict):
        raise ValueError("Expected an RSL-RL checkpoint dictionary.")
    if not math.isfinite(initial_std) or initial_std <= 0:
        raise ValueError("initial_std must be finite and positive.")
    infos = source.get("infos") or {}
    metadata = infos.get("microduck_curriculum", {}) if isinstance(infos, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError("Invalid source checkpoint profile metadata.")
    profile = metadata.get("profile")
    if profile == VISION_PROFILE:
        raise ValueError("This is already a vision checkpoint. Resume it directly instead of converting again.")
    if profile not in (None, CURRICULUM_PROFILE, LEGACY_VISION_PROFILE):
        raise ValueError(f"Unsupported source checkpoint profile: {profile!r}.")

    original = source.get("actor_state_dict")
    players = _validate_legacy_vision_actor(original) if profile == LEGACY_VISION_PROFILE else _validate_actor(original)
    original_width = original["mlp.0.weight"].shape[1]
    if profile == LEGACY_VISION_PROFILE:
        migrated = original
        mapping = LEGACY_VISION_COLUMNS
    elif profile == CURRICULUM_PROFILE:
        if players != 2:
            raise ValueError("A curriculum source must contain the complete 142-input, 28-action actor.")
        migrated = original
        mapping = LEGACY_COLUMNS
    else:
        migrated = prepare_curriculum_checkpoint(source, initial_std=initial_std)["actor_state_dict"]
        mapping = LEGACY_COLUMNS

    actor = OrderedDict((name, value.detach().clone()) for name, value in migrated.items())
    columns = torch.tensor(mapping, device=actor["mlp.0.weight"].device)
    for name, identity in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        value = actor[f"obs_normalizer.{name}"]
        expanded = value.new_full((1, ACTOR_WIDTH), identity)
        expanded[:, columns] = value
        actor[f"obs_normalizer.{name}"] = expanded
    weight = actor["mlp.0.weight"]
    expanded = weight.new_zeros((weight.shape[0], ACTOR_WIDTH))
    expanded[:, columns] = weight
    actor["mlp.0.weight"] = expanded
    actor["distribution.std_param"].fill_(initial_std)

    critic = OrderedDict()
    for name, identity in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        critic[f"obs_normalizer.{name}"] = weight.new_full((1, CRITIC_WIDTH), identity)
    critic["obs_normalizer.count"] = torch.zeros_like(actor["obs_normalizer.count"])
    widths = [CRITIC_WIDTH, 512, 256, 128, 1]
    for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        layer = torch.nn.Linear(inputs, outputs).to(weight)
        critic[f"mlp.{2 * index}.weight"] = layer.weight.detach()
        critic[f"mlp.{2 * index}.bias"] = layer.bias.detach()
    saved_state = metadata.get("curriculum_state") or {}
    source_stage = saved_state.get("stage") if isinstance(saved_state, dict) else None
    return {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "iter": 0,
        "infos": {
            "microduck_curriculum": {
                "profile": VISION_PROFILE,
                "source_profile": profile or "legacy",
                "source_iteration": source.get("iter"),
                "source_actor_inputs": original_width,
                "source_players": players,
                "source_curriculum_stage": source_stage,
                "actor_observation_width": ACTOR_WIDTH,
                "critic_observation_width": CRITIC_WIDTH,
                "per_player_actor_width": PER_PLAYER_WIDTH,
                "visual_ball_channels_per_player": [61, 67],
                "appended_channels_per_player": [
                    "visible",
                    "age",
                    "last_seen_bearing_x",
                    "last_seen_bearing_y",
                    "last_seen_bearing_z",
                ],
                "initial_std": initial_std,
            }
        },
    }


def main():
    """Write separate visual-task weights without replacing any existing checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-std", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    prepared = prepare_vision_checkpoint(source, args.initial_std)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(prepared, stream)
    print(
        f"Prepared vision actor: {ACTOR_WIDTH} inputs / 28 actions; "
        f"privileged critic: {CRITIC_WIDTH} inputs: {args.output}"
    )
    print("Fresh visual-task training requires --no-resume_optimizer_state.")


if __name__ == "__main__":
    main()
