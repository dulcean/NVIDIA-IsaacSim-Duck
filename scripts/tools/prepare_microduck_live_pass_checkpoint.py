# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep a trained RGB actor while adding observed partner and physical pass-plan inputs.

Each 76-input player block becomes an 89-input block. The thirteen added inputs
start with zero actor weights. Existing connections within each player are
retained, while connections across players are removed to prevent using the
other robot's privileged proprioception as a substitute for camera perception.
This changes a centralized source actor's actions. Source optimizer, critic,
curriculum gates and certification are never resumed.
"""

from __future__ import annotations

import argparse
import copy
from collections import OrderedDict
from pathlib import Path

import torch

LIVE_PASS_PROFILE = "microduck_volley_live_pass_v1"
SOURCE_PROFILE = "microduck_volley_vision_v2"
ACTOR_WIDTH = 178
PER_PLAYER_WIDTH = 89
CRITIC_WIDTH = 142
SOURCE_COLUMNS = tuple(range(76)) + tuple(range(89, 165))


def prepare_live_pass_checkpoint(source: dict) -> dict:
    """Retain trained local connections while isolating both players' actor computations."""
    if not isinstance(source, dict):
        raise ValueError("Expected an RSL-RL checkpoint dictionary.")
    infos = source.get("infos") or {}
    metadata = infos.get("microduck_curriculum", {}) if isinstance(infos, dict) else {}
    if not isinstance(metadata, dict) or metadata.get("profile") != SOURCE_PROFILE:
        raise ValueError(
            "A live-pass warm start requires a trained vision-v2 checkpoint; resume native live-pass weights."
        )
    if not isinstance(metadata.get("vision_contract"), dict) or not metadata["vision_contract"]:
        raise ValueError("The source visual checkpoint must carry its saved camera contract.")
    original = source.get("actor_state_dict")
    expected = {
        "obs_normalizer._mean": (1, 152),
        "obs_normalizer._var": (1, 152),
        "obs_normalizer._std": (1, 152),
        "obs_normalizer.count": (),
        "distribution.std_param": (28,),
    }
    widths = [152, 1024, 512, 256, 28]
    for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        expected[f"mlp.{2 * index}.weight"] = (outputs, inputs)
        expected[f"mlp.{2 * index}.bias"] = (outputs,)
    if not isinstance(original, dict) or set(original) != set(expected):
        raise ValueError("A live-pass warm start requires the complete 152-input, 28-action visual actor.")
    for name, shape in expected.items():
        value = original[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid visual actor tensor {name}; expected finite values with shape {shape}.")
    if any((original[f"obs_normalizer.{name}"] < 0).any() for name in ("_var", "_std", "count")):
        raise ValueError("Visual normalizer variance, standard deviation and count must be nonnegative.")

    actor = OrderedDict((name, value.detach().clone()) for name, value in original.items())
    weight = actor["mlp.0.weight"]
    columns = torch.tensor(SOURCE_COLUMNS, device=weight.device)
    expanded = weight.new_zeros((weight.shape[0], ACTOR_WIDTH))
    expanded[:, columns] = weight
    actor["mlp.0.weight"] = expanded
    for name, identity in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        value = actor[f"obs_normalizer.{name}"]
        expanded = value.new_full((1, ACTOR_WIDTH), identity)
        expanded[:, columns] = value
        actor[f"obs_normalizer.{name}"] = expanded
    for name, value in actor.items():
        if name.startswith("mlp.") and name.endswith(".weight"):
            outputs, inputs = value.shape
            value[: outputs // 2, inputs // 2 :] = 0
            value[outputs // 2 :, : inputs // 2] = 0

    critic = OrderedDict()
    for name, identity in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        critic[f"obs_normalizer.{name}"] = weight.new_full((1, CRITIC_WIDTH), identity)
    critic["obs_normalizer.count"] = torch.zeros_like(actor["obs_normalizer.count"])
    widths = [CRITIC_WIDTH, 512, 256, 128, 1]
    for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        layer = torch.nn.Linear(inputs, outputs).to(weight)
        critic[f"mlp.{2 * index}.weight"] = layer.weight.detach()
        critic[f"mlp.{2 * index}.bias"] = layer.bias.detach()
    provenance = {
        "profile": LIVE_PASS_PROFILE,
        "source_profile": SOURCE_PROFILE,
        "source_iteration": source.get("iter"),
        "source_actor_inputs": 152,
        "actor_observation_width": ACTOR_WIDTH,
        "critic_observation_width": CRITIC_WIDTH,
        "per_player_actor_width": PER_PLAYER_WIDTH,
        "source_vision_contract": copy.deepcopy(metadata["vision_contract"]),
        "expanded_visual_warm_start": True,
        "native_actor_warm_start": True,
        "actor_isolation": "per_player_local_inputs_all_layers",
        "source_policy_actions_preserved": False,
    }
    for name in ("task_settings", "player_symmetry"):
        if name in metadata:
            provenance[name] = copy.deepcopy(metadata[name])
    return {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "iter": 0,
        "infos": {"microduck_curriculum": provenance},
    }


def main():
    """Write separate warm-start weights and keep the original learned actor untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    prepared = prepare_live_pass_checkpoint(torch.load(args.source, weights_only=True, map_location="cpu"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(prepared, stream)
    print(f"Prepared live-pass actor: {ACTOR_WIDTH} inputs / 28 actions; fresh critic: {CRITIC_WIDTH} inputs.")


if __name__ == "__main__":
    main()
