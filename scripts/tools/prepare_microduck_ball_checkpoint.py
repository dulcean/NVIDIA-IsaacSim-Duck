# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Initialize MicroDuck ball policies from a walking or heading RSL-RL checkpoint.

No simulator is imported. Run on the training machine with ``uv run python``.
Always train the result with ``--no-resume_optimizer_state``.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def prepare_checkpoint(source: dict, players: int, initial_std: float = 0.15) -> dict:
    """Preserve the source actor's deterministic outputs for each robot and reset training state.

    Accepted sources are the 61/14 walking policy and the 71/14 heading policy.
    The two-player actor uses two block-diagonal copies; all weights remain trainable.
    A new, independent critic receives the complete 71- or 142-value observation.
    """
    if players not in (1, 2) or initial_std <= 0:
        raise ValueError("players must be 1 or 2 and initial_std must be positive.")
    old = source["actor_state_dict"]
    old_input = old["mlp.0.weight"].shape[1]
    expected = [(512, old_input), (256, 512), (128, 256), (14, 128)]
    if old_input not in (61, 71) or any(
        tuple(old[f"mlp.{2 * i}.weight"].shape) != shape for i, shape in enumerate(expected)
    ):
        raise ValueError("Expected a single MicroDuck actor with 61 or 71 inputs and 512/256/128/14 ELU layers.")
    known = {
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "obs_normalizer.count",
        "distribution.std_param",
        *(f"mlp.{2 * i}.{name}" for i in range(4) for name in ("weight", "bias")),
    }
    if set(old) != known:
        raise ValueError(f"Unsupported actor state keys: {set(old) ^ known}")
    actor = OrderedDict()
    for key, identity in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        value = old[f"obs_normalizer.{key}"]
        blocks = []
        for _ in range(players):
            block = value.new_full((1, 71), identity)
            block[:, :old_input] = value
            blocks.append(block)
        actor[f"obs_normalizer.{key}"] = torch.cat(blocks, dim=1)
    # Retain source normalization, with identity statistics for the additional physical quantities.
    actor["obs_normalizer.count"] = old["obs_normalizer.count"].clone()
    actor["distribution.std_param"] = old["distribution.std_param"].new_full((14 * players,), initial_std)
    for i, (rows, cols) in enumerate(expected):
        in_width = 71 if i == 0 else cols
        weights = old[f"mlp.{2 * i}.weight"].new_zeros((rows * players, in_width * players))
        for player in range(players):
            weights[player * rows : (player + 1) * rows, player * in_width : player * in_width + cols] = old[
                f"mlp.{2 * i}.weight"
            ]
        actor[f"mlp.{2 * i}.weight"] = weights
        actor[f"mlp.{2 * i}.bias"] = old[f"mlp.{2 * i}.bias"].repeat(players)

    critic = OrderedDict()
    for key, identity in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        critic[f"obs_normalizer.{key}"] = old[f"obs_normalizer.{key}"].new_full((1, 71 * players), identity)
    critic["obs_normalizer.count"] = torch.zeros_like(old["obs_normalizer.count"])
    widths = [71 * players, 512, 256, 128, 1]
    for i, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        layer = torch.nn.Linear(inputs, outputs)
        critic[f"mlp.{2 * i}.weight"] = layer.weight.detach()
        critic[f"mlp.{2 * i}.bias"] = layer.bias.detach()
    return {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "iter": 0,
        "infos": {"microduck_transfer": {"source_iteration": source.get("iter"), "players": players}},
    }


def main():
    """Write a new checkpoint without overwriting the source or an existing output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--players", type=int, choices=(1, 2), required=True)
    parser.add_argument("--initial_std", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    result = prepare_checkpoint(source, args.players, args.initial_std)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(result, stream)
    print(f"Prepared {71 * args.players} inputs / {14 * args.players} actions: {args.output}")
    print("Use --no-resume_optimizer_state when starting the new task.")


if __name__ == "__main__":
    main()
