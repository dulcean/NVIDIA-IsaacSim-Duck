# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warm-start the volley curriculum while preserving legacy deterministic actions.

Accepts standing (61/14), heading (71/14), and centralized volley (142/28)
RSL-RL checkpoints. The former zero command slots become physical/task state.
Their old normalized constant is folded into the first bias before replacing
them. The critic, optimizer and iteration start afresh for the new objective.

Run with ``uv run python scripts/tools/prepare_microduck_curriculum_checkpoint.py
SOURCE OUTPUT``. Train the result with ``--no-resume_optimizer_state``.
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path

import torch

if __package__:
    from .prepare_microduck_ball_checkpoint import prepare_checkpoint
else:
    from prepare_microduck_ball_checkpoint import prepare_checkpoint


CURRICULUM_PROFILE = "microduck_volley_curriculum_v1"
NORMALIZATION_EPS = 0.01  # EmpiricalNormalization default in pinned RSL-RL 5.5.1.
STATE_CHANNELS = tuple(range(48, 61)) + tuple(range(119, 132))


def _validate_actor(actor: dict) -> int:
    """Reject incompatible architectures and malformed/nonfinite state before conversion."""
    if not isinstance(actor, dict) or "mlp.0.weight" not in actor:
        raise ValueError("Expected an RSL-RL actor_state_dict with an MLP actor.")
    first = actor["mlp.0.weight"]
    if not isinstance(first, torch.Tensor) or first.ndim != 2:
        raise ValueError("The actor's first weight must be a two-dimensional tensor.")
    width = first.shape[1]
    players = 1 if width in (61, 71) else 2 if width == 142 else 0
    if players == 0:
        raise ValueError("Expected a MicroDuck actor with 61, 71, or 142 inputs.")
    widths = [width, 512 * players, 256 * players, 128 * players, 14 * players]
    expected = {
        "obs_normalizer._mean": (1, width),
        "obs_normalizer._var": (1, width),
        "obs_normalizer._std": (1, width),
        "obs_normalizer.count": (),
        "distribution.std_param": (14 * players,),
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
    return players


def prepare_curriculum_checkpoint(source: dict, initial_std: float = 0.15) -> dict:
    """Replace previously constant inputs without changing the actor's initial policy.

    The 13 slots at 48:61 in each 71-value block used to be exactly zero.
    Source normalization can turn zero into a nonzero constant, so simply
    zeroing their weights would change actions. Folding that contribution into
    the first bias preserves all legacy outputs, including cross-robot weights.
    New input connections remain trainable and start at zero.
    """
    if not math.isfinite(initial_std) or initial_std <= 0:
        raise ValueError("initial_std must be finite and positive.")
    if not isinstance(source, dict):
        raise ValueError("Expected an RSL-RL checkpoint dictionary.")
    infos = source.get("infos") or {}
    if isinstance(infos, dict) and infos.get("microduck_curriculum", {}).get("profile") == CURRICULUM_PROFILE:
        raise ValueError("This is already a curriculum checkpoint. Resume it directly instead of converting again.")
    original = source.get("actor_state_dict")
    players = _validate_actor(original)
    if players == 1:
        result = prepare_checkpoint(source, players=2, initial_std=initial_std)
        actor = result["actor_state_dict"]
        critic = result["critic_state_dict"]
    else:
        actor = OrderedDict((name, value.detach().clone()) for name, value in original.items())
        critic = OrderedDict()
        for name, identity in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            critic[f"obs_normalizer.{name}"] = original[f"obs_normalizer.{name}"].new_full((1, 142), identity)
        critic["obs_normalizer.count"] = torch.zeros_like(original["obs_normalizer.count"])
        widths = [142, 512, 256, 128, 1]
        for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
            layer = torch.nn.Linear(inputs, outputs).to(original["mlp.0.weight"])
            critic[f"mlp.{2 * index}.weight"] = layer.weight.detach()
            critic[f"mlp.{2 * index}.bias"] = layer.bias.detach()

    channels = torch.tensor(STATE_CHANNELS, device=actor["mlp.0.weight"].device)
    normalized_zero = (
        -actor["obs_normalizer._mean"][:, channels]
        / (actor["obs_normalizer._std"][:, channels] + NORMALIZATION_EPS)
    ).flatten()
    actor["mlp.0.bias"] += actor["mlp.0.weight"][:, channels] @ normalized_zero
    actor["mlp.0.weight"][:, channels] = 0.0
    for name, identity in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        actor[f"obs_normalizer.{name}"][:, channels] = identity
    actor["distribution.std_param"].fill_(initial_std)
    return {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "iter": 0,
        "infos": {
            "microduck_curriculum": {
                "profile": CURRICULUM_PROFILE,
                "source_iteration": source.get("iter"),
                "source_players": players,
                "normalization_eps": NORMALIZATION_EPS,
                "initial_std": initial_std,
            }
        },
    }


def main():
    """Write a separate warm-start file; never overwrite source or existing weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-std", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    result = prepare_curriculum_checkpoint(source, args.initial_std)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(result, stream)
    print(f"Prepared curriculum actor: 142 inputs / 28 actions: {args.output}")
    print("Fresh critic and optimizer required; use --no-resume_optimizer_state.")


if __name__ == "__main__":
    main()
