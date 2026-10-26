# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate curriculum checkpoints on balanced, fixed cohorts of held-out episodes.

No training, reward averaging, or ballistic prediction can satisfy this gate.
The physical environment records complete target/return-and-recovery outcomes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

CURRICULUM_PROFILE = "microduck_volley_curriculum_v1"
VISION_PROFILE = "microduck_volley_vision_v1"


def _curriculum_module():
    """Load the pure curriculum contract without importing all simulator tasks."""
    name = "isaaclab_tasks.contrib.microduck.volley_curriculum"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/volley_curriculum.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _validate_record(record: dict, stage: int, spec):
    """Fail closed if rollout metadata cannot prove the advertised objective."""
    fields = ("env_id", "episode_id", "server", "returns", "stage")
    if any(type(record.get(name)) is not int or record[name] < 0 for name in fields):
        raise ValueError("Episode records need non-negative integer IDs, side, return count, and stage.")
    if record["server"] not in (0, 1) or record["stage"] != stage:
        raise ValueError("Episode record is from a different stage or an unknown receiving side.")
    if record["server"] != record["env_id"] % 2:
        raise ValueError("Held-out evaluation must assign the initial receiver by world-ID parity.")
    if any(type(record.get(name)) is not bool for name in ("success", "unsafe")):
        raise ValueError("Episode success and unsafe outcomes must be booleans.")
    if record["success"] and (record["unsafe"] or record["returns"] < spec.required_returns):
        raise ValueError("An unsafe or too-short rally cannot be a successful held-out episode.")
    if not isinstance(record.get("reason"), str):
        raise ValueError("Episode records must include a terminal reason.")
    if "goal_reached" in record and type(record["goal_reached"]) is not bool:
        raise ValueError("The optional physical goal outcome must be a boolean.")
    if "launch_quality" in record:
        quality = record["launch_quality"]
        if type(quality) not in (int, float) or not math.isfinite(quality) or quality < 0:
            raise ValueError("The optional launch-quality diagnostic must be finite and non-negative.")
    for name in ("head_hits", "head_contacts"):
        if name in record and (type(record[name]) is not int or record[name] < 0):
            raise ValueError("Contact diagnostics must be non-negative integer counts.")
    if "recovered_miss" in record:
        if type(record["recovered_miss"]) is not bool:
            raise ValueError("The recovered-miss diagnostic must be a boolean.")
        if record["recovered_miss"] and (record["success"] or record["unsafe"]):
            raise ValueError("A recovered miss cannot also be a success or an unsafe episode.")
    if "recovery_stable_by_player" in record:
        stable = record["recovery_stable_by_player"]
        if not isinstance(stable, (list, tuple)) or len(stable) != 2 or any(type(v) is not bool for v in stable):
            raise ValueError("Recovery diagnostics must include a boolean for each player.")
    if "recovery_hold_s" in record:
        hold = record["recovery_hold_s"]
        if type(hold) not in (int, float) or not math.isfinite(hold) or hold < 0:
            raise ValueError("The recovery hold must be finite and non-negative.")
    for name in ("unsafe_reasons", "miss_reasons"):
        if name in record:
            reasons = record[name]
            if (
                not isinstance(reasons, (list, tuple))
                or any(not isinstance(reason, str) or not reason for reason in reasons)
                or len(set(reasons)) != len(reasons)
            ):
                raise ValueError("Diagnostic reasons must be unique, non-empty strings.")


def evaluate_policy(
    env,
    policy,
    *,
    stage: int,
    evaluation_id: int,
    episodes_per_side: int = 64,
    max_steps: int = 20000,
):
    """Collect first-episode outcomes for fixed balanced cohorts, without learning.

    ``env`` is an existing RSL-RL wrapper in held-out evaluation mode. Its raw
    environment must publish ``episode_records`` exactly once after a physical
    episode reset. The caller owns evaluation mode/seed, curriculum selection,
    global RNG preservation, and resetting training episodes after this call.

    Each batch fixes its world IDs before any episode runs and waits for *all*
    selected initial episodes. Auto-reset follow-up episodes are ignored. This
    prevents short failed or successful episodes from replacing slow outcomes
    and biasing the gate. A step limit produces an incomplete report, never a
    successful evaluation of the early-finishing subset.
    """
    import torch

    curriculum = _curriculum_module()
    if type(stage) is not int or not 0 <= stage < len(curriculum.STAGES):
        raise ValueError("Evaluation stage is outside the curriculum.")
    if any(type(value) is not int or value < 1 for value in (episodes_per_side, max_steps)):
        raise ValueError("Evaluation episode and step counts must be positive integers.")
    raw = env.unwrapped
    if not getattr(raw, "evaluation", getattr(raw.cfg, "evaluation", False)):
        raise ValueError("Set the environment to held-out evaluation mode before collecting episodes.")
    if raw.cfg.curriculum_stage != stage:
        raise ValueError("The evaluation environment must use the requested curriculum stage.")
    if env.num_envs < 2:
        raise ValueError("Balanced evaluation requires at least two parallel worlds.")
    spec = curriculum.STAGES[stage]
    attempts = [0, 0]
    successes = [0, 0]
    records = []
    steps = 0
    was_training = getattr(policy, "training", None)
    was_recording = getattr(raw, "recording", False)
    raw.recording = True
    if hasattr(policy, "eval"):
        policy.eval()
    try:
        with torch.inference_mode():
            while min(attempts) < episodes_per_side and steps < max_steps:
                # Fix membership before stepping; faster worlds cannot take another world's place.
                selected = set()
                for side in (0, 1):
                    selected.update(range(side, env.num_envs, 2)[: episodes_per_side - attempts[side]])
                obs, reset_extras = env.reset()
                reset_extras.pop("episode_records", None)
                raw.extras.pop("episode_records", None)
                if hasattr(policy, "reset"):
                    policy.reset()
                remaining = set(selected)
                while remaining and steps < max_steps:
                    obs, _, dones, extras = env.step(policy(obs))
                    steps += 1
                    if hasattr(policy, "reset"):
                        policy.reset(dones)
                    for record in extras.pop("episode_records", []):
                        _validate_record(record, stage, spec)
                        world = record["env_id"]
                        if world not in remaining:
                            continue
                        remaining.remove(world)
                        side = record["server"]
                        attempts[side] += 1
                        successes[side] += int(record["success"])
                        records.append(dict(record))
    finally:
        raw.recording = was_recording
        if was_training is not None and hasattr(policy, "train"):
            policy.train(was_training)
    unsafe = sum(record["unsafe"] for record in records)
    complete = attempts == [episodes_per_side, episodes_per_side]
    report = curriculum.EvaluationReport(
        evaluation_id=evaluation_id,
        stage=stage,
        required_returns=spec.required_returns,
        attempts_by_side=tuple(attempts),
        successes_by_side=tuple(successes),
        unsafe_episodes=unsafe,
        complete=complete,
    )
    rates = [success / count if count else 0.0 for success, count in zip(successes, attempts)]
    unsafe_rate = unsafe / len(records) if records else 0.0
    gate_passed = complete and min(attempts) >= 64 and min(rates) >= 0.8 and unsafe_rate <= 0.05
    metrics = {
        **asdict(report),
        "stage_name": spec.name,
        "episodes_per_side": episodes_per_side,
        "completed_episodes": len(records),
        "success_rate_by_side": rates,
        "unsafe_rate": unsafe_rate,
        "goal_reached_rate": sum(record.get("goal_reached", False) for record in records) / len(records)
        if records
        else 0.0,
        "mean_returns": sum(record["returns"] for record in records) / len(records) if records else 0.0,
        "mean_launch_quality": sum(record.get("launch_quality", 0.0) for record in records) / len(records)
        if records
        else 0.0,
        "returns_histogram": dict(Counter(record["returns"] for record in records)),
        "terminal_reasons": dict(Counter(record["reason"] for record in records)),
        "steps": steps,
        "complete": complete,
        "gate_passed": gate_passed,
        "max_steps_reached": not complete,
        "scope": "Fixed held-out episode cohorts; true physical stage success including final recovery.",
    }
    diagnostic_fields = (
        "head_hits",
        "head_contacts",
        "recovered_miss",
        "recovery_stable_by_player",
        "recovery_hold_s",
        "unsafe_reasons",
        "miss_reasons",
    )
    diagnosed = [record for record in records if all(name in record for name in diagnostic_fields)]
    metrics["diagnostic_episodes"] = len(diagnosed)
    if diagnosed:
        count = len(diagnosed)
        metrics.update(
            mean_head_hits=sum(record["head_hits"] for record in diagnosed) / count,
            head_hit_rate=sum(record["head_hits"] > 0 for record in diagnosed) / count,
            mean_head_contacts=sum(record["head_contacts"] for record in diagnosed) / count,
            recovered_miss_rate=sum(record["recovered_miss"] for record in diagnosed) / count,
            recovery_stable_rate_by_player=[
                sum(record["recovery_stable_by_player"][player] for record in diagnosed) / count for player in (0, 1)
            ],
            recovery_stable_rate_by_role={
                role: sum(record["recovery_stable_by_player"][record["server"] ^ other] for record in diagnosed) / count
                for role, other in (("receiver", 0), ("partner", 1))
            },
            mean_recovery_hold_s=sum(record["recovery_hold_s"] for record in diagnosed) / count,
            unsafe_reasons=dict(Counter(reason for record in diagnosed for reason in record["unsafe_reasons"])),
            miss_reasons=dict(Counter(reason for record in diagnosed for reason in record["miss_reasons"])),
        )
    return report, metrics


def collect(args: argparse.Namespace) -> dict:
    """Load a tagged curriculum actor and run deterministic physical evaluation."""
    import torch
    import warp as wp

    wp.config.enable_backward = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = (checkpoint.get("infos") or {}).get("microduck_curriculum", {})
    profile = metadata.get("profile")
    if profile not in (CURRICULUM_PROFILE, VISION_PROFILE):
        raise ValueError("Convert the source checkpoint for the curriculum or vision task first.")
    stage = args.stage
    if stage is None:
        stage = metadata.get("curriculum_state", metadata.get("state", {})).get("stage", metadata.get("stage"))
    if type(stage) is not int or not 0 <= stage < len(_curriculum_module().STAGES):
        raise ValueError("Supply --stage for a prepared checkpoint without saved curriculum progress.")

    from rsl_rl.runners import OnPolicyRunner

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    if profile == VISION_PROFILE:
        from isaaclab_tasks.contrib.microduck.agents.ball_ppo_cfg import MicroDuckVolleyVisionPPORunnerCfg as AgentCfg
        from isaaclab_tasks.contrib.microduck.volley_vision_env import MicroDuckVolleyVisionEnv as Environment
        from isaaclab_tasks.contrib.microduck.volley_vision_env_cfg import MicroDuckVolleyVisionEnvCfg as EnvCfg
    else:
        from isaaclab_tasks.contrib.microduck.agents.ball_ppo_cfg import (
            MicroDuckVolleyCurriculumPPORunnerCfg as AgentCfg,
        )
        from isaaclab_tasks.contrib.microduck.volley_curriculum_env import MicroDuckVolleyCurriculumEnv as Environment
        from isaaclab_tasks.contrib.microduck.volley_curriculum_env_cfg import MicroDuckVolleyCurriculumEnvCfg as EnvCfg

    cfg = EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.visualizer_cfgs = []
    cfg.seed = args.seed
    cfg.curriculum_stage = stage
    cfg.evaluation = True
    agent = AgentCfg()
    agent.device = args.device
    agent.seed = args.seed
    with launch_simulation(cfg.sim, {"device": args.device, "visualizer": ["none"]}):
        sim_utils.create_new_stage()
        env = Environment(cfg)
        try:
            if profile == VISION_PROFILE and "vision_contract" in metadata:
                if metadata["vision_contract"] != env.vision_contract():
                    raise ValueError("Saved visual observation/camera contract differs from this environment.")
            env.set_evaluation(True, seed=args.seed)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, to_dict(agent), log_dir=None, device=agent.device)
            runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            _, metrics = evaluate_policy(
                wrapped,
                policy,
                stage=stage,
                evaluation_id=args.evaluation_id,
                episodes_per_side=args.episodes_per_side,
                max_steps=args.max_steps,
            )
            metrics.update(
                {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "seed": args.seed,
                    "physics_dt_s": cfg.sim.dt,
                    "policy_dt_s": env.step_dt,
                    "separation_m": cfg.separation,
                    "profile": profile,
                }
            )
            if profile == VISION_PROFILE:
                metrics["vision"] = env.vision_diagnostics()
            return metrics
        finally:
            env.close()


def main() -> int:
    """Write an evaluation report and return nonzero for an unmet or incomplete gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", type=int, choices=range(6))
    parser.add_argument("--seed", type=int, default=1000042)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes-per-side", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--evaluation-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("logs/microduck_volley_evaluation.json"))
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.num_envs < 2 or min(args.episodes_per_side, args.max_steps) < 1:
        parser.error("Supply an existing checkpoint, at least two worlds, and positive episode/step counts.")
    if args.seed < 1000000 or args.evaluation_id < 0:
        parser.error("Use a held-out seed >=1000000 and a non-negative evaluation ID.")
    metrics = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {args.output}")
    return 0 if metrics["gate_passed"] else 1 if metrics["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
