# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train nonvisual MicroDuck ballet and save measured curriculum levels with every checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def evaluate_full_routine(runner, wrapped, env, seed, episodes=128):
    """Evaluate each member of a fixed cohort once, including slow successes and early failures."""
    import torch

    saved = env.curriculum.state_dict()
    old_enabled = env.cfg.curriculum
    env.cfg.curriculum = False
    policy = runner.get_inference_policy(device=env.device)
    count = min(episodes, env.num_envs)
    active = torch.arange(env.num_envs, device=env.device) < count
    successes = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    try:
        env.seed(seed)
        env.curriculum.stage.fill_(4)
        with torch.inference_mode():
            wrapped.reset()
            observations = wrapped.get_observations()
            for _ in range(env.max_episode_length + 1):
                observations, _, done, _ = wrapped.step(policy(observations))
                finished = active & done.bool()
                successes[finished] = env.last_success[finished]
                active &= ~finished
                if not bool(active.any()):
                    break
        if bool(active.any()):
            raise RuntimeError("Full ballet evaluation did not finish its fixed cohort within the episode limit.")
        return {
            "seed": seed,
            "episodes": count,
            "successes": int(successes.sum()),
            "success_rate": float(successes.sum()) / count,
            "stage": 4,
            "deterministic": True,
            "objective": "one_foot_full_turn_then_body_and_head_bow_then_grounded_recovery",
            "certified": False,
        }
    finally:
        env.curriculum.load_state_dict(saved)
        env.cfg.curriculum = old_enabled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="Warm start from the 61-input single MicroDuck walking actor.")
    source.add_argument("--resume", type=str, help="Resume a native ballet checkpoint, or latest.")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument(
        "--iterations", type=int, default=20000, help="Additional learning iterations, including on resume."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viz", choices=("none", "viser"), default="none")
    parser.add_argument("--max-visible-envs", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.num_envs, args.iterations, args.max_visible_envs) < 1 or args.seed < 0:
        parser.error("Environment/iteration counts must be positive and seed nonnegative.")

    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.contrib.microduck.agents.ballet_ppo_cfg import MicroDuckBalletPPORunnerCfg
    from isaaclab_tasks.contrib.microduck.ballet import BALLET_PROFILE, OBSERVATION_WIDTH, walking_actor_warm_start
    from isaaclab_tasks.contrib.microduck.ballet_env import MicroDuckBalletEnv
    from isaaclab_tasks.contrib.microduck.ballet_env_cfg import MicroDuckBalletEnvCfg

    root = Path("logs/rsl_rl/microduck_ballet")
    source_path = args.checkpoint or (Path(args.resume) if args.resume and args.resume != "latest" else None)
    if args.resume == "latest":
        candidates = sorted(root.glob("*/latest.pt"))
        if not candidates:
            parser.error("No saved ballet latest.pt exists.")
        source_path = candidates[-1]
    saved = None
    warm_actor = None
    metadata = {}
    if source_path is not None:
        if not source_path.is_file():
            parser.error(f"Checkpoint does not exist: {source_path}")
        saved = torch.load(source_path, map_location="cpu", weights_only=True)
        metadata = (saved.get("infos") or {}).get("microduck_ballet", {})
        if args.resume and metadata.get("profile") != BALLET_PROFILE:
            parser.error("--resume requires a native ballet checkpoint; use --checkpoint for walking weights.")
        if args.checkpoint and metadata:
            parser.error("Use --resume to continue a ballet checkpoint.")
        if args.checkpoint:
            warm_actor = walking_actor_warm_start(saved["actor_state_dict"])

    cfg = MicroDuckBalletEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    agent = MicroDuckBalletPPORunnerCfg()
    agent.seed = args.seed
    agent.device = args.device
    if args.resume:
        cfg.support_foot = metadata["support_foot"]
        cfg.turn_direction = metadata["turn_direction"]
        if metadata.get("physics_dt") != cfg.sim.dt:
            raise ValueError("Saved ballet physics clock differs from the current environment.")
    run_dir = root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")

    with launch_simulation(
        cfg.sim, {"device": args.device, "visualizer": [args.viz], "max_visible_envs": args.max_visible_envs}
    ):
        sim_utils.create_new_stage()
        env = MicroDuckBalletEnv(cfg)

        class BalletRunner(OnPolicyRunner):
            def save(self, path, infos=None):
                payload = dict(infos or {})
                payload["microduck_ballet"] = {
                    "profile": BALLET_PROFILE,
                    "curriculum": env.curriculum.state_dict(),
                    "support_foot": cfg.support_foot,
                    "turn_direction": cfg.turn_direction,
                    "physics_dt": cfg.sim.dt,
                    "source": str(source_path) if source_path else None,
                }
                temporary = str(path) + ".tmp"
                super().save(temporary, infos=payload)
                os.replace(temporary, path)
                latest = run_dir / "latest.pt"
                if Path(path) != latest:
                    shutil.copyfile(path, run_dir / "latest.pt.tmp")
                    os.replace(run_dir / "latest.pt.tmp", latest)

        try:
            # Check the real scene/sensor and actual 89/14 interface before a long run.
            with torch.inference_mode():
                env.reset()
                seen = torch.zeros(2, dtype=torch.bool, device=env.device)
                for _ in range(12):
                    obs, reward, _, _, _ = env.step(torch.zeros(env.num_envs, 14, device=env.device))
                    frame = env._measure()
                    if (
                        obs["policy"].shape != (env.num_envs, OBSERVATION_WIDTH)
                        or not bool(frame.finite.all())
                        or not bool(torch.isfinite(reward).all())
                    ):
                        raise RuntimeError("Ballet preflight found an invalid physical state or observation interface.")
                    seen |= (frame.forces[..., 2] > 0.05).any(0)
                if not bool(seen.all()):
                    raise RuntimeError("Ballet preflight could not measure upward support forces on both named feet.")
                env.reset()
            (run_dir / "preflight.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "observations": OBSERVATION_WIDTH,
                        "actions": 14,
                        "foot_support_measured": seen.tolist(),
                    },
                    indent=2,
                )
                + "\n"
            )
            wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
            runner = BalletRunner(wrapped, to_dict(agent), log_dir=str(run_dir), device=args.device)
            if args.resume:
                env.curriculum.load_state_dict(metadata["curriculum"])
                runner.load(str(source_path), map_location=args.device)
                env.reset()
            elif args.checkpoint:
                runner.alg.actor.load_state_dict(warm_actor, strict=True)
                print(
                    f"[Ballet] Walking actor loaded; fresh critic, optimizer and curriculum: {source_path}", flush=True
                )
            (run_dir / "environment.json").write_text(json.dumps(to_dict(cfg), default=str, indent=2) + "\n")
            (run_dir / "agent.json").write_text(json.dumps(to_dict(agent), default=str, indent=2) + "\n")
            runner.save(str(run_dir / "initial.pt"))
            print(f"[Ballet] Run: {run_dir}; no cameras, no scripted forces, all 14 servos trainable.", flush=True)
            finished = True
            try:
                runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
            except KeyboardInterrupt:
                finished = False
                runner.save(str(run_dir / "interrupted.pt"))
                print(f"[Ballet] Interrupted checkpoint: {run_dir / 'interrupted.pt'}", flush=True)
            if finished:
                report = evaluate_full_routine(runner, wrapped, env, args.seed + 1000000)
                (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"[Ballet evaluation] {json.dumps(report)}", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
