# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare, train, validate and checkpoint the entire MicroDuck volleyball curriculum.

Run on the NVIDIA host. A ready.pt file is produced only after the final physical
rally/recovery gate and a separate test seed pass. Exhaustion or stagnation is
reported explicitly and keeps resumable weights; it is never reported as success.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def _tool(name):
    """Load neighboring tools both from a checkout CLI and from unit tests."""
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"scripts.tools.{name}")


def progress_score(metrics):
    """Select progress by physical outcomes; reward and value loss cannot select the best model."""
    if not metrics["complete"]:
        return None
    return (
        min(metrics["success_rate_by_side"])
        + 0.1 * metrics.get("goal_reached_rate", 0.0)
        - 0.1 * metrics["unsafe_rate"]
        + 0.01 * metrics.get("mean_launch_quality", 0.0)
    )


def resume_metadata(checkpoint, profile):
    """Refuse legacy weights or incomplete curriculum state as a full training resume."""
    infos = checkpoint.get("infos") or {}
    metadata = infos.get("microduck_curriculum", {}) if isinstance(infos, dict) else {}
    if not isinstance(metadata, dict) or metadata.get("profile") != profile or "curriculum_state" not in metadata:
        raise ValueError(
            "--resume requires a checkpoint saved by this curriculum driver; use --checkpoint to warm start."
        )
    optimizer = checkpoint.get("optimizer_state_dict")
    if (
        not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("state"), dict)
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer["param_groups"]
    ):
        raise ValueError("Full resume requires optimizer state.")
    if any(
        not isinstance(checkpoint.get(name), dict) or not checkpoint[name]
        for name in ("actor_state_dict", "critic_state_dict")
    ):
        raise ValueError("Full resume requires actor and critic state dictionaries.")
    iteration = checkpoint.get("iter")
    completed = metadata.get("completed_iterations")
    if (
        type(iteration) is not int
        or iteration < 0
        or type(completed) is not int
        or not iteration <= completed <= iteration + 1
    ):
        raise ValueError("Invalid or inconsistent resume iteration counts.")
    try:
        saved = metadata["curriculum_state"]
        contract = _tool("evaluate_microduck_volley")._curriculum_module()
        validator = contract.CurriculumState(**saved["gate"])
        validator.load_state_dict(saved)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Incomplete or incompatible curriculum state.") from error
    if metadata.get("certified", False) and (
        not validator.complete
        or not isinstance(metadata.get("certification"), dict)
        or not metadata["certification"].get("gate_passed", False)
    ):
        raise ValueError("A certified checkpoint needs a completed curriculum and passing independent test.")
    return metadata


def _task_components(vision):
    """Load one complete observation/profile contract before constructing the simulation."""
    if vision:
        from isaaclab_tasks.contrib.microduck.agents.ball_ppo_cfg import MicroDuckVolleyVisionPPORunnerCfg as AgentCfg
        from isaaclab_tasks.contrib.microduck.volley_vision_env import MicroDuckVolleyVisionEnv as Environment
        from isaaclab_tasks.contrib.microduck.volley_vision_env_cfg import MicroDuckVolleyVisionEnvCfg as EnvCfg

        prepare = _tool("prepare_microduck_vision_checkpoint")
        prepare_weights = prepare.prepare_vision_checkpoint
        profile = prepare.VISION_PROFILE
        experiment = "microduck_volley_vision"
    else:
        from isaaclab_tasks.contrib.microduck.agents.ball_ppo_cfg import (
            MicroDuckVolleyCurriculumPPORunnerCfg as AgentCfg,
        )
        from isaaclab_tasks.contrib.microduck.volley_curriculum_env import MicroDuckVolleyCurriculumEnv as Environment
        from isaaclab_tasks.contrib.microduck.volley_curriculum_env_cfg import MicroDuckVolleyCurriculumEnvCfg as EnvCfg

        prepare = _tool("prepare_microduck_curriculum_checkpoint")
        prepare_weights = prepare.prepare_curriculum_checkpoint
        profile = prepare.CURRICULUM_PROFILE
        experiment = "microduck_volley_curriculum"
    return EnvCfg, Environment, AgentCfg, prepare_weights, profile, experiment


def _vision_metrics(env, vision):
    """Keep runtime perception diagnostics separate from curriculum outcome counts."""
    return {"vision": env.vision_diagnostics()} if vision else {}


def train(args):
    """Use one environment pool for learning and fixed-cohort evaluation, with independent RNG streams."""
    import warp as wp

    wp.config.enable_backward = False

    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.contrib.microduck.volley_curriculum import CurriculumState

    EnvCfg, Environment, AgentCfg, prepare_weights, profile, experiment = _task_components(args.vision)
    evaluator = _tool("evaluate_microduck_volley")
    state = CurriculumState(min_attempts_per_side=args.eval_episodes_per_side)
    run_dir = args.output or Path("logs/rsl_rl") / experiment / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    source_path = args.resume or args.checkpoint
    source = torch.load(source_path, weights_only=True, map_location="cpu")
    history = {}
    total = 0
    test_trials = 0
    certified = False
    certification = None
    if args.resume:
        metadata = resume_metadata(source, profile)
        state.load_state_dict(metadata["curriculum_state"])
        total = int(metadata.get("completed_iterations", source["iter"]))
        history = metadata.get("best_by_stage", {})
        test_trials = metadata.get("test_trials", 0)
        certified = metadata.get("certified", False)
        certification = metadata.get("certification")
        if state.complete and not certified:
            state.complete = False  # A final validation pass interrupted before the independent test is not certified.
        initial = source_path
    else:
        converted = prepare_weights(source)
        initial = run_dir / "initial.pt"
        torch.save(converted, initial)
    cfg = EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.curriculum_stage = state.stage
    cfg.seed = args.seed
    if args.viz == "none":
        cfg.sim.visualizer_cfgs = []
    agent = AgentCfg()
    agent.device = args.device
    agent.seed = args.seed
    (run_dir / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    (run_dir / "environment.json").write_text(json.dumps(to_dict(cfg), default=str, indent=2) + "\n")
    (run_dir / "agent.json").write_text(json.dumps(to_dict(agent), default=str, indent=2) + "\n")
    best = dict(history)
    status = {"status": "passed" if certified else "starting", "run_directory": str(run_dir), "stage": state.stage}
    last_safe = None
    learning = False
    uncertain_update = False

    class CurriculumRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            nonlocal last_safe
            payload = {
                "microduck_curriculum": {
                    "profile": profile,
                    "curriculum_state": state.state_dict(),
                    "completed_iterations": self.current_learning_iteration + int(learning),
                    "best_by_stage": best,
                    "test_trials": test_trials,
                    "certified": certified,
                    "certification": certification if certified else None,
                    "vision_contract": env.vision_contract() if args.vision else None,
                }
            }
            temporary = f"{path}.tmp"
            super().save(temporary, infos=payload)
            os.replace(temporary, path)
            last_safe = Path(path)

    def save_status():
        temporary = run_dir / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        os.replace(temporary, run_dir / "status.json")

    def close_writer(runner):
        writer = runner.logger.writer
        if writer is not None and hasattr(writer, "close"):
            writer.close()
        runner.logger.writer = None

    with launch_simulation(
        cfg.sim,
        {
            "device": args.device,
            "visualizer": [args.viz],
            "max_visible_envs": args.visible_envs,
        },
    ):
        sim_utils.create_new_stage()
        env = Environment(cfg)
        runner = None
        try:
            if args.resume and args.vision and metadata.get("vision_contract") != env.vision_contract():
                raise ValueError("Saved visual observation/camera contract differs from this environment.")
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = CurriculumRunner(wrapped, to_dict(agent), log_dir=str(run_dir), device=args.device)
            runner.add_git_repo_to_log(__file__)
            runner.load(
                str(initial),
                load_cfg={"actor": True, "critic": True, "iteration": True, "optimizer": bool(args.resume)},
                map_location=args.device,
            )
            runner.current_learning_iteration = total
            runner.save(str(run_dir / "latest.pt"))
            policy = runner.get_inference_policy(device=args.device)
            # Exercise full physical episodes and auto-reset metadata, not just the frames before a serve arrives.
            devices = [torch.device(args.device).index or 0] if args.device.startswith("cuda") else []
            seen, supported = set(), [False, False]
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                env.set_evaluation(True, seed=999999)
                obs, _ = wrapped.reset()
                for preflight_step in range(env.max_episode_length + 2):
                    obs, reward, dones, _ = wrapped.step(policy(obs))
                    if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                        raise RuntimeError("GPU preflight produced nonfinite observations or reward.")
                    for record in env.extras.pop("episode_records", []):
                        evaluator._validate_record(record, state.stage, state.spec)
                        if record["env_id"] in (0, 1):
                            seen.add(record["server"])
                    for player, (sensor, feet) in enumerate(zip(env._sensors, env._feet)):
                        supported[player] |= bool(
                            (sensor.data.net_normal_forces_w.torch[:, feet, 2] > cfg.supported_force).any()
                        )
                    policy.reset(dones)
                    if len(seen) == 2 and all(supported):
                        break
                else:
                    raise RuntimeError("GPU preflight did not observe both episode outcomes and ankle support.")
            if args.vision and any(count == 0 for count in env.vision_diagnostics()["camera_frames_seen"]):
                raise RuntimeError("Visual preflight did not observe fresh RGB frames from both head cameras.")
            (run_dir / "preflight.json").write_text(
                json.dumps(
                    {
                        "completed_sides": sorted(seen),
                        "ankle_support": supported,
                        "steps": preflight_step + 1,
                        "runtime_passed": True,
                        "skill_assessed": False,
                        "vision": env.vision_diagnostics() if args.vision else None,
                    },
                    indent=2,
                )
                + "\n"
            )
            env.set_evaluation(False)
            with torch.inference_mode():
                wrapped.reset()
            end_iteration = total + args.iterations
            restore_used = set()
            while total < end_iteration and not state.complete:
                amount = min(args.eval_interval, end_iteration - total)
                status.update(status="training", stage=state.stage, completed_iterations=total)
                save_status()
                learning = True
                runner.learn(amount, init_at_random_ep_len=False)
                learning = False
                close_writer(runner)
                total += amount
                runner.current_learning_iteration = total
                evaluation_id = state.last_evaluation_id + 1
                policy = runner.get_inference_policy(device=args.device)
                devices = [torch.device(args.device).index or 0] if args.device.startswith("cuda") else []
                with torch.random.fork_rng(devices=devices):
                    env.set_evaluation(True, seed=1000000 + args.seed + evaluation_id * 10007)
                    report, metrics = evaluator.evaluate_policy(
                        wrapped,
                        policy,
                        stage=state.stage,
                        evaluation_id=evaluation_id,
                        episodes_per_side=args.eval_episodes_per_side,
                    )
                key = str(state.stage)
                metrics.update(_vision_metrics(env, args.vision))
                score = progress_score(metrics)
                improved = score is not None and score > best.get(key, {}).get("score", -float("inf")) + 1e-6
                if improved:
                    best[key] = {"score": score, "path": str((run_dir / f"stage_{key}_best.pt").resolve())}
                    state.failed_evaluations = 0
                decision = state.update_evaluation(report)
                metrics["decision"] = {"status": decision.status, "reason": decision.reason}
                (run_dir / f"evaluation_{evaluation_id:04d}.json").write_text(json.dumps(metrics, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "evaluation": evaluation_id,
                            "stage": int(key),
                            "decision": decision.status,
                            "success": metrics["success_rate_by_side"],
                            "unsafe": metrics["unsafe_rate"],
                        }
                    ),
                    flush=True,
                )
                if improved:
                    # Save the stage selection state, not an unlabelled inference-only actor.
                    runner.save(best[key]["path"])
                if state.complete:
                    test_trials += 1
                    with torch.random.fork_rng(devices=devices):
                        env.set_evaluation(True, seed=3000000 + args.seed + test_trials * 100003)
                        _, certification = evaluator.evaluate_policy(
                            wrapped,
                            policy,
                            stage=state.stage,
                            evaluation_id=1000000 + test_trials,
                            episodes_per_side=args.eval_episodes_per_side,
                        )
                    (run_dir / f"final_test_{test_trials}.json").write_text(json.dumps(certification, indent=2) + "\n")
                    if certification["gate_passed"]:
                        certified = True
                        runner.save(str(run_dir / "ready.pt"))
                        status.update(status="passed", final_test=str(run_dir / f"final_test_{test_trials}.json"))
                        break
                    state.complete = False
                    state.consecutive_passes = 0
                    state.failed_evaluations = 0
                if decision.status == "stalled":
                    candidate = best.get(key)
                    if candidate and key not in restore_used and Path(candidate["path"]).is_file():
                        runner.load(candidate["path"], map_location=args.device)
                        runner.current_learning_iteration = total
                        restore_used.add(key)
                        state.failed_evaluations = 0
                        print(f"Restored best stage {key} model after stagnant evaluations.", flush=True)
                    else:
                        status.update(status="stalled", reason=decision.reason)
                        break
                env.set_stage(state.stage)
                env.set_evaluation(False)
                with torch.inference_mode():
                    wrapped.reset()
                    runner.logger.cur_reward_sum.zero_()
                    runner.logger.cur_episode_length.zero_()
                runner.save(str(run_dir / "latest.pt"))
                status.update(stage=state.stage, completed_iterations=total)
                save_status()
            if status["status"] not in ("passed", "stalled"):
                status.update(
                    status="budget_exhausted", reason="Iteration budget ended before the physical final gate passed."
                )
        except KeyboardInterrupt:
            uncertain_update = learning
            learning = False
            status.update(status="interrupted", reason="Stopped by SIGINT; latest checkpoint is resumable.")
        except BaseException as error:
            uncertain_update = True
            learning = False
            status.update(status="error", reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            if runner is not None and not uncertain_update:
                runner.save(str(run_dir / "latest.pt"))
            elif last_safe is not None:
                destination = run_dir / "latest.pt"
                if last_safe != destination:
                    shutil.copy2(last_safe, destination)
                safe = torch.load(destination, weights_only=True, map_location="cpu")
                total = safe["infos"]["microduck_curriculum"]["completed_iterations"]
                state.load_state_dict(safe["infos"]["microduck_curriculum"]["curriculum_state"])
            status.update(stage=state.stage, completed_iterations=total)
            save_status()
            if runner is not None:
                close_writer(runner)
            env.close()
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint", type=Path, help="Legacy standing, heading, or paired weights for fresh curriculum."
    )
    source.add_argument("--resume", type=Path, help="latest.pt from this driver; restores curriculum and optimizer.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-envs", type=int, help="Defaults to 32 with --vision, otherwise 1024.")
    parser.add_argument("--vision", action="store_true", help="Use real head RGB tracking for the actor's ball inputs.")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-episodes-per-side", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viz", choices=("none", "viser"), default="none")
    parser.add_argument("--visible-envs", type=int, default=4)
    args = parser.parse_args()
    if args.num_envs is None:
        args.num_envs = 32 if args.vision else 1024
    path = args.resume or args.checkpoint
    if not path.is_file() or args.num_envs < 2 or min(args.iterations, args.eval_interval) < 1:
        parser.error("Supply an existing source checkpoint, >=2 worlds, and positive iteration counts.")
    if args.eval_episodes_per_side < 64 or args.seed < 0 or args.visible_envs < 0:
        parser.error("Use at least 64 evaluation episodes per side and a non-negative training seed.")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
