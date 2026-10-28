# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure actual visual-policy impacts and flights on the NVIDIA host, without learning."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


def impact_energy_diagnostics(before, after):
    """Measure ball translational energy, including height; this is not a passive-impact test.

    A moving robot can supply energy. Spin, stored contact energy and actuator
    work are not measured here, so an increase alone cannot certify a solver bug.
    Quantities per unit mass avoid assuming an unmeasured effective contact mass.
    """
    states = [[sample[f"ball_v_{axis}"] for axis in "xyz"] + [sample["ball_z"]] for sample in (before, after)]
    if not all(math.isfinite(value) for state in states for value in state):
        return None
    incoming_squared, outgoing_squared = [sum(value**2 for value in state[:3]) for state in states]
    return {
        "incoming_speed_m_s": math.sqrt(incoming_squared),
        "outgoing_speed_m_s": math.sqrt(outgoing_squared),
        "speed_ratio": math.sqrt(outgoing_squared / incoming_squared) if incoming_squared > 1e-12 else None,
        "translational_energy_change_j_per_kg": (
            0.5 * (outgoing_squared - incoming_squared) + 9.81 * (states[1][3] - states[0][3])
        ),
        "scope": "Ball translation plus gravity only; robot work and ball spin are unmeasured.",
    }


def flight_diagnostics(row, player, *, minimum_flight=0.16, elapsed_since_hit=0.0):
    """Predict a free descending target-plane crossing from a post-impact sample.

    This is a diagnostic, not a measured success or an actor input. Targets
    come from the actual curriculum target, including a coach's virtual disk,
    rather than the parked partner's physical head. Impossible crossings have
    null errors, never a spurious perfect zero.
    """
    target = [row[f"target_{player}_{axis}"] for axis in "xyz"]
    position = [row[f"ball_{axis}"] for axis in "xyz"]
    velocity = [row[f"ball_v_{axis}"] for axis in "xyz"]
    if not all(math.isfinite(value) for value in (*target, *position, *velocity)):
        return {"finite": False, "reachable_plane": False, "quality_gate": False, "predicted_miss_m": None}
    delta = [target[axis] - position[axis] for axis in range(2)]
    horizontal = math.hypot(*delta)
    toward = sum(velocity[axis] * delta[axis] for axis in range(2)) / max(horizontal, 1e-9)
    discriminant = velocity[2] ** 2 + 2 * 9.81 * (position[2] - target[2])
    time = (velocity[2] + math.sqrt(max(discriminant, 0))) / 9.81
    reachable = discriminant >= 0 and time > 0
    miss_x = position[0] + velocity[0] * time - target[0]
    miss_y = position[1] + velocity[1] * time - target[1]
    return {
        "finite": True,
        "velocity_m_s": velocity,
        "toward_target_speed_m_s": toward,
        "apex_height_m": position[2] + max(velocity[2], 0) ** 2 / (2 * 9.81),
        "reachable_plane": reachable,
        "descending_crossing_time_s": time if reachable else None,
        "predicted_miss_m": math.hypot(miss_x, miss_y) if reachable else None,
        "predicted_error_x_m": miss_x if reachable else None,
        "predicted_error_y_m": miss_y if reachable else None,
        "legacy_quality_gate": discriminant > 0 and velocity[2] > 0.1 and time > minimum_flight,
        "quality_gate": discriminant > 0 and velocity[2] > 0.1 and time + elapsed_since_hit > minimum_flight,
    }


def summarize_flights(rows, records, *, release_delay=0.02, minimum_flight=0.16):
    """Use the first separated, delayed sample; never borrow across impacts or resets."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["world"], row["episode"]), []).append(row)
    strikes = []
    for key, samples in grouped.items():
        for index, hit in enumerate(samples):
            candidates = [side for side in ("a", "b") if hit[f"candidate_hit_{side}"]]
            if hit["event"] != "physics" or len(candidates) != 1:
                continue
            player = candidates[0]
            before = samples[index - 1] if index > 0 else None
            if before is not None and (before["event"] != "physics" or before[f"contact_{player}"]):
                before = None
            separated = None
            for after in samples[index + 1 :]:
                if after["event"] == "terminal" or after["raw_head_onsets"]:
                    break
                if after["time_s"] - hit["time_s"] + 1e-6 >= release_delay and not after[f"contact_{player}"]:
                    separated = after
                    break
            strikes.append(
                {
                    "world": key[0],
                    "episode": key[1],
                    "player": player,
                    "hit_time_s": hit["time_s"],
                    "sample_delay_s": None if separated is None else separated["time_s"] - hit["time_s"],
                    "impact_energy": None
                    if before is None or separated is None
                    else impact_energy_diagnostics(before, separated),
                    "flight": None
                    if separated is None
                    else flight_diagnostics(
                        separated,
                        player,
                        minimum_flight=minimum_flight,
                        elapsed_since_hit=separated["time_s"] - hit["time_s"],
                    ),
                }
            )
    sides = {}
    for player in ("a", "b"):
        selected = [entry for entry in strikes if entry["player"] == player]
        valid = [entry["flight"] for entry in selected if entry["flight"] and entry["flight"]["finite"]]
        misses = [flight["predicted_miss_m"] for flight in valid if flight["predicted_miss_m"] is not None]
        sides[player] = {
            "candidate_strikes": len(selected),
            "separated_samples": len(valid),
            "quality_gate_passes": sum(flight["quality_gate"] for flight in valid),
            "legacy_quality_gate_passes": sum(flight["legacy_quality_gate"] for flight in valid),
            "clock_gate_disagreements": sum(
                flight["quality_gate"] and not flight["legacy_quality_gate"] for flight in valid
            ),
            "moving_away": sum(flight["toward_target_speed_m_s"] < 0 for flight in valid),
            "non_upward": sum(flight["velocity_m_s"][2] <= 0.1 for flight in valid),
            "cannot_reach_target_height": sum(not flight["reachable_plane"] for flight in valid),
            "mean_predicted_miss_m": sum(misses) / len(misses) if misses else None,
            "max_outgoing_speed_m_s": max(
                (entry["impact_energy"]["outgoing_speed_m_s"] for entry in selected if entry["impact_energy"]),
                default=None,
            ),
        }
    return {
        "episodes": len(records),
        "success_by_initial_receiver": [
            sum(r["success"] for r in records if r["server"] == side)
            / max(1, sum(r["server"] == side for r in records))
            for side in range(2)
        ],
        "terminal_reasons": dict(Counter(r["reason"] for r in records)),
        "by_player": sides,
        "strikes": strikes,
        "episode_records": records,
        "scope": (
            "Candidate contact onsets before final control-step safety checks; "
            "free-flight predictions are diagnostics, not measured successes."
        ),
    }


def _live_partner_trace_fields(env, player, side):
    """Keep raw RGB partner estimates and the privileged reward reference separately labeled."""
    partner = env.partner_tracks[player]
    fields = {
        f"partner_rgb_visible_{side}": partner.visible,
        f"partner_rgb_valid_{side}": partner.valid,
        f"partner_rgb_age_{side}": partner.age,
        f"reference_valid_{side}": env._reference_valid[:, player],
    }
    for prefix, tensor in (
        ("partner_rgb", partner.position_w - env.scene.env_origins),
        ("partner_rgb_velocity", partner.velocity_w),
        ("desired_release", env._desired_release[:, player]),
    ):
        for index, axis in enumerate("xyz"):
            fields[f"{prefix}_{side}_{axis}"] = tensor[:, index]
    return fields


def collect(args):
    """Collect balanced first-episode cohorts from two actual camera/physics worlds."""
    import torch
    import warp as wp
    from rsl_rl.runners import OnPolicyRunner

    wp.config.enable_backward = False
    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.contrib.microduck.volley_task_settings import apply_task_settings, resolve_task_settings

    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    metadata = (checkpoint.get("infos") or {}).get("microduck_curriculum", {})
    profile = metadata.get("profile")
    if profile not in ("microduck_volley_vision_v2", "microduck_volley_live_pass_v1"):
        raise ValueError("Trace requires native vision-v2 or live-pass weights, without checkpoint adaptation.")
    live_pass = profile == "microduck_volley_live_pass_v1"
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    components = importlib.import_module("scripts.tools.train_microduck_volley")._task_components(True, live_pass)
    EnvCfg, Environment, AgentCfg = components[:3]
    actor = checkpoint.get("actor_state_dict")
    if not isinstance(actor, dict) or any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in actor.values()
    ):
        raise ValueError("Trace requires a finite actor state dictionary.")
    cfg = EnvCfg()
    task_settings = resolve_task_settings(metadata, damping_ratio=args.ball_damping_ratio, live_pass=live_pass)
    apply_task_settings(cfg, task_settings)
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.sim.visualizer_cfgs = []
    cfg.head_camera_panels = False
    cfg.curriculum_stage = args.stage
    cfg.seed = args.seed
    collision = cfg.scene.ball.spawn.collision_props[1]
    rows, records = [], []

    class TracedVision(Environment):
        """Observe the real production hooks, including the final pre-reset state."""

        trace_keys = None
        trace_tick = 0

        def snapshot(self, event, hits=None, onsets=None):
            if not self.trace_keys:
                return
            fields = {}
            for prefix, tensor in (
                ("ball", self._ball.data.root_link_pos_w.torch - self.scene.env_origins),
                ("ball_v", self._ball.data.root_link_lin_vel_w.torch),
            ):
                for index, axis in enumerate("xyz"):
                    fields[f"{prefix}_{axis}"] = tensor[:, index]
            for player, side in enumerate(("a", "b")):
                for prefix, tensor in (
                    ("target", self._targets[:, player] - self.scene.env_origins),
                    ("head", self._head_positions()[:, player] - self.scene.env_origins),
                    ("root", self._robots[player].data.root_link_pos_w.torch - self.scene.env_origins),
                ):
                    for index, axis in enumerate("xyz"):
                        fields[f"{prefix}_{side}_{axis}"] = tensor[:, index]
                quaternion = self._robots[player].data.body_link_quat_w.torch[:, self._head_ids[player]]
                for index, axis in enumerate("xyzw"):
                    fields[f"head_quat_{side}_{axis}"] = quaternion[:, index]
                data, body = self._robots[player].data, self._head_ids[player]
                for prefix, tensor in (
                    ("head_origin", data.body_link_pos_w.torch[:, body] - self.scene.env_origins),
                    ("head_linear_velocity", data.body_link_lin_vel_w.torch[:, body]),
                    ("head_angular_velocity", data.body_link_ang_vel_w.torch[:, body]),
                ):
                    for index, axis in enumerate("xyz"):
                        fields[f"{prefix}_{side}_{axis}"] = tensor[:, index]
                sensor = self._sensors[player].data
                points = sensor.contact_pos_w.torch[:, self._head_masks[player], 0].squeeze(1)
                points = points - self.scene.env_origins
                for index, axis in enumerate("xyz"):
                    fields[f"contact_point_{side}_{axis}"] = points[:, index]
                force = sensor.normal_force_matrix_w.torch[:, self._head_masks[player]].sum(dim=(1, 2))
                for index, axis in enumerate("xyz"):
                    fields[f"head_ball_force_{side}_{axis}"] = force[:, index]
                fields[f"contact_{side}"] = self._contact[:, player]
                fields[f"candidate_hit_{side}"] = torch.zeros_like(self._hits) if hits is None else hits[:, player]
                tracker = self._visual_trackers[player]
                fields[f"rgb_visible_{side}"] = tracker.visible
                fields[f"rgb_valid_{side}"] = tracker.valid
                fields[f"rgb_age_{side}"] = tracker.age
                if live_pass:
                    fields.update(_live_partner_trace_fields(self, player, side))
            fields["raw_head_onsets"] = torch.zeros_like(self._hits) if onsets is None else onsets
            fields["world"] = torch.arange(2, device=self.device)
            fields["episode"] = self._episode_ids
            fields["phase"] = self._episode.phase
            fields["terminal"] = self._terminal_pending
            numbers = torch.stack(tuple(fields.values()), dim=-1).detach().cpu().tolist()
            for values in numbers:
                row = dict(zip(fields, values))
                row["world"], row["episode"] = int(row["world"]), int(row["episode"])
                if (row["world"], row["episode"]) not in self.trace_keys:
                    continue
                if event == "terminal" and not row["terminal"]:
                    continue
                row.update(event=event, time_s=self.trace_tick * self.cfg.sim.dt)
                rows.append(row)

        def _update_contacts(self):
            before = self._player_hit_event.clone()
            raw_before = self._raw_hits.clone()
            super()._update_contacts()
            self.trace_tick += 1
            self.snapshot("physics", self._player_hit_event - before, self._raw_hits - raw_before)

        def _get_rewards(self):
            reward = super()._get_rewards()
            self.snapshot("terminal")
            return reward

    agent = AgentCfg()
    agent.device = args.device
    with launch_simulation(cfg.sim, {"device": args.device, "visualizer": ["none"]}):
        sim_utils.create_new_stage()
        env = TracedVision(cfg)
        try:
            if live_pass:
                env.validate_source_contract(metadata)
            elif metadata.get("vision_contract") != env.vision_contract():
                raise ValueError("Checkpoint and current camera/task contracts differ.")
            env.set_evaluation(True, seed=args.seed)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, to_dict(agent), log_dir=None, device=args.device)
            policy = runner.get_inference_policy(device=args.device)
            policy.load_state_dict(actor, strict=True)
            steps = 0
            with torch.inference_mode():
                for _ in range(args.episodes_per_side):
                    env.trace_keys = None
                    obs, _ = wrapped.reset()
                    if env._server.tolist() != [0, 1]:
                        raise RuntimeError("Diagnostic cohorts require receiver A/world 0 and B/world 1.")
                    env.extras.pop("episode_records", None)
                    policy.reset()
                    wanted = {(world, int(episode)) for world, episode in enumerate(env._episode_ids.tolist())}
                    env.trace_keys = wanted.copy()
                    while wanted and steps < args.max_steps:
                        obs, _, dones, extras = wrapped.step(policy(obs))
                        steps += 1
                        policy.reset(dones)
                        for record in extras.pop("episode_records", []):
                            key = record["env_id"], record["episode_id"]
                            if key in wanted:
                                records.append(record)
                                wanted.remove(key)
                    if wanted:
                        break
            return (
                rows,
                records,
                {
                    "steps": steps,
                    "complete": len(records) == 2 * args.episodes_per_side,
                    "checkpoint_iteration": checkpoint.get("iter"),
                    "checkpoint_profile": profile,
                    "stage": args.stage,
                    "objective": env.stages[args.stage].objective,
                    "physics_dt_s": cfg.sim.dt,
                    "release_delay_s": cfg.release_delay,
                    "minimum_flight_s": cfg.minimum_flight_time,
                    "target_radius_m": cfg.target_radius,
                    "seed": args.seed,
                    "authored_ball_solref": list(collision.solref),
                    "task_settings": task_settings,
                    "contact_experiment": args.ball_damping_ratio is not None,
                    "training_performed": False,
                    "vision": env.vision_diagnostics(),
                },
            )
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=range(6), default=2)
    parser.add_argument("--episodes-per-side", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1000042)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ball-damping-ratio",
        type=float,
        help="Diagnostic-only solref damping override; saved weights remain unchanged.",
    )
    parser.add_argument("--output", type=Path, default=Path("logs/microduck_aim_trace.json"))
    args = parser.parse_args()
    if not args.checkpoint.is_file() or min(args.episodes_per_side, args.max_steps) < 1 or args.seed < 1000000:
        parser.error("Supply visual weights, positive episode/step counts and a held-out seed >=1000000.")
    if args.ball_damping_ratio is not None and not 0 < args.ball_damping_ratio <= 1:
        parser.error("Use 0 < --ball-damping-ratio <= 1 for the diagnostic contact comparison.")
    rows, records, metadata = collect(args)
    report = summarize_flights(
        rows, records, release_delay=metadata["release_delay_s"], minimum_flight=metadata["minimum_flight_s"]
    )
    report.update(metadata, checkpoint=str(args.checkpoint.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if rows:
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps({key: value for key, value in report.items() if key not in ("strikes", "episode_records")}, indent=2)
    )
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
