# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace a saved volley policy at physics-step resolution without changing rewards or training.

Run on the NVIDIA host with --checkpoint PATH. Writes a CSV and a JSON summary.
The CSV captures impacts before automatic resets, including subsequent torso collapse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def predicted_miss(row: dict, receiver: str) -> float | None:
    """Predict XY error at the receiver's initial forehead height, assuming free ballistic flight.

    This is a diagnostic approximation: the receiver may move and a contact may
    still be applying impulse at the sampled time. It is not a measured return.
    """
    height = row[f"home_head_{receiver}_z"]
    discriminant = row["ball_v_z"] ** 2 + 2 * 9.81 * (row["ball_z"] - height)
    if not math.isfinite(discriminant) or discriminant < 0:
        return None
    flight = (row["ball_v_z"] + math.sqrt(discriminant)) / 9.81
    if flight <= 0:
        return None
    error_x = row["ball_x"] + row["ball_v_x"] * flight - row[f"home_head_{receiver}_x"]
    error_y = row["ball_y"] + row["ball_v_y"] * flight - row[f"home_head_{receiver}_y"]
    return math.hypot(error_x, error_y)


def summarize(rows: list[dict]) -> dict:
    """Compare the first, second and subsequent candidate strikes without crossing episode boundaries."""
    episodes = {}
    for row in rows:
        episodes.setdefault(row["episode"], []).append(row)
    strikes = []
    endings = []
    for episode, samples in episodes.items():
        endings.extend(row for row in samples if row["event"] == "terminal")
        for index, hit in enumerate(samples):
            if hit["candidate_hit_a"] and hit["candidate_hit_b"]:
                continue  # Substep sampling cannot establish a physical order for simultaneous candidates.
            for player, name in enumerate(("a", "b")):
                if not hit[f"candidate_hit_{name}"]:
                    continue
                # Use the first sample >=20 ms later, but never a terminal/reset or a newer impact.
                after = None
                for candidate in samples[index + 1 :]:
                    if candidate["event"] == "terminal" or candidate["physical_head_onsets"]:
                        break
                    if candidate["time_s"] + 1e-6 >= hit["time_s"] + 0.02:
                        after = candidate
                        break
                window = [r for r in samples[index:] if r["time_s"] <= hit["time_s"] + 0.25]
                strikes.append(
                    {
                        "episode": episode,
                        "strike": hit["strike"],
                        "player": name,
                        "time_s": hit["time_s"],
                        "tilt_at_hit_deg": hit[f"tilt_{name}_deg"],
                        "max_tilt_next_250ms_deg": max(r[f"tilt_{name}_deg"] for r in window),
                        "inward_root_drift_next_250ms_m": max(
                            (1 if player == 0 else -1) * (r[f"root_{name}_x"] - hit[f"root_{name}_x"]) for r in window
                        ),
                        "velocity_sample_delay_s": None if after is None else after["time_s"] - hit["time_s"],
                        "sampled_ball_speed_m_s": None
                        if after is None
                        else math.sqrt(sum(after[f"ball_v_{axis}"] ** 2 for axis in "xyz")),
                        "predicted_home_head_miss_m": None
                        if after is None
                        else predicted_miss(after, "b" if player == 0 else "a"),
                    }
                )
    grouped = {}
    for number, label in ((1, "first_strike"), (2, "second_strike"), (3, "later_strikes")):
        selected = [s for s in strikes if s["strike"] == number or (number == 3 and s["strike"] > 3)]
        grouped[label] = {"count": len(selected)}
        for metric in (
            "tilt_at_hit_deg",
            "max_tilt_next_250ms_deg",
            "inward_root_drift_next_250ms_m",
            "predicted_home_head_miss_m",
        ):
            values = [s[metric] for s in selected if s[metric] is not None and math.isfinite(s[metric])]
            grouped[label][f"mean_{metric}"] = mean(values) if values else None
            grouped[label][f"samples_{metric}"] = len(values)
    return {
        "completed_episodes": len(endings),
        "ambiguous_candidate_steps": sum(bool(r["candidate_hit_a"] and r["candidate_hit_b"]) for r in rows),
        "by_strike": grouped,
        "terminal_reason_fractions": {
            key: mean(row[key] for row in endings) if endings else None
            for key in (
                "fell",
                "robot_collision",
                "crossed_center",
                "ball_outside",
                "ball_ground",
                "invalid_contact",
                "timeout",
            )
        },
        "strikes": strikes,
        "scope": (
            "Candidate legal contact onsets; same-step safety may cancel reward. "
            "Ballistic predictions assume a stationary initial receiver and no later contacts. "
            "Tilt/drift windows are clipped at reset, not at the next strike. Termination causes can overlap."
        ),
    }


def collect(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Evaluate the same task/checkpoint and capture states before automatic environment resets."""
    import warp as wp

    wp.config.enable_backward = False

    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.contrib.microduck.agents.ball_ppo_cfg import MicroDuckVolleyPPORunnerCfg
    from isaaclab_tasks.contrib.microduck.ball_env import MicroDuckBallEnv
    from isaaclab_tasks.contrib.microduck.ball_env_cfg import MicroDuckVolleyEnvCfg

    rows = []

    class TracedVolley(MicroDuckBallEnv):
        """Observe the production environment; leave actions, physics, rewards and resets intact."""

        def __init__(self, cfg):
            self.trace_episode = -1
            self.trace_tick = 0
            self.trace_begin = 0
            self.trace_strike = 0
            self.trace_completed = 0
            super().__init__(cfg)

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            self.trace_episode += 1
            self.trace_begin = self.trace_tick
            self.trace_strike = 0
            self.trace_homes = self._head_positions()[0].clone()

        def _apply_action(self):
            super()._apply_action()
            self.trace_tick += 1

        def _snapshot(self, event, hits=(0, 0), physical_onsets=0):
            fields = {}
            origin = self.scene.env_origins[0]

            def vector(prefix, value, axes="xyz"):
                for axis, component in zip(axes, value):
                    fields[f"{prefix}_{axis}"] = component.reshape(1)

            vector("ball", self._ball.data.root_link_pos_w.torch[0] - origin)
            vector("ball_v", self._ball.data.root_link_lin_vel_w.torch[0])
            heads = self._head_positions()[0]
            for i, (name, robot) in enumerate(zip(("a", "b"), self._robots)):
                data = robot.data
                vector(f"root_{name}", data.root_link_pos_w.torch[0] - origin)
                vector(f"root_v_{name}", data.root_link_lin_vel_w.torch[0])
                vector(f"root_angular_{name}", data.root_link_ang_vel_b.torch[0])
                vector(f"head_{name}", heads[i] - origin)
                vector(f"home_head_{name}", self.trace_homes[i] - origin)
                vector(f"head_quat_{name}", data.body_link_quat_w.torch[0, self._head_ids[i]], "wxyz")
                fields[f"tilt_{name}_deg"] = torch.rad2deg(
                    torch.acos((-data.projected_gravity_b.torch[0, 2]).clamp(-1, 1))
                ).reshape(1)
            numbers = torch.cat(list(fields.values())).detach().cpu().tolist()
            flags = (
                torch.stack(
                    [
                        self._fell[0],
                        self._robot_collision[0],
                        self._crossed_center[0],
                        self._ball_outside[0],
                        self._ground[0],
                        self._rally_invalid[0],
                        self.reset_time_outs[0],
                        self._post_rally.active[0],
                    ]
                )
                .cpu()
                .tolist()
            )
            row = {
                "episode": self.trace_episode,
                "time_s": (self.trace_tick - self.trace_begin) * self.cfg.sim.dt,
                "event": event,
                "strike": self.trace_strike,
                "candidate_hit_a": int(hits[0]),
                "candidate_hit_b": int(hits[1]),
                "physical_head_onsets": int(physical_onsets),
            }
            row.update(zip(fields, numbers))
            row.update(
                zip(
                    (
                        "fell",
                        "robot_collision",
                        "crossed_center",
                        "ball_outside",
                        "ball_ground",
                        "invalid_contact",
                        "timeout",
                        "post_rally",
                    ),
                    flags,
                )
            )
            rows.append(row)

        def _update_contacts(self):
            previous = self._player_hit_event[0].clone()
            raw = self._raw_hits[0].clone()
            super()._update_contacts()
            hits = (self._player_hit_event[0] - previous).cpu().tolist()
            onsets = int((self._raw_hits[0] - raw).item())
            self.trace_strike += int(sum(hits))
            self._snapshot("impact" if onsets else "physics", hits, onsets)

        def _get_rewards(self):
            result = super()._get_rewards()
            if bool(self.reset_buf[0]):
                self.trace_completed += 1
                self._snapshot("terminal")
            return result

    cfg = MicroDuckVolleyEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.sim.visualizer_cfgs = []
    cfg.seed = args.seed
    agent = MicroDuckVolleyPPORunnerCfg()
    agent.device = args.device
    agent.seed = args.seed
    with launch_simulation(cfg.sim, {"device": args.device, "visualizer": ["none"]}):
        sim_utils.create_new_stage()
        env = TracedVolley(cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, to_dict(agent), log_dir=None, device=agent.device)
            runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            obs = wrapped.get_observations()
            with torch.inference_mode():
                for _ in range(args.max_steps):
                    obs, _, dones, _ = wrapped.step(policy(obs))
                    policy.reset(dones)
                    if env.trace_completed >= args.episodes:
                        break
            return rows, {
                "device": args.device,
                "physics_dt_s": cfg.sim.dt,
                "policy_dt_s": env.step_dt,
                "separation_m": cfg.separation,
                "serve_height_m": cfg.serve_height,
                "serve_down_speed_m_s": cfg.serve_down_speed,
                "post_rally_duration_s": cfg.post_rally_duration,
                "max_steps_reached": env.trace_completed < args.episodes,
            }
        finally:
            env.close()


def main() -> int:
    """Save a reproducible one-world rollout and first-versus-second strike diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path("logs/microduck_volley_trace.json"))
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.episodes < 1 or args.max_steps < 1:
        parser.error("Supply an existing checkpoint and positive episode/step counts.")
    rows, metadata = collect(args)
    report = summarize(rows)
    report.update(metadata)
    report.update(
        {"checkpoint": str(args.checkpoint.resolve()), "seed": args.seed, "requested_episodes": args.episodes}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if rows:
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: value for key, value in report.items() if key != "strikes"}, indent=2))
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")
    return 1 if metadata["max_steps_reached"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
