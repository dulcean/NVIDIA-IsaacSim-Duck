# MicroDuck volleyball curriculum

One driver prepares weights, trains receive/return/recover lessons, checks physical outcomes,
and progresses to an unassisted pair. Legacy tasks/checkpoints remain available.
This is an implemented protocol, **not a GPU-verified learned policy**. CPU tests cannot
guarantee that PPO will learn volleyball within a fixed iteration budget.

## Start on the NVIDIA machine

After pulling, use the preserved paired model, not the failed root/direction experiments:

```bash
nohup uv run --extra viser python scripts/tools/train_microduck_volley.py \
  --checkpoint logs/rsl_rl/microduck_volley_post_rally_20k/2026-10-25_23-57-20/model_40250.pt \
  --num-envs 1024 --iterations 20000 --viz viser \
  > volley_curriculum.log 2>&1 < /dev/null &
```

Standing 61/14 and heading 71/14 checkpoints are also accepted. The driver converts a copy
to 142/28 with a fresh critic/optimizer, then checks finite observations/rewards, completed
episodes, reset records and ankle support before learning. This is a runtime preflight,
not a successful skill evaluation. No source file or checkpoint is overwritten.

Omit `--extra viser` and `--viz viser` to train without the browser visualizer.
Viser displays at most four worlds by default (`--visible-envs`), independently of training batch size.
`--iterations` is the additional learning budget; evaluation is extra. The process can stop
early on success or stagnation. A completed budget never implies passing.

```bash
tail -f volley_curriculum.log
```

Unique runs are stored under `logs/rsl_rl/microduck_volley_curriculum/`:

| Output | Meaning |
|---|---|
| `initial.pt` | Converted source copy for a fresh curriculum |
| `model_*.pt`, `latest.pt` | Full model, optimizer and curriculum state |
| `stage_*_best.pt` | Best measured physical result at each lesson |
| `preflight.json` | Runtime episode/reset/support check |
| `evaluation_*.json` | Held-out outcomes and advancement decision |
| `final_test_*.json` | Independent final-stage test |
| `ready.pt` | Written only after final validation and test pass |
| `status.json` | Training, passing, stalled, exhausted, interrupted or error status |
| `launch.json`, `environment.json`, `agent.json`, `git/` | Parameters, configs and source provenance |

`stalled` or `budget_exhausted` means incomplete, with process exit code 2 and resumable
weights. After several evaluations without measured progress the driver restores the best
current-stage model once; if progress still does not return, it stops. It never promotes
by iteration count or rising reward.

## Training and success contract

| Stage | Distribution | Physical objective |
|---|---|---|
| 0 | Varied vertical toss, 0.35 m root spacing | Return through virtual target, then recovery |
| 1 | Lateral incoming ball, 0.40 m spacing | Same objective on both receiving sides |
| 2 | Varied lateral flight/offsets, 0.45 m spacing | Same objective across the wider distribution |
| 3 | Actual pair, 0.45 m spacing | 2 alternating returns and recovery |
| 4 | Actual pair | 4 returns and recovery |
| 5 | Actual pair | 8 returns and recovery |

In coach lessons the other robot is parked farther away, remains policy-controlled, and
must also recover. All leg/head joints remain trainable. Ball position and velocity are
set only at reset using an exact ballistic solve; there are no artificial in-flight kicks,
steering, repeated assisted re-serves, scripted hits or walls.

A pair return requires a new alternating forehead contact after an actual center crossing
above 0.20 m and at least 0.16 s of flight. Prolonged holding, near-center dribbling, invalid
surfaces, collisions and boundary violations do not prove a volley. Coach success requires
an actual descending swept crossing of a 6.5 cm target disk. Targets use ball-center clearance
above the initial forehead. Outgoing trajectory shaping is evaluated after separation and
at least 20 ms after contact onset; a prediction never counts as success.

Every goal gets a full 0.6 s recovery window. Both players must be stable throughout its
final 0.25 s: near standing height/home position, modest torso tilt/angular rate, low
vertical/planar speed, and upward support on at least one ankle. Jumping and brief crouching
are allowed; airborne motion cannot count as completed recovery. A miss receives a recovery
observation window but cannot become success. Safety failures and timeouts override success.

Every 500 iterations by default, evaluation freezes normalization, uses a separate RNG stream
and measures at least 64 complete episodes per receiving side. Fixed cohorts wait for slow
episodes rather than replacing them with fast auto-resets. Incomplete batches cannot pass.
Each side needs >=80% full success and unsafe episodes <=5%, twice consecutively, to advance.
A separate test seed must pass the final gate before `ready.pt` is created.

This is a finite eight-return-and-recovery benchmark under the implemented simulation
distribution, not a claim of endless rallies, RGB perception, hardware transfer or arbitrary
new physics. Raw simulator ball/robot state is used; the camera direction governs valid
forehead strikes but there is no RGB image encoder.

## Observations, weights and rewards

The policy stays 142 inputs / 28 actions. The 13 formerly zero channels per 71-value block
now disclose root linear velocity, home displacement, relative height, own-hit age, recovery
elapsed time/hold progress, lesson mode, phase and stage. Existing channels describe ball,
target and touch order. Conversion folds the normalized old zero-channel contribution into
the first bias before zeroing those weights and giving new channels identity statistics.
Tests with real RSL-RL verify unchanged legacy deterministic actions, including dense
cross-player weights and nonzero normalization means. All new channels and joints train.

Rewards: 1 per qualified physical return, 0.05 per legal strike, <=0.2 post-separation launch
quality, 3 for full goal plus recovery, safety cost 3, miss/recovery failure costs. Idle
no-contact episodes cannot earn positive return. Balance-potential improvement is bounded,
closes at terminal states and uses the same 0.995 discount as PPO. Reward/value loss never
select advancement or certify readiness.

## Stop, resume, evaluate and view

Find the Python process with `pgrep -af train_microduck_volley.py` and send `kill -SIGINT PID`.
If an optimizer update is interrupted, the last fully written safe checkpoint is retained;
a temporary partial write cannot replace it.

```bash
nohup uv run --extra viser python scripts/tools/train_microduck_volley.py \
  --resume logs/rsl_rl/microduck_volley_curriculum/RUN/latest.pt \
  --num-envs 1024 --iterations 20000 --viz viser \
  > volley_curriculum_resume.log 2>&1 < /dev/null &
```

Resume restores model, optimizer, lesson, gate streaks and evaluation numbering. Physics
episodes restart; it is not a bit-identical continuation of each world or RNG stream.
Incompatible curriculum contracts and warm-start-only weights are rejected by `--resume`.

```bash
uv run --extra viser isaaclab play \
  --rl_library rsl_rl --task IsaacContrib-Volley-Curriculum-MicroDuck \
  --num_envs 1 --viz viser \
  --checkpoint logs/rsl_rl/microduck_volley_curriculum/RUN/ready.pt
```

Playback selects the actual pair at stage 5 and permits up to 32 returns/30 s before goal
recovery/reset. That longer preview is distinct from the verified eight-return criterion.
For the exact benchmark with a fresh independent test seed:

```bash
uv run python scripts/tools/evaluate_microduck_volley.py \
  --checkpoint logs/rsl_rl/microduck_volley_curriculum/RUN/ready.pt \
  --stage 5 --seed 5000042
```

CPU tests cover serve geometry, swept targets, conversion/trainability, episode priority,
evaluation cohort bias, advancement/resume and physical checkpoint selection. The GPU test
`source/isaaclab_tasks/test/test_microduck_volley_curriculum.py` exercises all stages, actual
ankle support, both robots' actuation and partial reset metadata. It has not been executed
on the development Mac; runtime preflight is not a substitute for passing learned behavior.
The generated environment-browser check also remains unavailable in this partial development
checkout: its `docs/source` browser assets and local simulation/Gym dependencies are absent.
The registered task and workflow are documented here instead of fabricating generated rows or previews.
