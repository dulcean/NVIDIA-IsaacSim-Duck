# MicroDuck head cameras and visual volleyball

For the camera-only live partner profile, independent actors, physical passing
gates and uninterrupted long runs, use [the live passing guide](microduck-live-pass.md).
The coach-profile commands below retain their earlier lesson semantics.

`IsaacContrib-Volley-Vision-MicroDuck` attaches actual RGB sensors to both moving heads.
It uses the existing Newton/MJWarp physics and Viser. The actor's ball inputs come only
from camera pixels; the critic retains simulator state during training. All 28 joint
actions remain trainable, including the legs. This implements vision, not a demonstrated
volleyball policy: rendering and policy learning still need validation on an NVIDIA GPU.

## Source model and calibration

The implementation reads the original hardware/software conventions:

- [MicroDuck alpha-family camera calibration](https://github.com/pollen-robotics/microduck/blob/main/robotd-params/src/lib.rs):
  1280×720 intrinsic matrix and five Brown–Conrady distortion coefficients.
- [MicroDuck media pipeline](https://github.com/pollen-robotics/microduck/blob/main/mediad/src/pipeline.rs):
  native sensor frames, clockwise display rotation and 30 Hz capture.
- [MicroDuck RL camera orientation issue](https://github.com/pollen-robotics/microduck_rl/issues/66):
  the exported MJCF camera faces backwards; the corrected upright view looks along
  local −Z with image-up along head +X.

The published calibration was measured on the alpha unit `graphite`; it is a family
reference, not a calibration of the user's specific robot. Images are rendered at
160×90 without cropping, with intrinsics scaled by eight and the original lens distortion.
The detector consumes native frames. Only the displayed/saved images rotate clockwise
to 90×160; the intrinsic matrix and tracker never mix rotated and raw coordinates.

The USD's `head_camera` is a guide sphere, not a functioning sensor. A new `HeadCamera`
under the rigid `jaw_soft` follows the physical head joints. Its optical origin is placed
at head-local `(0.0155, -0.0000913778, -0.083)` m, just in front of the opaque lens mesh.
The guide centre and older camera-direction proxy are inside opaque geometry. Newton's
ray renderer ignores the near clipping plane, so clipping cannot repair that obstruction.
This render-origin correction is not a measurement of the real entrance pupil.

## First run: camera preview without training

On the NVIDIA host, after pulling:

```bash
uv run --extra viser python scripts/tools/view_microduck_vision.py \
  --checkpoint logs/rsl_rl/microduck_volley_curriculum/2026-10-26_13-12-26-462912/model_1400.pt \
  --num-envs 4 --steps 1500 --viz viser
```

The existing Viser server on port 8080 keeps its 3D scene and adds a **Head cameras**
folder with both images and a world selector. Only the selected images transfer to the
CPU, at up to 30 wall-clock updates per second. Rendering and tracking continue without
browser clients. Omit the `viser` extra and set `--viz none` for headless inspection.
Preview is paced to physical time by default; `--no-real-time` runs without pacing.
`--no-head-panels` keeps the 3D scene while excluding the image panels. `--panel-fps`
sets their presentation ceiling; it does not change sensor or policy timing.

`logs/microduck_vision_preview/` receives the first/last valid RGB snapshots and
`preview.json`: fresh-frame counts, accepted pixel measurements, track ages and measured
position errors against simulation truth. Truth labels also define training rewards and
task qualification; they never initialize the tracker or replace actor ball features. Zero error samples
means no accuracy measurement, even if the error sum is zero. A ball above the forehead
may be outside a forward-facing camera; zero detections alone do not prove sensor failure.
Check the images and the physical camera direction. Preview checks runtime execution,
not successful volleyball. It does not change weights or run optimizer updates.

The preview accepts legacy standing/heading/paired weights, teacher curriculum weights,
and vision-v1 weights, adapting a copy in memory to `microduck_volley_vision_v2`.
Source checkpoint files remain unchanged. A current v2 checkpoint loads directly and
validates its saved camera contract. Migration preserves weights, not proof that the new
visual curriculum has been learned.

For the physical sensor integration check on that host:

```bash
uv run --extra test python -m pytest source/isaaclab_tasks/test/test_microduck_vision.py -q
```

This uses one two-world Newton scene, actual distorted RGB, actual airborne spheres,
head actuation and isolated resets. It verifies pixel-based position estimates separately
from volleyball skill. This test has not been run on the development Mac.

## Perception and control

`BallVisualTracker` finds compact orange regions in actual RGB, rejects ambiguous,
border-clipped and poorly shaped candidates, undistorts their image coordinates and
estimates a sphere's range from its known 2.5 cm radius. It receives no depth images,
semantic/instance labels, simulated ball pose or hidden initial state. This is a calibrated
color/sphere detector, not a general learned visual encoder. A similar round orange object
can be ambiguous; different balls, lighting and materials need detector validation.

Measurements transform through the captured camera pose into world coordinates before
velocity estimation, compensating head motion. Camera timing is independent of the 50 Hz
policy. The requested 1/30 s capture period quantizes to the global physics clock (normally
35 ms with 5 ms physics steps); tracking uses the actual fresh-frame interval. Re-reading
a cached frame does not produce another measurement. After a missed frame, a bounded
ballistic prediction lasts at most 0.15 s; expiry returns zero ball features and an invalid
age flag. Partial resets clear only the selected tracks and discard captures from the
intermediate reset scene. No tracker is initialized from the reset ball coordinates.

The v2 actor has **152 inputs: two 76-value blocks**. Each block retains 71
task/proprioceptive channels, replaces all six simulator ball position/velocity channels
with visual estimates, and appends measurement visibility, normalized track age and a
three-value search bearing. The bearing points towards the last RGB-measured world
position in the robot's current root frame. It survives short-track expiry until that
world resets; it is stale search memory, not a hidden estimate of the ball's current pose.
Before any visual measurement, the bearing and ball features are zero. Expired ball
position/velocity features remain zero even when search memory exists.

The separate 142-input critic may use ball truth. Referee turn/phase, root kinematics
and joint/IMU state remain available; this does not model a full hardware
odometry/localization stack. Exposure, motion blur and camera/estimator latency have
not been calibrated to hardware.

## Learn to look before learning to return

The visual curriculum has its own objectives and graduation evidence. It does not
reuse the teacher curriculum's first lesson or progress:

| Stage | Objective before grounded recovery |
| --- | --- |
| 0 — watch | Receive at least 0.30 s of continuous, fresh RGB evidence of the real ball before a head impact. |
| 1 — receive | Make a legal contact on the raised, camera-facing forehead. |
| 2 — aim | Return the ball through the virtual partner's measured target disk. |
| 3 — pair | Complete two actual alternating replies. |
| 4 — pair | Complete four actual alternating replies. |
| 5 — pair | Complete eight actual alternating replies with both players' qualified visibility. |

Incoming throws begin ahead of the receiver and vary in flight time, height and target
offset. A reset sets the initial ball position/velocity; physical gravity and contacts
determine everything afterwards. No joint is frozen and no neck trajectory is scripted.
Every objective is followed by the full 0.6 s recovery window; both robots must remain
supported and stable during its final 0.25 s. A predicted landing score cannot satisfy
a target crossing or an actual pair return.

Small active-gaze terms reward moving the camera towards the ball even before it enters
the image, and maintaining RGB evidence after acquisition. Their combined maximum is
0.20 reward per second, averaged over both robots. Only actual contact and the first
80 ms after that robot's own strike mask gaze; mere proximity does not. Current physical
camera poses and ball truth label these rewards. Visibility credit additionally requires
real RGB detections matched to the ball; projection or a predicted tracker position alone
cannot earn it. These labels never enter actor features.

Final-stage success requires at least eight replies after the initial strike, followed
by grounded recovery. In each successful final episode, **each robot** also needs real
ball visibility for at least 60% of its eligible active-play time, excluding the strike
grace above. This does not require 100% visibility during impact and does not certify
an endless rally.

## Visual training and resume

One command prepares a new v2 warm start, checks sensors, trains, evaluates held-out
episodes, saves best checkpoints and advances lessons automatically:

```bash
nohup uv run python scripts/tools/train_microduck_volley.py \
  --vision \
  --checkpoint logs/rsl_rl/microduck_volley_curriculum/2026-10-26_13-12-26-462912/model_1400.pt \
  --num-envs 32 --iterations 20000 --eval-interval 500 --viz none \
  > volley_vision_v2.log 2>&1 < /dev/null &
```

`--vision` defaults to 32 worlds because two RGB views per world change the GPU workload.
Actual throughput and memory use for this v2 pipeline have not been measured here.
Increase the count only after measuring on the target GPU. The long-run command is
headless; for a live Viser UI use `uv run --extra viser` and `--viz viser` instead.
Visual runs live under `logs/rsl_rl/microduck_volley_vision/`.
The converter preserves the source actor's mapping when old and visual ball features match,
inserts initially neutral new feature connections, and creates a fresh critic/optimizer.
It does not assume that noisy/absent visual estimates preserve the old physical behavior.

Before any optimizer update, the driver calls the production sensor probe in the existing
environment. It checks visible spheres and background rejection for both receiver identities,
fresh nonconstant RGB, calibration agreement and localization error. It also checks the
first actual camera exposure of every lesson's normal serve, with no replacement
fixture, to catch a blind initial distribution before learning. It writes
`vision_preflight.json` and stops if the sensor check fails. A separate physical preflight
then checks finite policy stepping, episode/reset records and ankle support.

Every 500 iterations, evaluation collects fixed cohorts with at least 64 complete episodes
per initial receiver. A lesson advances only after two consecutive evaluations with at
least 80% full success on **each** side and no more than 5% unsafe episodes. Fast episodes
cannot replace unfinished worlds in the cohort. Visual objective labels are validated
independently, so watching, striking and returning cannot exchange success evidence.

`evaluation_*.json` includes per-side `watch_fraction_by_server`, peak visibility duration,
camera/role visibility, physical failure causes and recovery outcomes. Partial watch
progress may select a better checkpoint and postpone stagnation, in 5% duration bins;
it cannot advance a lesson. Training reward alone never selects the best model.

`ready.pt` appears only after the final qualified eight-return/recovery gate and a separate
held-out test seed pass. `latest.pt` and `stage_*_best.pt` remain intermediate weights.
Twenty thousand iterations is a maximum budget, not a promise of convergence: the driver
can stop earlier on a failed preflight, interruption or unresolved stagnation. The current
v2 preflight and learning behavior have not yet been validated on the target GPU. CPU
tests and migrated weights do not establish that the robots can play.

Monitor the run with `tail -f volley_vision_v2.log` and its `status.json`. A `stalled` or
`budget_exhausted` status preserves weights and reports incomplete training; neither is success.

Resume a visual run with its own saved optimizer and sensor contract:

```bash
uv run --extra viser python scripts/tools/train_microduck_volley.py \
  --vision --resume logs/rsl_rl/microduck_volley_vision/RUN/latest.pt \
  --num-envs 32 --iterations 20000 --viz viser
```

This resume command applies only to a current v2 driver checkpoint and adds up to another
20,000 iterations. Vision-v1 checkpoints must use `--checkpoint` for fresh v2 preparation;
their old optimizer and lesson progress are incompatible with the new objectives.
Camera optics, image geometry, timing, tracker/search settings, gaze rates and final
visibility qualification are stored in **sensor contract version 3** with the checkpoint.
Incompatible sensor contracts fail explicitly. The evaluator detects visual checkpoints
automatically, runs the RGB actor, and keeps physical outcome gates separate from cumulative
perception diagnostics. Direct task playback selects the pair through `play_mode()`; preview
defaults to lesson 0 and accepts `--stage 0` through `--stage 5` for exact lesson inspection.

CPU validation covers analytic image geometry, distorted projection, camera motion,
dropouts, actor truth isolation and real RSL-RL checkpoint conversion. GPU rendering and
learning are not established by these CPU tests. This partial development checkout also
lacks the generated environment-browser assets under `docs/source`; the registry updater
cannot run here without the simulation/Gym dependency environment.

## Continuing a run that learned only one player

The uploaded run `2026-10-26_23-24-49-874011` passed actual RGB localization
for both cameras (about 3 mm error in the controlled near-ball probe) and the
first-exposure checks for all lessons. Starting at iteration 1500, every held-out
evaluation gave receiver success `[0.0, 1.0]`, without unsafe episodes. All 64
failed receiver-A episodes ended with invalid head contact before the watch
objective. At iteration 5000, A's mean peak continuous visibility was only about
25 ms against the 300 ms lesson requirement. This establishes a learned
asymmetry, rather than evidence that either camera never produces usable pixels.
The first-exposure checks alone do not establish correct tracking throughout a moving-head trajectory.

Use `--vision --resume RUN/latest.pt --player-symmetry` to enable RSL-RL's player
exchange augmentation and policy-consistency loss. Each real experience is also
trained with complete A/B observation and action blocks exchanged. The actor's
76-value blocks and critic's 71-value blocks are already robot-local, so this is
a pi rotation of the court without flipping servo or local vector signs. It
does not introduce privileged critic ball data into actor inputs.

This changes the learning method without changing the actor architecture,
physical task, camera definitions, success thresholds or stage. Existing actor,
critic and Adam state are restored. The first explicit activation resets only
the stagnation counter and pass streak; subsequent resumes retain the enabled
method and counters. PPO logs the additional `symmetry` loss. This shares learning
signals; it does not directly copy B's output weights into A or guarantee physical
skill transfer. A short held-out run must confirm improved A success and retained B success.

CPU verification covers the actual RSL-RL augmented PPO update and an update with
the uploaded 5000-iteration weights/Adam state, not a physical rollout. Run a
bounded continuation before another long training budget:

```bash
nohup uv run python scripts/tools/train_microduck_volley.py \
  --vision --player-symmetry \
  --resume logs/rsl_rl/microduck_volley_vision/2026-10-26_23-24-49-874011/latest.pt \
  --num-envs 256 --iterations 1000 --eval-interval 500 --viz none \
  > volley_vision_symmetry.log 2>&1 < /dev/null &
```

The driver creates a new run directory and adds 1000 iterations to the saved
5000. It evaluates both receivers at 5500 and 6000; completion of this budget
alone is not a passing skill result. Future resumes of its new `latest.pt`
automatically keep player symmetry enabled.

## Diagnosing inaccurate aimed returns

The next uploaded run `2026-10-27_13-52-42-834914` passed forehead reception
and reached `aim_return`. Across its stage-2 evaluations, legal-hit counts were
about 0.89--0.98 per episode while target success stayed below 10% per side.
The final cohort contained 124 misses and four successes; 70 episodes included
a ball-outside miss and 35 a ball-ground miss. Reason counts may overlap. RGB
position errors were about 8--9 mm over accepted measurements across the run;
this does not measure accuracy during every impact or camera dropout.

A confirmed reward-clock mismatch has been corrected: delayed launch quality
now counts elapsed time since the strike. A correct 170 ms arc observed 20 ms
after impact has only 150 ms remaining, but still satisfies the actual 160 ms
strike-to-target horizon. Previously its launch bonus was zero. The impact-time
caller's default remains zero elapsed time. This correction does not establish
the cause of all observed misses or demonstrate improved physical behavior.

Before allocating another learning budget, collect actual post-impact motion:

```bash
uv run python scripts/tools/trace_microduck_vision.py \
  --checkpoint logs/rsl_rl/microduck_volley_vision/2026-10-27_13-52-42-834914/stage_2_best.pt \
  --stage 2 --episodes-per-side 8 \
  --output logs/microduck_aim_trace.json
```

This loads the existing visual actor without adaptation, optimizer updates or
physics changes. Two normal worlds use balanced, fixed first-episode cohorts.
It writes JSON and CSV with 5 ms samples, actual target disks, ball velocities,
head orientations, contact points/forces, tracker validity, and terminal episode
records. It samples outgoing flight only after separation and the configured
20 ms delay, without crossing resets or subsequent head impacts. Predictions
use the real virtual target, not the parked coach's head. The summary separates
wrong-direction, non-upward and unreachable-height launches and compares old
versus corrected timing gates. Predictions do not count as measured successes.
CPU tests cover the analytic diagnostic boundaries; actual contact comparisons
require execution on the NVIDIA host.

The first held-out trace of `stage_2_best.pt` (iteration 14000) completed 16
episodes: one success, 15 candidate legal strikes and no unsafe episodes. The
median predicted target miss after separation was 44 cm, against a 6.5 cm goal
radius. The old and corrected flight-clock gates agreed on every sampled
strike, so that clock fix does not explain this cohort's poor accuracy.
Eight of the 15 strikes increased the ball's translational mechanical energy
(including gravity). In the largest case speed rose from 2.92 to 6.53 m/s;
nearby strike-point motion was about 0.2--0.3 m/s. These observations warrant
a contact-model check before further training. They do not prove passive
energy creation: the articulated head can perform work, and neither ball spin
nor actuator work was recorded. A passing floor-drop test alone does not
validate the head collision.

For a single-variable comparison, use the same checkpoint, seed and episode
count, overriding only the ball's authored MuJoCo damping ratio:

```bash
uv run python scripts/tools/trace_microduck_vision.py \
  --checkpoint logs/rsl_rl/microduck_volley_vision/2026-10-27_13-52-42-834914/stage_2_best.pt \
  --stage 2 --episodes-per-side 8 --seed 1000042 \
  --ball-damping-ratio 0.5 \
  --output logs/microduck_aim_damping_050.json
```

This is a diagnostic intervention, not a new training default or a certificate
of task success. The original authored `solref` is `(0.02, 0.15)`; the candidate
is `(0.02, 0.5)`. Damping ratio is not restitution. The parameter convention
is documented in the [MuJoCo solver-parameter reference](https://mujoco.readthedocs.io/en/3.8.0/modeling.html#solver-parameters).
The report labels this intervention, records the authored parameters and seed,
and reports incoming/outgoing speeds and translational energy changes without
classifying a powered strike as a passive bounce. New CSV traces also include
head-origin linear/angular velocities, so surface velocity can be evaluated at
the contact point rather than inferred from head-center motion alone. Check
that extreme speed gains disappear without eliminating useful upward returns;
do not choose damping from target success on 16 episodes alone. No weights,
optimizer state, task defaults, joint controls or camera settings are modified.

The `0.5` comparison subsequently completed 16 episodes with zero successes and
zero unsafe outcomes. All 15 sampled strikes lost translational mechanical
energy; maximum outgoing speed was 1.18 m/s, versus 6.53 m/s in the original
cohort. Five launches could not reach the target-height plane, and 13 failed the
hard quality gate. This supports investigating controlled stronger strikes under
the more damped contact. It does not calibrate restitution to a real ball or
establish timestep convergence. Identical seeds do not make later cohorts exact
paired serves: different terminal times can trigger extra automatic resets and
consume the serve generator differently.

An opt-in continuous release reward now ranks weak launches without changing
the physical success metric. It scores the descending target-height crossing
when reachable, or the future apex in 3D when the ball cannot reach the target
height, with a continuous flight-time deficit. This removes the zero-feedback
region for weak strikes while still assigning low scores to large overshoots.
It is paid once after a legal release through the existing 0.2 coefficient;
there are no scripted forces or joint restrictions. `Metrics/launch_progress`
tracks this shaping separately from the unchanged gated `Metrics/launch_quality`.
Neither prediction can satisfy a curriculum gate or certify a playable policy.

For a bounded adaptation run from the best visual actor:

```bash
nohup uv run python scripts/tools/train_microduck_volley.py \
  --vision --player-symmetry \
  --checkpoint logs/rsl_rl/microduck_volley_vision/2026-10-27_13-52-42-834914/stage_2_best.pt \
  --start-stage 2 \
  --ball-damping-ratio 0.5 --continuous-launch-shaping \
  --num-envs 256 --iterations 1000 \
  --eval-interval 250 --eval-episodes-per-side 64 \
  --device cuda:0 --seed 42 --viz none \
  > volley_aim_adaptation.log 2>&1 &
```

This is an exploratory adaptation changing contact damping and release shaping
together, not a single-variable causal ablation or a verified convergence recipe.
The old actor and its normalizer load exactly; the critic, optimizer, counters,
best-model history and certification start fresh. Camera/sensor contract changes
are still rejected. `--start-stage` selects a fresh lesson only and cannot be
combined with a full resume. Starting at stage 2 is an explicit warm start, not
a claim that earlier lessons passed under the modified physics.

New checkpoints save `task_settings`. Resumes, the visual viewer, evaluator and
trace tool restore those contact/reward settings even when their CLI flags are
omitted. A changed contact or reward on `--resume` is rejected; use `--checkpoint`
to start a new objective. Old checkpoints retain their authored `(0.02, 0.15)`
contact and legacy gated reward by default. Preserve the original checkpoints.
Compare `evaluations/*.json`, actual per-side target success and safety before
extending the 1000-iteration budget; reward growth alone does not prove accuracy.

## Preview performance

Capture uses a global clock independent of each world's episode reset: at the default
settings both cameras capture every seven physics ticks (35 ms). Scene sensor updates
are lazy; contact data is still read every 5 ms. The renderer therefore receives synchronized
worlds rather than rendering its entire camera batch whenever any independently reset
world happens to be due. The UI and artifact saver read the last captured buffers without
requesting new renders. Tracking predictions still advance at physics resolution.

The tracker batches candidate geometry, reuses pixel grids and avoids copying snapshots
on each physics tick. Detection uses orange color ratios with a lower brightness floor;
this preserves the shaded brown portion of the real rendered sphere instead of measuring
only its bright cap. Uploaded RGB frames provide a regression test for that failure.

The report's `timing` separates setup/compilation, the first ten warmup steps, playback,
policy calls, environment steps, diagnostics, artifacts and pacing waits. Rates exclude
setup and warmup. `control_steps_per_second` measures policy steps, and `real_time_factor`
measures simulated seconds per wall second; neither is a browser FPS measurement.
`diagnostics.panels` measures the head panels only. These are host wall timings, not a
CUDA kernel profile. Interrupted previews retain completed-step counts and the last valid
snapshots. GPU throughput still needs measurement on the target hardware.

For a short unpaced measurement with the same checkpoint and four worlds:

```bash
uv run --extra viser python scripts/tools/view_microduck_vision.py \
  --checkpoint logs/rsl_rl/microduck_volley_curriculum/2026-10-26_13-12-26-462912/model_1400.pt \
  --num-envs 4 --steps 210 --viz viser --no-real-time \
  --output logs/microduck_vision_performance
```

Keep the browser connected during that measurement. Comparing `--viz none` with Viser,
or `--no-head-panels` with the full UI, isolates presentation overhead without learning.
Camera scheduling is part of sensor contract version 3 along with the v2 visual curriculum
and search memory. A full visual resume carrying an older contract is rejected. Ordinary
curriculum/legacy weights and vision-v1 weights remain valid inputs to the preview's
in-memory v2 visual adapter.
