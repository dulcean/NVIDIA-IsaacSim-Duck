# MicroDuck head cameras and visual volleyball

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
position errors against simulation truth. Truth is diagnostic only. Zero error samples
means no accuracy measurement, even if the error sum is zero. A ball above the forehead
may be outside a forward-facing camera; zero detections alone do not prove sensor failure.
Check the images and the physical camera direction. Preview checks runtime execution,
not successful volleyball. It does not change weights or run optimizer updates.

The preview accepts legacy standing/heading/paired weights and curriculum weights,
adapting a copy in memory. A vision checkpoint loads directly and validates its saved
camera contract. Checkpoints from earlier tasks do not already contain a learned visual policy.

For the physical sensor integration check on that host:

```bash
uv run python -m pytest source/isaaclab_tasks/test/test_microduck_vision.py -q
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

The actor has 146 inputs: two 73-value blocks. Each block preserves 71 task/proprioceptive
channels, replaces the six ball position/velocity channels with visual estimates, then
appends fresh-measurement visibility and normalized track age. All true ball position and
velocity channels are replaced, including when the camera sees nothing. The separate
142-input critic may use ball truth. Referee turn/phase, root kinematics and joint/IMU
state remain available; this does not model a full hardware odometry/localization stack.
Exposure, motion blur and camera/estimator latency have not been calibrated to hardware.

## Visual training and resume

After camera/pixel tracking validation, the existing driver supports the visual task:

```bash
nohup uv run --extra viser python scripts/tools/train_microduck_volley.py \
  --vision \
  --checkpoint logs/rsl_rl/microduck_volley_curriculum/2026-10-26_13-12-26-462912/model_1400.pt \
  --num-envs 32 --iterations 500 --viz viser \
  > volley_vision.log 2>&1 < /dev/null &
```

`--vision` defaults to 32 worlds because two RGB views per world change the GPU workload.
Actual throughput and memory use have not been measured here. Increase the count only
after measuring on the target GPU. Visual runs live under `logs/rsl_rl/microduck_volley_vision/`.
The converter preserves the source actor's mapping when old and visual ball features match,
inserts initially neutral visibility/age connections, and creates a fresh critic/optimizer.
It does not assume that noisy/absent visual estimates preserve the old physical behavior.

The existing first lesson and physical success/recovery criteria remain unchanged. Uploaded
checks at 500/1000 iterations had zero target crossings; adding vision does not resolve that
known physical learning failure. Do not interpret a completed visual training budget, fresh
camera frames or increasing reward as a working volley. The curriculum still requires
balanced held-out physical success and a separate final test before writing `ready.pt`.

Resume a visual run with its own saved optimizer and sensor contract:

```bash
uv run --extra viser python scripts/tools/train_microduck_volley.py \
  --vision --resume logs/rsl_rl/microduck_volley_vision/RUN/latest.pt \
  --num-envs 32 --iterations 500 --viz viser
```

Camera optics, image geometry, timing and tracker settings are stored with the checkpoint.
Incompatible sensor contracts fail explicitly. The evaluator detects visual checkpoints
automatically, runs the RGB actor, and keeps physical outcome gates separate from cumulative
perception diagnostics. Direct task playback selects the pair through `play_mode()`; preview
defaults to lesson 0 and accepts `--stage 0` through `--stage 5` for exact lesson inspection.

CPU validation covers analytic image geometry, distorted projection, camera motion,
dropouts, actor truth isolation and real RSL-RL checkpoint conversion. GPU rendering and
learning are not established by these CPU tests. This partial development checkout also
lacks the generated environment-browser assets under `docs/source`; the registry updater
cannot run here without the simulation/Gym dependency environment.

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
Camera scheduling is part of sensor contract version 2; a full visual resume carrying
the older contract is rejected. Ordinary curriculum/legacy weights remain valid inputs
to the preview's in-memory visual adapter.
