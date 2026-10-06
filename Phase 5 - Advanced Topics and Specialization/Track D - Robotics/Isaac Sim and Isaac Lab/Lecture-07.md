# Lecture 07: Sensors, Rendering, and Synthetic Data

## Overview

Physics is usually cheap. Pixels usually are not. One 640×480 camera with RGB, depth and segmentation can cost more GPU memory and frame time than the physics of thousands of state-only environments. On an 8 GB card, rendering is the resource that decides what you can build. This lecture is the VRAM lecture of the course.

It covers the Isaac Sim 6.x sensor stack: RTX cameras with render products and annotators, tiled cameras, RTX lidar and radar, and the physics sensors (IMU, contact, raycast). It then uses Replicator to generate a randomized, labelled dataset. Every lab measures what each sensor costs in milliseconds, megabytes of VRAM and megabytes per second of disk. The lecture ends with a measured answer to "how many cameras fit on my card?"

By the end you should be able to:

* create cameras and lidars with `isaacsim.sensors.experimental.rtx` and read their data as Warp arrays without needless host copies
* explain render products, annotators, sensor tick rates and multi-tick rendering, and set a sensor rate on purpose
* use the physics sensors (`IMUSensor`, `ContactSensor`, raycast) and say why they are nearly free next to RTX sensors
* build a Replicator pipeline (randomize → step → annotate → write) and measure its frames/s and MB/s
* predict and then measure the VRAM cost of N cameras at a given resolution and annotator set, with and without tiling
* state the maximum camera count at 128², 256² and 640×480 on your GPU

---

## 1. Why it matters: pixels are the expensive part

| Job | Sensors | What breaks first on 8 GB |
|---|---|---|
| Sim-in-the-loop (Nav2, perception stacks) | 1-4 cameras at real resolutions, a lidar, an IMU | Frame time → real-time factor < 1 (Lecture 08) |
| Synthetic data generation | 1-few cameras, many annotators, high quality settings | Render time per frame, then disk throughput |
| Vision-based RL | One small camera per env × hundreds of envs | VRAM per env (Lecture 10) |

The same camera is cheap in one job and too expensive in another. The cost model in §5 lets you reason about all three before you run anything. The labs then replace the model's unknowns with your GPU's numbers.

---

## 2. Mental model: authoring prim → render product → annotators → your code

```text
  USD prim (authoring)           render product (runtime)          annotators / AOVs            consumer
  ─────────────────────          ────────────────────────          ─────────────────            ────────
  Camera + OmniSensorAPI   ──►   one RTX render target per    ──►   rgb, distance_to_image_   ──► get_data() → wp.array (CUDA)
  (RtxCamera)                    camera at H×W                      plane, semantic_seg, …       writers → backend → disk
  OmniLidar  (Lidar)       ──►   lidar render product         ──►   GenericModelOutput        ──► point cloud, ROS 2 PointCloud2
  tick rate (Hz) decides                                                                         Isaac Lab Camera (Lecture 10)
  when it renders
```

A **render product** is a camera, or other RTX sensor, bound to an output resolution. It is what actually costs GPU time and memory. The GUI viewport is itself a render product, which is one reason headless runs are cheaper. **Annotators** read named outputs (AOVs) of the render product and hand them to you. Each one you attach adds buffers and possibly an extra copy.

### 2.1 The 6.x sensor API

Isaac Sim 6.0 deprecated `isaacsim.sensors.camera`, `isaacsim.sensors.rtx` (the 5.x lidar/radar classes) and `isaacsim.sensors.physics`. The replacements split **authoring** (create or wrap the USD prim) from **runtime** (render product, annotators, `get_data()`):

| Class | Extension | Role | Data |
|---|---|---|---|
| `RtxCamera` | `isaacsim.sensors.experimental.rtx` | Authoring: USD `Camera` + `OmniSensorAPI`; `tick_rate`; optics through `.camera` (focal length, apertures, clipping) | — |
| `CameraSensor` | same | Runtime: one render product at `resolution=(H, W)` + `annotators=[...]` | `get_data("rgb")` → `(wp.array, info)` |
| `TiledCameraSensor` | same | Runtime: **many cameras packed into one tiled render product** | `get_data(a, tiled=False)` → `(N, H, W, C)`; `tiled=True` → one mosaic |
| `SingleViewDepthCameraSensor`, `StructuredLightCamera` | same | Stereo-depth simulation (disparity, noise, outliers); projected-pattern depth | depth with realistic artifacts |
| `Lidar` / `LidarSensor` | same | `Lidar.create(path, config=..., tick_rate=...)`; runtime annotator `"generic-model-output"` | `parse_generic_model_output_data(...)` → `x, y, z, numElements`, … |
| `Radar` / `RadarSensor`, `Acoustic` | same | RTX radar (needs Motion BVH) and ultrasonic | GenericModelOutput |
| `IMU` / `IMUSensor` | `isaacsim.sensors.experimental.physics` | Accelerometer + gyro on a rigid body, read every physics step | `get_data()` dict |
| `Contact` / `ContactSensor` | same | Contact-pad style force on a rigid body (PhysX or Newton, via the tensor API) | `get_data()` → `in_contact`, `force`, … |
| Physics raycast | same | Explicit per-ray origins/directions, optional time offsets (solid-state, rotating, beam curtain) | hits, distances |

CameraSensor annotators in v6.1.0: `rgb`, `rgba`, `distance_to_camera`, `distance_to_image_plane`, `normals`, `motion_vectors`, `pointcloud`, `semantic_segmentation`, `instance_segmentation`, `instance_id_segmentation`, `bounding_box_2d_tight`, `bounding_box_2d_loose`, `bounding_box_3d`. The tiled sensor supports the per-pixel ones only, with no bounding boxes or point cloud. `get_data()` returns a Warp array on the GPU, or `None` while the annotator warms up. `get_data(a, out=prealloc)` fills your own buffer. `.numpy()` is an explicit device→host copy. In NVIDIA's device-check example, `rgb` is `(H, W, 3) uint8` and `distance_to_image_plane` is `(H, W, 1) float32`.

> **You will see this in older code.** 5.x scripts use `from isaacsim.sensors.camera import Camera` with `camera.get_rgba()`, `CameraView` for batched cameras, `LidarRtx`, and `IMUSensor` from `isaacsim.sensors.physics`. ROS 2 helper nodes there use `frameSkipCount`. All of these are deprecated in 6.0, and `tick_rate` replaces `frameSkipCount`.

> **Two resolution conventions.** `CameraSensor`/`TiledCameraSensor` take `resolution=(height, width)` (NumPy/OpenCV). Replicator's `rep.create.render_product(cam, (1280, 720))` takes **(width, height)**. Mixing them up gives you a transposed image and the wrong VRAM estimate.

### 2.2 Camera intrinsics

A USD camera has a focal length \( f \) and horizontal/vertical apertures \( a_h, a_v \). Both are in tenths of a stage unit, so their ratio is unitless. With output resolution \( W \times H \) and no aperture offset:

$$
f_x = \frac{W\,f}{a_h}, \qquad f_y = \frac{H\,f}{a_v}, \qquad c_x = \frac{W}{2}, \qquad c_y = \frac{H}{2}
$$

`CameraSensor` calls `enforce_square_pixels`, so \( f_x = f_y \) once the resolution is set. Read the values with `cam.camera.get_focal_lengths()` and `cam.camera.get_apertures()`, or attach Replicator's `camera_params` annotator. RtxCamera supports OpenCV pinhole and fisheye distortion through the `OmniLensDistortionOpenCvPinholeAPI` / `...FisheyeAPI` schemas. Use those to match a calibrated real camera, so that a policy or detector sees the same projection in sim and on hardware.

### 2.3 Rates: physics, sensors, app

There are three clocks. Physics steps at its own `dt`, often several times per app update. The app updates, and in GUI mode the viewport renders on each update. Each sensor renders at its own **tick rate**. **Multi-tick rendering is on by default since 6.0.** `omni:sensor:tickRate` (the `tick_rate=` argument) set to 0 means *autotrigger*, which renders every frame. A non-zero value renders at that frequency, independent of the frame rate. That makes it the cheapest performance knob in this lecture. A 30 Hz camera in a 60 Hz loop renders on about half the frames.

Two rules from the docs:

* **For lidars, `tick_rate` must equal `omni:sensor:Core:scanRateBaseHz`.** The shipped `Example_Rotary` config is 10 Hz. A mismatch silently produces partial scans every frame.
* **Sensor timestamps follow sensor ticks, not your loop.** Physics sensors (IMU, contact, raycast) read every physics step. RTX sensors only refresh when they tick. When you fuse them, use the timestamps in the data, never the loop counter.

---

## 3. RTX lidar, radar, and physics sensors

**RTX lidar** is ray-traced on RT cores at render time. Its results land in the `GenericModelOutput` buffer. `Lidar.create(path, config="Example_Rotary" | "Example_Solid_State" | vendor name, variant=..., tick_rate=..., attributes={...})` loads a configured sensor from the asset library: Ouster, HESAI, SICK and others, listed in `SUPPORTED_LIDAR_CONFIGS`. `accumulate_outputs=True` (the default) accumulates a full scan before output. `aux_output_level` (`"NONE"` → `"FULL"`) adds per-point fields such as object IDs. Cost scales with rays per scan × scan rate and with scene complexity. Lidar returns depend on non-visual material properties, so materials matter for lidar realism, not just looks.

**RTX radar** requires **Motion BVH**, which is off by default "for performance reasons". Enable it with `SimulationApp({..., "enable_motion_bvh": True})` or the documented `--/renderer/raytracingMotion/...` flags. Turning it on costs every RTX sensor something, so measure frame time with it on and off.

**Physics sensors** are not rendered at all. They read physics state (the tensor API for contact) on every step:

* `IMUSensor(IMU.create("/World/body/imu", linear_acceleration_filter_size=10, ...))` returns linear acceleration, angular velocity and orientation (wxyz). With `read_gravity=True` it reports **specific force**, which is what a real accelerometer measures: +g at rest, 0 in free fall. The sensor must sit under a rigid body. Moving it while playing invalidates it.
* `ContactSensor(Contact.create("/World/foot/contact", min_threshold=..., max_threshold=...))` returns `in_contact`, `force` and `number_of_contacts`. `get_raw_data()` gives per-contact records (bodies, position, normal, impulse). It needs an enabled rigid-body ancestor with colliders. On Newton, each dynamic body also needs a `MassAPI` with non-zero mass.
* The **physics raycast** sensor fires explicit rays every physics step. With `rayTimeOffsets` it can sweep like a rotating lidar. It reads its attributes once when the simulation starts.

On the GPU budget these are close to free next to an RTX sensor. They cost CPU time per step and a little per-body bookkeeping. So use a contact sensor rather than a camera to detect a grasp, and an IMU rather than visual odometry to estimate tilt, whenever the policy or test allows.

---

## 4. Synthetic data with Replicator

Replicator (`omni.replicator.core`) is the SDG engine. A pipeline has five parts:

```text
  randomize (functional API or graph randomizers) → step (render, rt_subframes) → annotators → writer → backend (disk, …)
```

The 6.1 getting-started scripts use this pattern, and Lab 7c reuses it:

* **Manual capture.** `rep.orchestrator.set_capture_on_play(False)`, then `rep.orchestrator.step(rt_subframes=...)` once per frame you want, and `rep.orchestrator.wait_until_complete()` at the end, because writers are asynchronous.
* **Scene building.** The functional API: `rep.functional.create.cube/plane/camera/dome_light(...)`, `rep.functional.modify.position(prim, xyz)`, and labels through `semantics={"class": "box"}` or `rep.functional.modify.semantics(...)`. Annotators such as segmentation and boxes only see labelled prims.
* **Graph randomizers.** These fire on events: `with rep.trigger.on_custom_event(event_name="x"): rep.create.light(..., color=rep.distribution.uniform(...))`, then `rep.utils.send_og_event(event_name="x")`.
* **Writer and backend.** `backend = rep.backends.get("DiskBackend"); backend.initialize(output_dir=...)`, then `writer = rep.writers.get("BasicWriter"); writer.initialize(backend=backend, rgb=True, semantic_segmentation=True, bounding_box_2d_tight=True); writer.attach(rp)`.
* **Reproducibility.** `rep.set_global_seed(...)` plus your own `random.seed(...)`.
* **Quality knobs.** `rt_subframes` renders the same frame several times. It removes ghosting after teleports and gives materials time to load, and it multiplies render time. The examples set DLSS to Quality (`/rtx/post/dlss/execMode` = 2) for SDG.

Three related tools:

* **`isaacsim.replicator.experimental.domain_randomization`** replaces the deprecated `isaacsim.replicator.domain_randomization`. It randomizes *physics* in cloned environments: gravity, masses, forces, joint states on reset. It uses `dr.physics_view.register_*` plus `dr.trigger.on_rl_frame` / `dr.gate.on_interval` / `dr.gate.on_env_reset`. That is visual and physical DR for RL, and Isaac Lab's EventManager (Lecture 09) does the same job inside Lab.
* **`CosmosWriter`** writes the RGB, depth, segmentation, shaded segmentation and edge modalities that NVIDIA Cosmos Transfer takes as input. Sim frames become the control signal for a world model that re-renders them photorealistically. Expect this writer to be heavier on disk than BasicWriter with RGB only.
* **Recorders.** The Synthetic Data Recorder UI wraps BasicWriter for GUI capture.

---

## 5. The hardware view: the VRAM and time cost of pixels

**A VRAM model.** Each render product carries a fixed overhead \( V_{\text{rp}} \) (render targets, renderer per-view state). Each pixel then costs renderer-internal buffers \( \beta \) (G-buffer, denoiser and DLSS history; unknown, so you measure it) plus the annotator outputs \( b_a \) bytes per pixel (rgb 3, rgba 4, depth float32 4, …):

$$
V \;\approx\; V_{\text{fixed}} \;+\; \sum_{i=1}^{N_{\text{rp}}} \Big( V_{\text{rp}} + H_i W_i \big(\beta + \textstyle\sum_{a \in A_i} b_a\big) \Big)
$$

With tiling, N cameras share **one** render product, so the \( N \cdot V_{\text{rp}} \) term collapses to one \( V_{\text{rp}} \) at the tiled resolution. That is why `TiledCameraSensor` (and Isaac Lab's `Camera`, which now includes the tiled path) is the default for multi-env vision RL. One detail from the v6.1.0 source: the tile grid is `round(sqrt(N))` rows by enough columns to fit N, so N = 5 renders a 2×3 mosaic. Count rows × columns tiles in the pixel term, not N. Lab 7a fits \( V_{\text{rp}} \) and \( \beta \) for your card by regressing VRAM on (N, N·H·W).

**A time model.** Frame time is physics plus the render cost of every sensor that ticks this frame plus copies. Render cost has a per-product part (scheduling, per-view passes) and a per-pixel part, so many small separate cameras are worse than one tiled product with the same pixel count. Annotators that need extra passes (segmentation, motion vectors) add to it. `.numpy()` adds a PCIe copy of `H·W·b_a` bytes per annotator per frame. Keep data on the GPU (Warp → `wp.to_torch`, Lecture 03) when the consumer is a GPU model.

**Levers, in rough order of payoff:**

| Lever | Moves | Note |
|---|---|---|
| Headless; no viewport | VRAM, frame ms | The viewport is a render product |
| Fewer render products: tile them | \( V_{\text{rp}} \) term, per-product time | `TiledCameraSensor` / Isaac Lab `Camera` |
| Resolution | Both pixel terms | 128² has 1/18.75 the pixels of 640×480 |
| Annotators: only what you consume | \( \sum b_a \), extra passes | Depth-only policies skip RGB |
| `tick_rate` | Time (frames where the sensor ticks) | Not VRAM: buffers stay allocated |
| Disable render-product updates between captures (`rp.hydra_texture.set_updates_enabled(False)`) | SDG time | Used in NVIDIA's object-based SDG example |
| `rt_subframes`, DLSS mode, Motion BVH | Time | Quality vs speed; measure each |

Small render products have one gotcha. Below 300 px on a side, `CameraSensor` disables post anti-aliasing on its render product and logs a warning, because "RTX post-AA/DLSS requires input dimensions of at least 300 pixels". Expect that warning at 128² and 256², and expect slightly different image statistics than at 640×480.

**Disk is the SDG ceiling.** Throughput is

$$
\text{MB/s} = \text{frames/s} \times \sum_{a} \text{bytes per frame}_a
$$

Raw 640×480 RGB is 640·480·3 = 921,600 bytes per frame before PNG compression. Segmentation, depth (`.npy`, float32) and JSON labels add to it. If the writer's queue grows faster than the disk drains it, the loop rate you see is a lie until `wait_until_complete()` returns. Always report both rates.

> **8 GB budget.** On an RTX 4060, which is below NVIDIA's 16 GB minimum, start from the Lecture 01 fixed cost and treat the rest as your pixel budget. Run headless, use only the annotators you consume, and tile anything above a handful of cameras. Small tiled cameras (64²-128²) for vision RL and one or two 640×480 cameras for sim-in-the-loop are realistic. Many 640×480+ cameras with segmentation, multi-camera SDG at high `rt_subframes`, or vision RL with hundreds of envs alongside a large policy are where you move to a cloud L40S or RTX PRO 6000. Lab 7d tells you where your line is. A100/H100 cannot run these sensors at all, because they lack RT cores.

---

## 6. Build it

Benchmarks extend `isaac_bench/` from Lecture 01. The SDG lab lives in `sdg_lab/`. Log `nvidia-smi` in a second terminal for every run.

### Lab 7a — Camera cost vs count, resolution, annotators, tiling

```python
# isaac_bench/camera_sweep.py
import argparse, json, subprocess, time
p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1)
p.add_argument("--res", type=int, nargs=2, default=[256, 256])      # (height, width)
p.add_argument("--annotators", nargs="+", default=["rgb"])
p.add_argument("--tiled", action="store_true")
p.add_argument("--frames", type=int, default=300)
p.add_argument("--out", default="isaac_bench/camera_results.jsonl")
args = p.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np, omni.timeline
from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera, TiledCameraSensor

def vram_mib():  # whole-GPU number (GPU 0): close other GPU apps
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"]).decode().split()[0])

DomeLight("/World/Dome").set_intensities(500)
GroundPlane("/World/ground", sizes=20.0)
for i in range(16):
    Cube(f"/World/box_{i}", sizes=0.2, positions=[[(i % 4) * 0.5 - 0.75, (i // 4) * 0.5 - 0.75, 0.1]])
for _ in range(10):
    simulation_app.update()
v_scene = vram_mib()

side = int(np.ceil(np.sqrt(args.n)))
for i in range(args.n):            # USD cameras look along -Z: at z = 2 with identity orientation they look down
    RtxCamera(f"/World/cam_{i}", translations=np.array([(i % side) * 0.3 - 1.0, (i // side) * 0.3 - 1.0, 2.0]))
res = tuple(args.res)
if args.tiled:
    sensors = [TiledCameraSensor("/World/cam_.*", resolution=res, annotators=args.annotators)]
else:
    sensors = [CameraSensor(f"/World/cam_{i}", resolution=res, annotators=args.annotators) for i in range(args.n)]

omni.timeline.get_timeline_interface().play()
for _ in range(30):                # warm-up: shaders, buffers, annotator first frames
    simulation_app.update()
v_cams = vram_mib()

def timed(fn, k):
    t = time.perf_counter()
    for _ in range(k):
        fn()
    return (time.perf_counter() - t) / k * 1e3

fetch = lambda: [s.get_data(a) for s in sensors for a in args.annotators]              # stays on the GPU
to_host = lambda: [d.numpy() for d, _ in fetch() if d is not None]                      # explicit D2H copy
ms_update = timed(simulation_app.update, args.frames)
ms_fetch, ms_host = timed(fetch, 100), timed(to_host, 50)
shape = [list(d.shape) for d, _ in fetch()[:1] if d is not None]
row = {"n": args.n, "res_hw": res, "annotators": args.annotators, "tiled": args.tiled,
       "vram_scene_mib": v_scene, "vram_cams_mib": v_cams, "vram_per_cam_mib": (v_cams - v_scene) / args.n,
       "update_ms": ms_update, "fetch_ms": ms_fetch, "to_host_ms": ms_host, "first_shape": shape}
print(json.dumps(row)); open(args.out, "a").write(json.dumps(row) + "\n")
simulation_app.close()
```

Sweep with one process per configuration, so every VRAM reading starts clean:

```bash
for res in "128 128" "256 256" "480 640"; do
  for n in 1 2 4 8 16; do
    python isaac_bench/camera_sweep.py --n $n --res $res
    python isaac_bench/camera_sweep.py --n $n --res $res --tiled
  done
  python isaac_bench/camera_sweep.py --n 4 --res $res --annotators rgb distance_to_image_plane semantic_segmentation
done
```

Then fit the §5 model. Regress `vram_cams_mib - vram_scene_mib` on `n` and `n·H·W` separately for tiled and untiled runs. The slope on `n` estimates \( V_{\text{rp}} \), and the slope on pixels estimates \( \beta + \sum b_a \). Expect the untiled `n` slope to be much larger than the tiled one. If it is not, check that you are reading the tiled sensor with `tiled=False` and that warm-up finished (`get_data` returned non-`None`).

### Lab 7b — RTX lidar: points per scan and frame-time spikes

```python
# isaac_bench/lidar_bench.py  (scene: copy the light/ground/boxes block from camera_sweep.py)
import numpy as np, time
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data

SimulationManager.setup_simulation(dt=1 / 60)
lidar = Lidar.create("/World/lidar", config="Example_Rotary", tick_rate=10.0,     # == scanRateBaseHz (10)
                     translations=np.array([0.0, 0.0, 1.0]))
sensor = LidarSensor(lidar, annotators=["generic-model-output"])
# play, warm up, then for 600 frames: time each simulation_app.update() and call
#   data, info = sensor.get_data("generic-model-output")
#   if data is not None: n = parse_generic_model_output_data(data).numElements
# print info.keys() once to see which timestamp your version reports, and count distinct scans with it.
```

Report p50 and p99 frame ms with and without the lidar (`--no-lidar`), points per scan, and scans per second. At 10 Hz in a 60 Hz loop, the scan's render cost should land on a subset of frames. That shows up in p99, not the mean, and p99 is what breaks a real-time factor contract in Lecture 08. Then try `tick_rate=20` without changing `scanRateBaseHz` and watch the point count per scan drop: that is the partial-scan failure the docs warn about. Swap in a vendor config such as an Ouster `OS1` variant and compare points per scan and frame cost.

### Lab 7c — A randomized dataset with Replicator, with throughput

```python
# sdg_lab/random_boxes.py
import argparse, os, random, time
p = argparse.ArgumentParser()
p.add_argument("--frames", type=int, default=200)
p.add_argument("--wh", type=int, nargs=2, default=[640, 480])      # Replicator: (width, height)
p.add_argument("--subframes", type=int, default=-1)                # -1 = default; >0 renders extra subframes
p.add_argument("--out", default="sdg_lab/_out")
args = p.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb.settings, omni.replicator.core as rep
import isaacsim.core.experimental.utils.stage as stage_utils

stage_utils.create_new_stage()
rep.orchestrator.set_capture_on_play(False)
random.seed(0); rep.set_global_seed(0)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
rep.functional.create.xform(name="World")
rep.functional.create.plane(position=(0, 0, 0), scale=(5, 5, 1), semantics={"class": "floor"})
boxes = [rep.functional.create.cube(position=(0, 0, 0.1), scale=0.2, semantics={"class": "box"},
                                    parent="/World", name=f"Box_{i}") for i in range(8)]
with rep.trigger.on_custom_event(event_name="randomize_light"):
    rep.create.light(light_type="Dome", color=rep.distribution.uniform((0.2, 0.2, 0.2), (1, 1, 1)))
cam = rep.functional.create.camera(position=(3, 3, 3), look_at=(0, 0, 0), parent="/World", name="Camera")
rp = rep.create.render_product(cam, tuple(args.wh))

backend = rep.backends.get("DiskBackend"); backend.initialize(output_dir=os.path.abspath(args.out))
writer = rep.writers.get("BasicWriter")
writer.initialize(backend=backend, rgb=True, semantic_segmentation=True, bounding_box_2d_tight=True)
writer.attach(rp)

t0 = time.perf_counter()
for i in range(args.frames):
    for b in boxes:
        rep.functional.modify.position(b, (random.uniform(-1, 1), random.uniform(-1, 1), 0.1))
    if i % 10 == 0:
        rep.utils.send_og_event(event_name="randomize_light")
    rep.orchestrator.step(rt_subframes=args.subframes)
t_loop = time.perf_counter() - t0
rep.orchestrator.wait_until_complete()                             # writers are asynchronous
t_all = time.perf_counter() - t0
mb = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(args.out) for f in fs) / 1e6
print(f"frames={args.frames} loop_fps={args.frames / t_loop:.1f} end_to_end_fps={args.frames / t_all:.1f} "
      f"MB={mb:.1f} MB/s={mb / t_all:.1f} MB/frame={mb / args.frames:.2f}")
writer.detach(); rp.destroy()
simulation_app.close()
```

Run it at `--subframes -1, 4, 16` and at two resolutions. Open a few outputs and check that the boxes are labelled in the segmentation and the 2D boxes line up. Compare loop fps with end-to-end fps. If they diverge, your disk (or PNG encoding on the CPU) is the bottleneck, not the GPU. Measure your disk's sequential write speed (`dd` or `fio`) and put it in the report next to MB/s.

### Lab 7d — How many cameras fit on 8 GB?

Use `camera_sweep.py` as the probe. For each resolution (128², 256², 640×480) and each mode (untiled, tiled), with `rgb` + `distance_to_image_plane`, double `--n` until the run fails or `vram_cams_mib` exceeds your safety line. A sensible line is total VRAM minus about 10% headroom for the desktop and allocator spikes. Then bisect between the last pass and the first fail. Run each probe in a fresh process: an out-of-memory failure can leave the app in a bad state. Record whether failure was a clean error, a crash, or a sudden frame-time cliff (spilling, swapping, thrashing). Compare the measured maximum with what your fitted model from Lab 7a predicts. A gap of more than one doubling means the model is missing a term. Find it.

---

## 7. Use it in the real stack

* **ROS 2.** Camera and RTX lidar publishers (Lecture 08) consume the same render products. `tick_rate` sets their publish rate, and DDS image bandwidth is the next bottleneck after VRAM. The standalone examples include `camera_ros.py` and `isaacsim.ros2.bridge/rtx_lidar.py`.
* **Isaac Lab cameras (Lecture 10).** In 3.0, `TiledCamera` is a deprecated alias, and `Camera` includes the tiled optimizations. Your Lab 7a per-pixel and per-product numbers predict how many `Isaac-Cartpole-Camera` envs fit.
* **Perception training data.** The scene-based and object-based SDG workflows in the 6.1 Replicator tutorials scale Lab 7c up: distractors, physics-based placement, multiple cameras, render products disabled between captures. CosmosWriter feeds Cosmos Transfer when you need photoreal variation beyond what randomization gives you.
* **Sim-to-real.** Visual DR (lights, textures, camera pose) plus matched intrinsics and distortion are the visual half of [Deep RL Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md)'s domain randomization. Physics DR comes from `isaacsim.replicator.experimental.domain_randomization` or Isaac Lab events.

---

## 8. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **VRAM per camera** (untiled vs tiled) at each resolution | Lab 7a slope | The number that sizes vision RL and multi-camera rigs |
| \( V_{\text{rp}} \) and per-pixel bytes | Lab 7a regression | Lets you predict new configurations before running them |
| **Frame ms** vs camera count, per annotator set | Lab 7a `update_ms` | Real-time factor for sim-in-the-loop |
| **Fetch ms vs to-host ms** | Lab 7a | Cost of `.numpy()`; argues for staying on the GPU |
| **Lidar p50/p99 frame ms, points/scan** | Lab 7b | Spikes break real-time contracts |
| **SDG loop fps, end-to-end fps, MB/s, MB/frame** | Lab 7c | GPU-bound vs disk-bound |
| **Max cameras on your GPU** at 128², 256², 640×480 | Lab 7d | The headline number for your card |

---

## 9. Ship it

Commit to `isaac_bench/` and `sdg_lab/`:

* `camera_sweep.py`, `run_camera_sweep.sh`, `camera_results.jsonl`, and `camera_vram.png` (VRAM vs N per resolution, tiled vs untiled) with the fitted model drawn on top
* `lidar_bench.py` and a p50/p99 table (no lidar, `Example_Rotary`, one vendor config)
* `random_boxes.py`, a 20-frame sample of the dataset, and a throughput table (resolution × `rt_subframes` → loop fps, end-to-end fps, MB/s)
* `SENSORS.md` with your fitted \( V_{\text{rp}} \) and per-pixel cost, the max-camera table from Lab 7d, and one paragraph on which job (sim-in-the-loop, SDG, vision RL) your card can and cannot do

---

## Exit criteria

You can move on when you can:

* create a camera and a lidar with the 6.x API, read their data on the GPU, and explain every annotator you attached
* set sensor tick rates deliberately, and explain the lidar `tick_rate` = `scanRateBaseHz` rule
* predict the VRAM of a new camera configuration from your fitted model to within a factor you have measured
* produce a labelled, randomized dataset and say whether its generation is GPU-bound or disk-bound
* state your GPU's maximum camera count at three resolutions, tiled and untiled

---

## Self-check

1. A teammate needs 64 cameras at 128² for vision RL and creates 64 `CameraSensor`s. VRAM blows up long before the pixel count suggests it should. Explain the term in the cost model they are paying 64 times, and the fix.
2. Your 640×480 depth camera feeds a PyTorch policy on the same GPU, and profiling shows a large chunk of each step in `.numpy()`. What is happening on the bus, and how do you keep the data on the GPU?
3. An Ouster lidar is publishing scans with far fewer points than its datasheet, and nothing is logged. Which two attributes do you compare first, and why does the renderer not complain?
4. Your SDG loop reports a high frame rate, but the run takes much longer than frames ÷ fps, and most of that time is after the loop. What is the bottleneck, and which two numbers prove it?
5. To detect whether a gripper holds a cube, you can use a wrist camera with segmentation or a contact sensor on each finger. Compare their GPU cost, update rate, and failure modes.
6. A `CameraSensor` built with `resolution=(640, 480)` produces images you expected to be landscape but that come out portrait, and your Replicator render product with `(640, 480)` looks right. What happened?

---

## References

* Camera sensors (RtxCamera, CameraSensor, TiledCameraSensor, intrinsics, distortion, tick rate) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_camera.html)
* RTX lidar and RTX radar (Motion BVH requirement) — [lidar](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_rtx_lidar.html) · [radar](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_rtx_radar.html)
* Multi-tick rendering (per-sensor tick rates, lidar tick-rate rule, the three clocks) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_multitick_rendering.html)
* Physics sensors — [IMU](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_physics_imu.html) · [contact](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_physics_contact.html) · [raycast](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_physics_raycast.html)
* Replicator in Isaac Sim — [overview](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/replicator_tutorials/tutorial_replicator_overview.html) · [getting started scripts](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/replicator_tutorials/tutorial_replicator_getting_started.html) · [useful snippets (`step()`, `rt_subframes`)](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html) · [troubleshooting](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/replicator_tutorials/troubleshooting.html)
* Cosmos SDG (CosmosWriter) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/replicator_tutorials/tutorial_replicator_cosmos.html)
* v6.1.0 standalone examples used here — [`isaacsim.sensors.experimental.rtx/`](https://github.com/isaac-sim/IsaacSim/tree/v6.1.0/source/standalone_examples/api/isaacsim.sensors.experimental.rtx) · [`sdg_getting_started_03.py`](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_03.py) · [`randomization_demo.py`](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/standalone_examples/api/isaacsim.replicator.experimental.domain_randomization/randomization_demo.py) · [`camera_sensor.py` (annotator list, post-AA floor)](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/extensions/isaacsim.sensors.experimental.rtx/python/impl/camera_sensor.py)
* Isaac Lab 3.0 camera (`TiledCamera` alias) — [source](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py)
* Isaac Sim 6.1 requirements (RT cores required; A100/H100 unsupported) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)

---

## Next in this special course

* Next: [Lecture 08 — ROS 2 and OmniGraph](Lecture-08.md)
* Previous: [Lecture 06 — Robots: Import, Tune, and Move](Lecture-06.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
