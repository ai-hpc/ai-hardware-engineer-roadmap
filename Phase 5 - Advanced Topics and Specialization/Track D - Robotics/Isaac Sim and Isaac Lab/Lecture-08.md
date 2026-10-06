# Lecture 08: ROS 2 and OmniGraph

## Overview

Up to now the simulator has been a closed box: your Python script stepped it and read its state. A real robot software stack does not work like that. Nav2, MoveIt 2, perception nodes, and `ros2_control` controllers expect a robot that publishes `/clock`, `/tf`, `/joint_states` and sensor topics, and accepts commands, over DDS, at a steady rate. This lecture makes Isaac Sim 6.1 that robot.

Two things make it hard. First, the integration surface changed in 6.x: OmniGraph publisher nodes no longer read prims themselves, sensor publish rates moved onto the sensor prim, and 6.1 added an in-process `ros2_control` Controller Manager. Second, a ROS-connected simulator has a **contract with wall-clock time**. Every camera you add costs render time, every image costs DDS bandwidth, and if the frame takes longer than its share of real time, the real-time factor falls below 1 and every node that assumed real time misbehaves.

By the end you should be able to:

* explain how `isaacsim.ros2.bridge` loads ROS 2 libraries (bundled vs system), and when custom interfaces need Python 3.12
* build ROS 2 action graphs in Python with verified 6.1 node types: clock, TF, joint states, joint commands, cameras
* use `/clock` and `use_sim_time` correctly, and choose between simulation and system timestamps
* drive an arm from an external ROS 2 node, and host `ros2_control` inside the simulator with `isaacsim.ros2.control`
* measure real-time factor, topic bandwidth, and stamp-to-receive latency, and predict how they move with sensor count and resolution

---

## 1. Why it matters: the simulator as a ROS 2 robot

There are four ways to connect Isaac Sim 6.1 to ROS 2. They are not interchangeable:

| Style | What runs where | Use it for | Cost |
|---|---|---|---|
| **OmniGraph bridge nodes** (`isaacsim.ros2.bridge`) | Publishers and subscribers are graph nodes ticked by the simulation | Standard topics: clock, TF, joint states, odometry, cameras, lidar | Per-tick graph evaluation plus serialization |
| **rclpy inside the standalone script** | Your Python publishes using the bundled or system `rclpy` | Custom timing, asynchronous image publishing (`camera_rclpy_async.py` example) | Your code owns threading and copies |
| **In-process `ros2_control`** (`isaacsim.ros2.control`, new in 6.1) | A real Controller Manager runs inside Isaac Sim, driven by the physics step | Using the same controller YAML as the real robot; MoveIt 2 through standard controllers | Controllers share the sim process |
| **Simulation control services** (`isaacsim.ros2.sim_control`) | ROS 2 services and actions from `simulation_interfaces` | Play/pause/step, spawn/delete entities, load worlds from tests and CI | Service round trips |

The first is the workhorse and the subject of most of this lecture. The prerequisite is the [Advanced Robot Operating System lecture](../Advanced%20Robot%20Operating%20System/Lecture-01.md): nodes, topics, QoS, TF, and launch files are assumed.

---

## 2. Mental model

### 2.1 The extensions

| Extension (6.1) | Role |
|---|---|
| `isaacsim.ros2.bridge` | The one you enable. Brings up the ROS 2 runtime and the OmniGraph nodes. |
| `isaacsim.ros2.core` / `isaacsim.ros2.nodes` | In 6.1 the node implementations live here. Their **type names still start with `isaacsim.ros2.bridge.`**, so graphs and scripts written against the bridge keep working. |
| `isaacsim.ros2.control` | In-process `ros2_control` Controller Manager (new in 6.1) |
| `isaacsim.ros2.sim_control` | `simulation_interfaces` services and actions |
| `isaacsim.ros2.tf_viewer`, `.urdf`, `.ui` | TF drawing in the viewport; URDF over ROS topics; the *Tools > Robotics > ROS 2 OmniGraphs* shortcut menu |

Enable it in a standalone script with `isaacsim.core.experimental.utils.app.enable_extension("isaacsim.ros2.bridge")`, after `SimulationApp` and before you build graphs. That is what the shipped 6.1 standalone examples do.

### 2.2 Which ROS 2 gets loaded

Isaac Sim runs on Python 3.12. Its ROS 2 bridge needs ROS libraries that match that Python, which is why the docs list distros per OS:

| Host | Distro | Notes |
|---|---|---|
| Ubuntu 24.04 | **Jazzy** (recommended) | Default system install. Jazzy on 24.04 already uses Python 3.12. |
| Ubuntu 22.04 | Humble | System Humble is Python 3.10. Custom interfaces need a **Python 3.12 build** of your workspace (Dockerfiles in `IsaacSim-ros_workspaces`, `./build_ros.sh -d humble -v 22.04`). |
| Windows 11 | Jazzy via Pixi (Zenoh RMW) | WSL Humble is deprecated |

There are two ways the libraries get into the process:

* **Bundled libraries.** Isaac Sim ships a minimal internal ROS 2. The workstation `python.sh` launcher configures it automatically when `ROS_DISTRO` is unset: Humble on 22.04, Jazzy on 24.04. You don't need to set `ROS_DISTRO`, `RMW_IMPLEMENTATION`, or `LD_LIBRARY_PATH` for the standard messages. Pass `./python.sh --no-ros-env script.py` to opt out.
* **System or workspace install.** Source it *before* launching, and the launcher keeps your environment. You need this for custom messages (`rclpy` imports your generated Python modules into the 3.12 runtime). On 22.04, don't source the Python 3.10 Humble install in Isaac Sim's terminal. If you installed with pip instead of the zip, read *Configuring Options and Enabling Internal ROS Libraries* on the install page; the pip package exposes a `ros2` extra (`isaacsim[all,extscache,ros2]`, as used in the docs' Pixi instructions) for the ROS components.

**Middleware.** The bridge uses Fast DDS when `RMW_IMPLEMENTATION` is unset. Cyclone DDS (`export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`) and, on Jazzy, Zenoh are supported alternatives. If you go across machines, set `FASTRTPS_DEFAULT_PROFILES_FILE` to the workspace's `fastdds.xml` in *every* terminal, Isaac Sim's included. `ROS_DOMAIN_ID` isolates you from other people's robots on the same network.

### 2.3 Action graphs: ticks, data, context

An **action graph** is an OmniGraph evaluated by the `execution` evaluator. Execution pins (`execIn` / `execOut` / `tick`) say *when* a node runs. Data pins say *what* it consumes. Three node kinds appear in every ROS graph:

* **A tick source.** `omni.graph.action.OnPlaybackTick` fires every simulation frame while playing. `omni.graph.action.OnImpulseEvent` fires only when you set its `state:enableImpulse`, which gives a standalone script exact control over publish times.
* **`isaacsim.ros2.bridge.ROS2Context`**, which owns the DDS context. It has `inputs:domain_id` and `inputs:useDomainIDEnvVar`, so it either reads `ROS_DOMAIN_ID` or pins a domain.
* **Data sources → publishers.** These are the 6.1 node types used in this lecture, read from the `.ogn` definitions at tag v6.1.0:

| Node type | UI name | Key pins |
|---|---|---|
| `isaacsim.core.nodes.IsaacReadSimulationTime` | Isaac Read Simulation Time | out `simulationTime`; in `resetOnStop` |
| `isaacsim.core.nodes.IsaacReadSystemTime` | Isaac Read System Time | out `systemTime` |
| `isaacsim.ros2.bridge.ROS2PublishClock` | ROS2 Publish Clock | in `timeStamp`, `topicName`, `context` |
| `isaacsim.sensors.physics.IsaacReadJointState` | Isaac Read Joint State | in `prim`; out `jointNames`, `jointPositions`, `jointVelocities`, `jointEfforts`, `jointDofTypes`, `stageMetersPerUnit`, `sensorTime` |
| `isaacsim.ros2.bridge.ROS2PublishJointState` | ROS2 Publish Joint State | the same arrays in, plus `timeStamp` |
| `isaacsim.ros2.bridge.ROS2SubscribeJointState` | ROS2 Subscribe Joint State | out `jointNames`, `positionCommand`, `velocityCommand`, `effortCommand` |
| `isaacsim.core.nodes.IsaacArticulationController` | Articulation Controller | in `robotPath` or `targetPrim`, command arrays |
| `isaacsim.core.nodes.IsaacComputeTransformTree` | Isaac Compute Transform Tree | in `targetPrims`, `parentPrim`; out `parentFrames`, `childFrames`, `translations`, `orientations` |
| `isaacsim.ros2.bridge.ROS2PublishTransformTree` | ROS2 Publish Transform Tree | the four frame arrays in, plus `timeStamp`, `staticPublisher` |
| `isaacsim.core.nodes.IsaacCreateRenderProduct` | Isaac Create Render Product | in `cameraPrim`, `width`, `height`; out `renderProductPath` |
| `isaacsim.ros2.bridge.ROS2CameraHelper` | ROS2 Camera Helper | in `renderProductPath`, `type` (`rgb`, `depth`, `rgb_h264`, `rgb_hevc`, ...), `topicName`, `frameId`, `useSystemTime` |

All of these work on both PhysX and Newton (6.0 release notes). The Camera Helper hides a whole post-processing network: when it first runs, it builds a session-only graph under `/Render/PostProcessing/SDGPipeline` that pulls the annotator from the renderer and feeds the publisher. Each Camera Helper publishes one data type, so RGB plus depth needs two helpers on the same render product.

### 2.4 Time

ROS 2 nodes started with `use_sim_time:=true` subscribe to `/clock` and use it for `now()`, timers and TF lookups. That gives you three rules:

1. **Publish `/clock` from simulation time** (`IsaacReadSimulationTime → ROS2PublishClock`) in every scene that ROS nodes consume.
2. **Stamp every message from the same source as `/clock`.** If the clock comes from sim time but a camera is stamped with system time, TF lookups in RViz fail or interpolate garbage. The `resetOnStop` input on the time node decides whether the clock restarts from zero after Stop. Set it deliberately, because a clock that jumps backwards confuses TF buffers.
3. **Use system time only for measurement.** `IsaacReadSystemTime`, or `useSystemTime=True` on the Camera Helper, stamps with the host clock. That makes `receive_time − header.stamp` a real latency on one machine. Lab 8a and 8b do this, then switch back.

### 2.5 What changed in 6.x (old graphs break here)

* **Compute nodes now feed publishers.** `ROS2PublishTransformTree` used to take `targetPrims` and `ROS2PublishJointState` used to take `targetPrim`. Both prim inputs are **deprecated in 6.0** and scheduled for removal. Now `IsaacComputeTransformTree` (an articulation root expands to its whole link tree) and `IsaacReadJointState` compute the arrays, and you wire them in. Odometry TF follows the same pattern.
* **Sensor rate lives on the sensor.** `frameSkipCount` on the Camera, Camera Info and RTX Lidar helpers is deprecated. Leave it at 0 and set `omni:sensor:tickRate` (Hz) on the sensor prim, which needs `OmniSensorAPI` applied (`prim.ApplyAPI("OmniSensorAPI")`). RTX Radar does not honor `tickRate`.
* **TF aggregation.** By default, compatible TF publishers are merged into one `TFMessage` per sim timestamp per topic. Downstream listeners never see half a tree, and Isaac Sim needs fewer DDS publishers. A side effect is that the newest stamp can run one tick ahead of the last published batch. To debug individual publishers, disable it with `--/exts/isaacsim.ros2.nodes/tfAggregation/enabled=false`.
* **CameraInfo** now reads distortion from OpenCV lens schemas on the camera prim. Without one, it publishes pinhole intrinsics with zero distortion.

> **Quaternion conventions.** ROS messages and `IsaacComputeTransformTree.outputs:orientations` are **(x, y, z, w)** (per the node's `.ogn`). Isaac Sim's Core Experimental prim API (`get_world_poses`) returns **(w, x, y, z)**. Isaac Lab 3.0 is xyzw. If you hand-build a TF from `XformPrim` poses, reorder the quaternion first.

You will see the old form in older code: graphs wiring `targetPrims` straight into `ROS2PublishTransformTree`, `frameSkipCount` for rates, and the `omni.isaac.ros2_bridge` names from before 4.5. Saved USD graphs from 5.x need the migration in the 6.0 guides.

---

## 3. The hardware view: real-time factor, bandwidth, latency

**Real-time factor** is the contract:

$$
\text{RTF} = \frac{\Delta t_{\text{sim}}}{\Delta t_{\text{wall}}}
$$

With one physics step and one render per loop iteration, RTF \( \approx \Delta t_{\text{physics}} / t_{\text{frame}} \). That is the physics timestep divided by the measured frame time from Lecture 01. At a 1/60 s step, a frame has to finish in about 16.7 ms to hold RTF ≥ 1. RTX cameras are what usually break it, because each render product adds render and annotator time (Lecture 07).

Two consequences surprise people:

* **Standalone scripts are not throttled.** In standalone Python (`isaacsim.exp.base.kit`), `/app/runLoops/main/rateLimitEnabled` defaults to false. A light scene then runs *faster* than real time. That is fine for nodes on `use_sim_time`, but wrong for anything on wall time, including a human driving with a joystick. The 6.1 publish-rate tutorial sets physics and loop rates together with `SimulationManager.setup_simulation(dt=...)` and `RenderingManager.set_dt(...)`, and turns rate limiting on when real time is wanted.
* **Publish rate is capped by RTF.** The docs give

$$
\text{publish rate} = \min(\text{RTF} \cdot \text{sensor tick rate},\ \text{app fps})
$$

  A 30 Hz camera in a scene running at RTF 0.5 publishes about 15 Hz of wall time. Measured against `/clock` it is still 30 Hz of sim time.

**Bandwidth** of raw images is arithmetic. Here \( W \cdot H \) is pixels, \( b \) bytes per pixel, and \( f \) the publish rate:

$$
B = W \cdot H \cdot b \cdot f
$$

| Stream | Bytes/frame | At 30 Hz |
|---|---|---|
| 320×240 rgb8 (3 B/px) | 0.23 MB | 6.9 MB/s |
| 640×480 rgb8 | 0.92 MB | 27.6 MB/s |
| 1280×720 rgb8 | 2.76 MB | 82.9 MB/s |
| 640×480 depth, float32 (4 B/px) | 1.23 MB | 36.9 MB/s |
| Franka `/joint_states` (9 joints) | well under 1 KB | negligible |

Two raw 720p cameras at 30 Hz already exceed gigabit Ethernet (about 125 MB/s before protocol overhead). Every raw image also costs a **GPU→host copy** plus serialization on the CPU. The Camera Helper's `rgb_h264` / `rgb_hevc` types use the GPU's hardware video encoder (NVENC on GeForce cards, a block separate from the SMs). Each message is a complete IDR frame, which trades encoder work and some quality for a large bandwidth cut. Subscribers need the `isaac_compressed_image_decoder` node from the docs (or their own decoder).

**Latency** has several parts: wait for the tick, render and annotate (cameras only), GPU→host copy, serialize, DDS transport, deserialize, and your callback. Measure the total with system-time stamps and `receive − stamp` on the same host (clocks agree). For cross-host measurement, synchronize clocks (PTP or chrony) or measure round trip.

> **8 GB budget.** The ROS bridge itself is cheap; the sensors behind it are not. On a 4060, use one robot, CPU physics (the shipped 6.1 ROS 2 examples call `setup_simulation(..., device="cpu")` for single-robot scenes, which leaves the GPU to RTX), and one or two cameras at 640×480 or below with `tickRate` set to what the consumer needs, not the frame rate. Run headless and watch RViz from the ROS side; the Isaac Sim viewport is a render product too. Drop depth if you only need RGB, and use `rgb_h264` when images leave the machine. Running Nav2 with several RTX lidars and cameras, or a perception stack that wants four 720p streams at 30 Hz with RTF ≥ 1, is when to move to a cloud L40S / RTX PRO 6000 (RT cores and NVENC needed; A100/H100 cannot render). Remember the 4060 is below the official 16 GB minimum.

---

## 4. Build it

Work in `ros_lab/`. Every Isaac-side script uses the Lecture 01 standalone pattern. Every ROS-side script runs with system Jazzy (`source /opt/ros/jazzy/setup.bash`) in a separate terminal. Keep `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1` logging in a third terminal.

### Lab 8a — Clock, TF and joint states for an arm, commanded from ROS 2

The Franka asset path and the graph wiring below follow the 6.1 `isaacsim.ros2.bridge/moveit.py` standalone example. The TF branch follows the 6.0 OmniGraph migration guide.

```python
# ros_lab/arm_bridge.py
import argparse, time
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--hz", type=float, default=60.0)                     # physics and loop rate
parser.add_argument("--stamp", choices=["sim", "system"], default="sim")  # header.stamp source
parser.add_argument("--realtime", action="store_true")                    # throttle loop to wall clock
parser.add_argument("--seconds", type=float, default=60.0)
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import carb, omni.graph.core as og, usdrt.Sdf
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path

app_utils.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
stage_utils.set_stage_units(meters_per_unit=1.0)

ROBOT = "/Franka"
root = get_assets_root_path()
stage_utils.add_reference_to_stage(root + "/Isaac/Environments/Grid/default_environment.usd", "/World/ground")
stage_utils.add_reference_to_stage(
    root + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda", ROBOT)
simulation_app.update()

stamp_node, stamp_out = (("isaacsim.core.nodes.IsaacReadSystemTime", "outputs:systemTime")
                         if args.stamp == "system" else
                         ("isaacsim.core.nodes.IsaacReadSimulationTime", "outputs:simulationTime"))
JS = ("jointNames", "jointPositions", "jointVelocities", "jointEfforts",
      "jointDofTypes", "stageMetersPerUnit")   # sensorTime left unwired: when > 0 it overrides timeStamp
CMD = ("jointNames", "positionCommand", "velocityCommand", "effortCommand")
TF = ("parentFrames", "childFrames", "translations", "orientations")

K = og.Controller.Keys
og.Controller.edit(
    {"graph_path": "/ROS", "evaluator_name": "execution"},
    {
        K.CREATE_NODES: [
            ("Tick", "omni.graph.action.OnPlaybackTick"),
            ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),    # always drives /clock
            ("Stamp", stamp_node),                                          # drives header.stamp
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("ReadJS", "isaacsim.sensors.physics.IsaacReadJointState"),
            ("PubJS", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubJS", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("Ctrl", "isaacsim.core.nodes.IsaacArticulationController"),
            ("TF", "isaacsim.core.nodes.IsaacComputeTransformTree"),
            ("PubTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
        ],
        K.CONNECT: [
            ("Tick.outputs:tick", "Clock.inputs:execIn"),
            ("SimTime.outputs:simulationTime", "Clock.inputs:timeStamp"),
            ("Tick.outputs:tick", "ReadJS.inputs:execIn"),
            ("ReadJS.outputs:execOut", "PubJS.inputs:execIn"),
            *[(f"ReadJS.outputs:{a}", f"PubJS.inputs:{a}") for a in JS],
            (f"Stamp.{stamp_out}", "PubJS.inputs:timeStamp"),
            ("Tick.outputs:tick", "SubJS.inputs:execIn"),
            ("Tick.outputs:tick", "Ctrl.inputs:execIn"),
            *[(f"SubJS.outputs:{a}", f"Ctrl.inputs:{a}") for a in CMD],
            ("Tick.outputs:tick", "TF.inputs:execIn"),
            ("TF.outputs:execOut", "PubTF.inputs:execIn"),
            *[(f"TF.outputs:{a}", f"PubTF.inputs:{a}") for a in TF],
            (f"Stamp.{stamp_out}", "PubTF.inputs:timeStamp"),
            *[("Context.outputs:context", f"{n}.inputs:context") for n in ("Clock", "PubJS", "SubJS", "PubTF")],
        ],
        K.SET_VALUES: [
            ("ReadJS.inputs:prim", [usdrt.Sdf.Path(ROBOT)]),
            ("Ctrl.inputs:robotPath", ROBOT),
            ("TF.inputs:targetPrims", [usdrt.Sdf.Path(ROBOT)]),
            ("PubJS.inputs:topicName", "joint_states"),
            ("SubJS.inputs:topicName", "joint_command"),
        ],
    },
)

SimulationManager.setup_simulation(dt=1.0 / args.hz, device="cpu")  # as the 6.1 ROS 2 examples do
RenderingManager.set_dt(1.0 / args.hz)                               # one physics step per loop iteration
if args.realtime:
    carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", True)
app_utils.play()
simulation_app.update()

sim_t = og.Controller.attribute("/ROS/SimTime.outputs:simulationTime")
w0, s0, frames = time.perf_counter(), sim_t.get(), 0
while simulation_app.is_running() and time.perf_counter() - w0 < args.seconds:
    simulation_app.update()
    frames += 1
wall, sim = time.perf_counter() - w0, sim_t.get() - s0
print(f"hz={args.hz} realtime={args.realtime} frames={frames} wall={wall:.1f}s sim={sim:.1f}s "
      f"RTF={sim / wall:.2f} frame_ms={wall / frames * 1e3:.2f}")
app_utils.stop()
simulation_app.close()
```

The ROS 2 side learns the joint names from the first `/joint_states` message (don't hardcode them; asset versions rename joints) and commands a slow sinusoid on two arm joints:

```python
# ros_lab/arm_commander.py   (system Jazzy: python3 arm_commander.py [--ros-args -p use_sim_time:=true -p stamp_is_system:=true])
import math, time
import numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class Commander(Node):
    def __init__(self):
        super().__init__("arm_commander")
        self.declare_parameter("stamp_is_system", False)
        self.names, self.home, self.lat = None, None, []
        self.pub = self.create_publisher(JointState, "joint_command", 10)
        self.create_subscription(JointState, "joint_states", self.on_state, 10)
        self.create_timer(0.02, self.on_timer)        # 50 Hz on this node's clock: sim time if use_sim_time

    def on_state(self, msg):
        if self.names is None:                        # arm joints only; command fingers separately
            keep = [i for i, n in enumerate(msg.name) if "finger" not in n]
            self.names = [msg.name[i] for i in keep]
            self.home = np.array([msg.position[i] for i in keep])
        if self.get_parameter("stamp_is_system").value:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.lat.append((time.time() - stamp) * 1e3)
            if len(self.lat) >= 500:
                p50, p95, p99 = np.percentile(self.lat, [50, 95, 99])
                self.get_logger().info(f"joint_states latency ms p50={p50:.2f} p95={p95:.2f} p99={p99:.2f}")
                self.lat.clear()

    def on_timer(self):
        if self.names is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        q = self.home.copy()
        q[0] += 0.4 * math.sin(0.5 * t)
        q[3] += 0.3 * math.sin(0.7 * t)
        self.pub.publish(JointState(name=self.names, position=q.tolist()))

rclpy.init()
rclpy.spin(Commander())
```

Steps:

1. `python ros_lab/arm_bridge.py --headless` (or `./python.sh` on the zip install). From ROS: `ros2 topic list` should show `/clock`, `/tf`, `/joint_states`, `/joint_command`. Check `ros2 topic hz /joint_states` and `ros2 run tf2_tools view_frames`. If the TF tree holds only one frame, point `targetPrims` at the prim that has `ArticulationRootAPI` (find it with your Lecture 02 inspector).
2. Run `arm_commander.py` and watch the arm move in RViz (RobotModel display plus TF). Then restart it with `use_sim_time:=true`, stop the simulation, and notice the timer stops too.
3. Run unthrottled, then with `--realtime`, and compare the printed RTF with `ros2 topic hz /clock`. In a light scene the unthrottled RTF exceeds 1.
4. Run with `--stamp system` and `stamp_is_system:=true` to get stamp-to-receive latency percentiles for joint states. Then switch back to sim stamps. Never leave a mixed-time setup in a scene RViz or Nav2 consumes.

### Lab 8b — Camera bandwidth and latency vs resolution

This lab follows the 6.1 `camera_periodic.py` pattern: a push graph on demand, evaluated once to build each camera's SDG publishing pipeline. Rate is set with `omni:sensor:tickRate` as in the 6.0 sensor-graph migration guide, and stamps use system time for measurement.

```python
# ros_lab/camera_bw.py
import argparse, time
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--rate", type=float, default=30.0)     # omni:sensor:tickRate, Hz
parser.add_argument("--type", default="rgb")                # rgb | depth | rgb_h264 | rgb_hevc
parser.add_argument("--seconds", type=float, default=30.0)
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.graph.core as og, omni.usd, usdrt.Sdf
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

app_utils.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
stage_utils.set_stage_units(meters_per_unit=1.0)
stage_utils.add_reference_to_stage(
    get_assets_root_path() + "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd", "/background")
stage = omni.usd.get_context().get_stage()

nodes, conns, vals = [("OnTick", "omni.graph.action.OnTick")], [], []
for i in range(args.n):
    path = f"/Cam_{i}"
    cam = UsdGeom.Camera(stage.DefinePrim(path, "Camera"))
    xf = UsdGeom.XformCommonAPI(cam)
    xf.SetTranslate(Gf.Vec3d(-1.0 + i, 5.0, 1.0))
    xf.SetRotate((90, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    prim = cam.GetPrim()
    prim.ApplyAPI("OmniSensorAPI")                               # needed for omni:sensor:tickRate
    prim.GetAttribute("omni:sensor:tickRate").Set(args.rate)
    rp, ch = f"RP{i}", f"Cam{i}"
    nodes += [(rp, "isaacsim.core.nodes.IsaacCreateRenderProduct"), (ch, "isaacsim.ros2.bridge.ROS2CameraHelper")]
    conns += [("OnTick.outputs:tick", f"{rp}.inputs:execIn"), (f"{rp}.outputs:execOut", f"{ch}.inputs:execIn"),
              (f"{rp}.outputs:renderProductPath", f"{ch}.inputs:renderProductPath")]
    vals += [(f"{rp}.inputs:cameraPrim", [usdrt.Sdf.Path(path)]), (f"{rp}.inputs:width", args.width),
             (f"{rp}.inputs:height", args.height), (f"{ch}.inputs:type", args.type),
             (f"{ch}.inputs:topicName", f"cam{i}/{args.type}"), (f"{ch}.inputs:frameId", f"cam{i}"),
             (f"{ch}.inputs:useSystemTime", True)]

K = og.Controller.Keys
graph, _, _, _ = og.Controller.edit(
    {"graph_path": "/ROS_Cameras", "evaluator_name": "push",
     "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND},
    {K.CREATE_NODES: nodes, K.CONNECT: conns, K.SET_VALUES: vals})
og.Controller.evaluate_sync(graph)                               # builds the SDG publishing pipelines once

og.Controller.edit({"graph_path": "/ROS_Clock", "evaluator_name": "execution"}, {
    K.CREATE_NODES: [("Tick", "omni.graph.action.OnPlaybackTick"),
                     ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                     ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock")],
    K.CONNECT: [("Tick.outputs:tick", "Clock.inputs:execIn"),
                ("SimTime.outputs:simulationTime", "Clock.inputs:timeStamp")]})

SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
RenderingManager.set_dt(1.0 / 60.0)
app_utils.play()
simulation_app.update()
t0, frames = time.perf_counter(), 0
while simulation_app.is_running() and time.perf_counter() - t0 < args.seconds:
    simulation_app.update()
    frames += 1
print(f"n={args.n} {args.width}x{args.height} {args.type} frame_ms={(time.perf_counter() - t0) / frames * 1e3:.2f}")
app_utils.stop()
simulation_app.close()
```

One ROS-side probe serves Labs 8b and 8c. In `image` mode it measures rate, bandwidth and stamp-to-receive latency. In `clock` mode it measures RTF the way ROS nodes experience it, as `/clock` against the wall clock:

```python
# ros_lab/ros_probe.py   (python3 ros_probe.py image /cam0/rgb   |   python3 ros_probe.py clock)
import sys, time
import numpy as np, rclpy
from rclpy.qos import qos_profile_sensor_data            # best effort: compatible with either publisher QoS
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CompressedImage, Image

rclpy.init()
node = rclpy.create_node("ros_probe")
st = {"t0": time.time(), "lat": [], "bytes": 0, "s0": None}
sec = lambda t: t.sec + t.nanosec * 1e-9

def on_image(msg):
    now = time.time()
    st["lat"].append((now - sec(msg.header.stamp)) * 1e3)   # valid with useSystemTime=True, same host
    st["bytes"] += len(msg.data)
    if now - st["t0"] >= 5.0:
        p50, p95, p99 = np.percentile(st["lat"], [50, 95, 99])
        print(f"{len(st['lat']) / (now - st['t0']):.1f} Hz  {st['bytes'] / (now - st['t0']) / 1e6:.2f} MB/s  "
              f"latency ms p50={p50:.1f} p95={p95:.1f} p99={p99:.1f}")
        st.update(t0=now, lat=[], bytes=0)

def on_clock(msg):
    now, s = time.time(), sec(msg.clock)
    if st["s0"] is None:
        st.update(t0=now, s0=s)
    elif now - st["t0"] >= 5.0:
        print(f"RTF={(s - st['s0']) / (now - st['t0']):.3f}")
        st.update(t0=now, s0=s)

if sys.argv[1] == "clock":
    node.create_subscription(Clock, "/clock", on_clock, qos_profile_sensor_data)
else:
    topic = sys.argv[2]
    msg_type = CompressedImage if ("h264" in topic or "hevc" in topic) else Image
    node.create_subscription(msg_type, topic, on_image, qos_profile_sensor_data)
rclpy.spin(node)
```

Sweep one camera over 320×240, 640×480 and 1280×720 at 30 Hz, for `rgb`, `depth` and `rgb_h264`. Record Hz, MB/s, latency percentiles, frame ms, and VRAM. Compare MB/s with the arithmetic table in §3. A big shortfall means the publisher is not keeping up (check Hz) or DDS is dropping messages (best-effort subscriber, large messages). Cross-check with `ros2 topic bw` and `ros2 topic hz`.

### Lab 8c — Real-time factor with N sensors

Run `camera_bw.py` for `--n ∈ {0, 1, 2, 4}` at 640×480 RGB (go past 4 only if VRAM allows), once unthrottled and once with rate limiting on (add the `--realtime` switch from Lab 8a). Plot RTF against camera count. Mark the count where throttled RTF drops below 1: that is how many cameras your 4060 can feed in real time. Check that each camera's wall-clock publish Hz follows the \( \min(\text{RTF} \cdot \text{tickRate}, \text{fps}) \) rule.

### Lab 8d (optional) — `ros2_control` inside the simulator

1. Smoke test, no workspace needed: `./python.sh standalone_examples/api/isaacsim.ros2.control/ros2_control_smoke_test.py --headless` should print `DEMO SMOKE TEST: PASS`.
2. Install `ros-jazzy-ros2-controllers ros-jazzy-ur-moveit-config`, build `isaac_ros2_control_demo` in `IsaacSim-ros_workspaces/jazzy_ws`, and run `ur10_ros2_control_demo.py`. Then `ros2 launch isaac_ros2_control_demo ur10_in_process.launch.py`, and check `ros2 control list_controllers` (expect `joint_state_broadcaster` and `scaled_joint_trajectory_controller` active).
3. Command the arm without MoveIt by publishing a `trajectory_msgs/JointTrajectory` to `/scaled_joint_trajectory_controller/joint_trajectory`, as in the tutorial.

How it maps: the graph is `OnPlaybackTick → isaacsim.ros2.control.ROS2ControlManager` (inputs `targetPrim`, `controllerConfig`, `urdfPath`, `namespace`, `publishRobotDescription`, `useSimTime`), plus a `/clock` publisher. On the first tick after Play, the node builds a URDF from the USD drives. A drive with stiffness becomes a position interface, damping only becomes velocity, neither gain becomes effort, and no drive means state only. It then starts the Controller Manager. The read/update/write loop runs from the physics step at the YAML `update_rate`, **clamped to the physics rate**. Stop tears the Controller Manager down; Play rebuilds it. Try `update_rate: 1000` with a 60 Hz physics step and find the clamp warning in the log.

---

## 5. Use it in the real stack

* **MoveIt 2 over topics.** The *Franka MoveIt* example and `standalone_examples/api/isaacsim.ros2.bridge/moveit.py` publish `isaac_joint_states` and subscribe `isaac_joint_commands`; `ros2 launch isaac_moveit isaac_moveit.launch.xml` brings up MoveIt and RViz. The alternative is the Lab 8d `ros2_control` path, which keeps the controller next to physics and uses the same YAML as hardware.
* **Nav2.** The Nova Carter sample publishes `/tf`, `/odom`, `/map` (generated with the Occupancy Map extension) and `/point_cloud`, with `pointcloud_to_laserscan` producing `/scan`. Navigation is the heaviest standard ROS workload you will run (RTX lidar plus often cameras), so it is where RTF breaks first on 8 GB.
* **Isaac ROS and Jetson.** Isaac ROS is a set of ROS 2 packages built for Jetson. The Isaac Sim docs pair Isaac ROS 4.6 with Isaac Sim 6.0/6.0.1, and 4.0–4.5 with 5.0/5.1; check the compatibility page before pairing with 6.1. The integration is the ROS 2 bridge plus Isaac ROS nodes on a Jetson on the same DDS domain. Treat image bandwidth (§3) as the first constraint and use compressed images across the link.
* **Tests and CI.** `isaacsim.ros2.sim_control` exposes `simulation_interfaces` services: `/set_simulation_state`, `/step_simulation`, `/reset_simulation`, `/spawn_entities`, `/get_entity_state`, `/load_world`, and the `/simulate_steps` action. Install `ros-jazzy-simulation-interfaces` and enable it with `--/isaac/startup/ros_sim_control_extension=True`. Stepping by service makes a ROS integration test deterministic in sim time no matter how slow the CI GPU is.
* **Policies.** The docs' *Running a Reinforcement Learning Policy through ROS 2 and Isaac Sim* tutorial is the template for the capstone (Lecture 13): the policy is a ROS node, and the simulator is the robot.

---

## 6. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **RTF (throttled / unthrottled)** | `ros_probe.py clock`; `arm_bridge.py` print | The real-time contract; whether the sim can run ahead for testing |
| **Frame ms vs sensors** | `camera_bw.py` print | Explains RTF; feeds the camera budget |
| **Topic rate (wall)** | `ros_probe.py image`, `ros2 topic hz` | Checks the \( \min(\text{RTF} \cdot \text{tickRate}, \text{fps}) \) rule |
| **Bandwidth MB/s** | `ros_probe.py image`, `ros2 topic bw` | Network and serialization budget; raw vs `rgb_h264` |
| **Stamp→receive latency p50/p95/p99** | System-time stamps, same host | Perception and control loop delay; tails matter more than medians |
| **VRAM per camera** | `nvidia-smi` slope across Lab 8c | How many sensors fit beside the scene |
| **Controller update rate** | `ros2 control` + log clamp warning | Whether `ros2_control` runs at the rate the YAML claims |

---

## 7. Ship it

Commit `ros_lab/` with:

* `arm_bridge.py`, `arm_commander.py`, `camera_bw.py`, `ros_probe.py`
* `frames.pdf` from `view_frames`, plus a screenshot of RViz showing the moving arm with TF
* `camera_sweep.csv` — resolution × type × (Hz, MB/s, latency p50/p95/p99, frame ms, VRAM)
* `rtf_vs_sensors.png` — RTF against camera count, throttled and unthrottled
* `ROS_BRIDGE.md` — your ROS distro and RMW, how the libraries were loaded, the max cameras at RTF ≥ 1 on your GPU, and one paragraph on when you would switch to `rgb_h264` or to `ros2_control`

---

## Exit criteria

You can move on when you can:

* get Isaac Sim 6.1 talking to system Jazzy, and explain bundled vs system ROS libraries and the Python 3.12 rule
* write a ROS 2 action graph in Python with the 6.1 node types, including the compute-node → publisher pattern for TF and joint states
* explain `/clock`, `use_sim_time`, `resetOnStop`, and why stamps must share one time source
* predict a camera topic's bandwidth from resolution and rate, and its wall-clock rate from RTF
* report RTF, bandwidth, and latency percentiles for your scene, and say which sensor to cut first when RTF < 1

---

## Self-check

1. A 5.1 graph with `ROS2PublishTransformTree.targetPrims` set loads in 6.1 with deprecation warnings, and `/tf` is missing the robot links. What do you add, and which outputs do you wire to which inputs?
2. RViz shows "Message removed because it is too old" for camera frames, but TF looks fine. The clock comes from `IsaacReadSimulationTime` and the Camera Helper has `useSystemTime=True` left over from a latency test. Explain the failure and the fix.
3. Your scene runs at RTF 0.6 with two 640×480 RGB cameras whose `tickRate` is 30 Hz. What rate does `ros2 topic hz` report, and what rate do nodes on `use_sim_time` experience? What two changes would most likely bring RTF to 1 on a 4060?
4. A teammate wants four 1280×720 RGB streams at 30 Hz from a workstation to a Jetson over gigabit Ethernet. Do the arithmetic, then propose a configuration that works.
5. Your `ros2_control` YAML says `update_rate: 500`, physics runs at 120 Hz, and the trajectory controller tracks worse than on the real robot. What is happening, and what are the two ways to fix it?
6. A custom message package works with system Jazzy on Ubuntu 24.04, but the same package fails to import in Isaac Sim on a 22.04 machine running Humble. Why, and what is the supported fix?

---

## References

* ROS 2 installation for Isaac Sim 6.1 (distros, Python 3.12 rule, Fast DDS / Cyclone / Zenoh) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_ros.html)
* ROS 2 bridge in the standalone workflow (`python.sh` auto-configuration, `--no-ros-env`, `OnImpulseEvent`, examples) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/bridge_configuration/tutorial_ros2_python.html)
* ROS 2 tutorial series — [clock](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/tutorial_series/tutorial_ros2_clock.html), [TF and odometry](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/tutorial_series/tutorial_ros2_tf.html), [cameras](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/tutorial_series/tutorial_ros2_camera.html), [publish rates](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/tutorial_series/tutorial_ros2_publish_rate.html), [RTF](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/tutorial_series/tutorial_ros2_rtf.html)
* 6.0 migration — [ROS 2 OmniGraph nodes](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/migration_guides/isaac_sim_6_0/ros2_omnigraph_migration.html), [sensor graphs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/migration_guides/isaac_sim_6_0/ros2_sensor_graph_migration.html)
* Robot control — [joint control](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/robot_control/tutorial_ros2_manipulation.html), [ROS 2 Control](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/robot_control/tutorial_ros2_control.html), [MoveIt 2](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/robot_control/tutorial_ros2_moveit.html), [Navigation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/robot_control/tutorial_ros2_navigation.html)
* Compressed images (H.264 / HEVC) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/advanced_sensors/tutorial_ros2_compressed_image.html); QoS — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/bridge_configuration/tutorial_ros2_qos.html)
* ROS 2 Simulation Control (`simulation_interfaces`) — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/ros2_tutorials/bridge_configuration/tutorial_ros2_simulation_control.html)
* NVIDIA Isaac ROS compatibility — [docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/nvidia_isaac_ros/isaac_ros_tutorials.html)
* Source at v6.1.0 — [`moveit.py` example](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/standalone_examples/api/isaacsim.ros2.bridge/moveit.py), [`camera_periodic.py` example](https://github.com/isaac-sim/IsaacSim/blob/v6.1.0/source/standalone_examples/api/isaacsim.ros2.bridge/camera_periodic.py), [ROS 2 node definitions](https://github.com/isaac-sim/IsaacSim/tree/v6.1.0/source/extensions/isaacsim.ros2.nodes)

---

## Next in this special course

* Next: [Lecture 09 — Isaac Lab 3.0 Architecture](Lecture-09.md)
* Previous: [Lecture 07 — Sensors, Rendering, and Synthetic Data](Lecture-07.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
