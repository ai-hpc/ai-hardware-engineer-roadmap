# Lecture 03 — Capture Geometry: Head-Mounted Cameras, SLAM and Time

Trace a 3D point from a head-mounted camera to a fixed world frame, naming every transform, camera model and clock, and verify the chain before any hand track depends on it.

**Prerequisites:** [Lecture 02](Lecture-02.md) (which recordings you may use), [VLA Foundations Lecture 06](../VLA%20Foundations%20to%20Frontier%202026/Lecture-06.md) (frames, robot state and control contracts), rigid transforms from [Robotics Foundations](../Robotics%20Foundations/README.md). **Effort:** reading 1.5 hours, lab 2 hours. **Hardware:** CPU track; [`labs/ego_geometry.py`](labs/ego_geometry.py) is standard-library Python. **Roles:** robot learning data engineer, perception engineer, embodied AI research engineer.

**Misconception this lecture corrects:** "a hand point from a headset is a world coordinate you can hand to a robot." It is measured in a camera or device frame that moves every frame. Getting it into a fixed frame takes a calibration, a SLAM pose at the right time on the right clock, and the right camera model, and each of these can be wrong without raising an error.

## 1. Frames, quaternions and the naming rule

The labs write a transform as `T_a_b`: it maps a point expressed in frame `b` into frame `a`. Chains read right to left and the inner frame names cancel: \( p_{\text{world}} = T_{\text{world,device}}(t)\, T_{\text{device,camera}}\, p_{\text{camera}} \), which is `camera_point_to_world` in the lab.

| Transform | Comes from | Per frame? | Project Aria source (projectaria_tools 2.3.0) |
|---|---|---|---|
| `T_device_camera` | Device calibration | No (constant per recording in the lab) | The device's calibration (see its docs); MPS also writes `slam/online_calibration.jsonl` |
| `T_world_device(t)` | Closed-loop SLAM | Yes | `slam/closed_loop_trajectory.csv` |
| `T_odometry_device(t)` | Open-loop odometry | Yes | `slam/open_loop_trajectory.csv`, an odometry frame, **not** the world frame |
| `T_Device_Wrist(t)` | MPS hand tracking | Yes | `hand_tracking/hand_tracking_results.csv`, with 21 landmarks per hand in meters, **already in the device frame** |

The lab's camera frame has +z forward along the optical axis, +x right and +y down; take each device frame from its own documentation. Rotations are unit quaternions in (w, x, y, z) order. Other libraries store (x, y, z, w), so check each library's documentation and convert at the boundary: reading one order as the other raises no error, it just gives a different, valid rotation.

Below, the device is yawed 90° and moved 1 m along world x, and the camera sits 10 cm in front of the device (synthetic values). Two negative cases follow: a device-frame landmark pushed through a camera calibration a second time, and the yaw quaternion read in the wrong order.

**Tested here** (from the repository root, with `EGO_COURSE` set as in the [Lab guide](Lab-Guide.md)):

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import math
from ego_geometry import SE3, quat_from_axis_angle, camera_point_to_world, quat_angle_between
T_world_device = SE3(quat_from_axis_angle((0, 0, 1), math.pi / 2), (1.0, 0.0, 0.0))  # SLAM
T_device_camera = SE3((1.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0))                          # calibration
p_world = camera_point_to_world((0.2, 0.0, 0.5), T_world_device, T_device_camera)
back = (T_world_device @ T_device_camera).inverse().apply(p_world)
print("world:", tuple(round(c, 6) for c in p_world), " back to camera:", tuple(round(c, 6) for c in back))
# Negative 1: a landmark already in the DEVICE frame, treated as a camera point (tilted synthetic calibration)
T_device_camera_tilted = SE3(quat_from_axis_angle((1, 0, 0), math.radians(20)), (0.05, 0.0, 0.0))
p_device = (0.0, 0.25, 0.35)
wrong = camera_point_to_world(p_device, T_world_device, T_device_camera_tilted)
print("frame mix-up error: %.1f cm" % (100 * math.dist(T_world_device.apply(p_device), wrong)))
# Negative 2: the same yaw stored xyzw by another library, read back as wxyz
q = T_world_device.q
misread = SE3((q[1], q[2], q[3], q[0]), T_world_device.t)
print("q (wxyz):", tuple(round(c, 4) for c in q), " rotation error if misread: %.1f deg"
      % math.degrees(quat_angle_between(q, misread.q)))
print("misread world point:", tuple(round(c, 3) for c in misread.apply((0.3, 0.0, 0.5))))
EOF
```

```text
world: (1.0, 0.3, 0.5)  back to camera: (0.2, 0.0, 0.5)
frame mix-up error: 15.8 cm
q (wxyz): (0.7071, 0.0, 0.0, 0.7071)  rotation error if misread: 120.0 deg
misread world point: (0.7, 0.5, 0.0)
```

The point is 0.3 m along device x (0.2 + 0.1), the 90° yaw turns device x into world y, and the inverse returns it exactly. A device-frame landmark needs only `T_world_device(t)`; applying a camera calibration as well moved it 15.8 cm. The misread quaternion sends the same device point to (0.7, 0.5, 0.0) instead of (1.0, 0.3, 0.5).

## 2. Camera models: pinhole and fisheye

A pinhole model, \( u = f_x x / z + c_x \), cannot represent a ray at or beyond 90° off-axis (\( z \le 0 \)) and stretches rays near 90° far outside the image. Head-mounted and hand-held cameras are wide-angle (UMI's GoPro has a 155° fisheye), so their calibrations often use fisheye models. The lab implements the Kannala–Brandt equidistant family:

$$
\theta = \operatorname{atan2}\!\big(\sqrt{x^2 + y^2},\, z\big), \qquad \theta_d = \theta\,(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8), \qquad r = f\,\theta_d
$$

projectaria_tools 2.3.0 defines the models `Linear`, `Spherical`, `KannalaBrandtK3`, `Fisheye624` and `Fisheye62`. `Fisheye624` is "Spherical + polynomial radial distortion up to 11-th order + tangential distortion", plus thin-prism terms in the code, with a single focal length: **15 parameters**, ordered `[f, c_u, c_v, k0..k5, p0, p1, s0..s3]` (`Fisheye62` has 11). The lab's model is a simpler relative with four radial terms and no tangential or thin-prism terms. Use it to understand and test the mechanism; use projectaria_tools for real Aria data. EgoDex stores a 3×3 `camera/intrinsic` matrix with no distortion terms listed, and its RGB is synthesized from several cameras, so 2D reprojections may not match exactly.

**Upstream documented, not tested here:** `pip install projectaria-tools==2.3.0`

**Tested here** (synthetic intrinsics; each faulty camera unprojects the pixel of a known point):

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import math
from ego_geometry import KannalaBrandt, Pinhole
err_deg = lambda ray, p: math.degrees(math.acos(min(1.0, sum(a * b for a, b in zip(ray, p)) / math.hypot(*p))))
th = math.radians(100)                               # a ray 100 deg off the optical axis
print("pinhole:", Pinhole(600, 600, 320, 240).project((math.sin(th), 0.0, math.cos(th))),
      " fisheye u: %.1f" % KannalaBrandt(240.0, 704.0, 704.0).project((math.sin(th), 0.0, math.cos(th)))[0])
cam, p = KannalaBrandt(240.0, 704.0, 704.0), (0.4, 0.2, 0.6)
uv = cam.project(p)
for name, c, pix in (("correct", cam, uv), ("f 240 -> 300", KannalaBrandt(300.0, 704.0, 704.0), uv),
                     ("cx 704 -> 720", KannalaBrandt(240.0, 720.0, 704.0), uv),
                     ("full-res K, half-res pixel", cam, (uv[0] / 2, uv[1] / 2))):
    print(f"{name:27s} error {err_deg(c.unproject(pix), p):7.3f} deg")
wrong = KannalaBrandt(300.0, 704.0, 704.0)
print("round trip inside the wrong model: %.3f deg" % err_deg(wrong.unproject(wrong.project(p)), p))
EOF
```

```text
pinhole: None  fisheye u: 1122.9
correct                     error   0.000 deg
f 240 -> 300                error   7.340 deg
cx 704 -> 720               error   3.770 deg
full-res K, half-res pixel  error 134.838 deg
round trip inside the wrong model: 0.000 deg
```

At 0.5 m, 7.34° is about 6.4 cm of arc and 3.77° about 3.3 cm; full-resolution intrinsics on a downscaled video are wrong by far more. The last line matters most: a project-then-unproject round trip passes with the wrong camera, because the model agrees with itself. Round trips test the code. Only a comparison with ground truth (a known target, or detected points reprojected into the image) tests the calibration.

## 3. SLAM: closed loop, open loop and `graphUid`

Aria MPS writes SLAM results to `slam/` (deprecated name `trajectory/`): `closed_loop_trajectory.csv`, `open_loop_trajectory.csv`, `semidense_points.csv.gz`, `semidense_observations.csv.gz`, `online_calibration.jsonl` and `summary.json`. Each closed-loop pose carries `trackingTimestamp` (µs), `utcTimestamp` (ns), `qualityScore`, `graphUid`, `T_world_device`, `deviceLinearVelocity_device`, `angularVelocity_device` and `gravity_world` (default {0, 0, -9.81}).

* **`graphUid`.** When it is the same, "poses, velocities and point clouds are defined in the same coordinate frame". Poses with different values are in different world frames: do not concatenate, difference or compare them without an alignment.
* **Open loop.** Open-loop poses are `T_odometry_device`. Never mix them with closed-loop `T_world_device` in one chain. Keep `qualityScore` in your record, and read the MPS documentation for its meaning before filtering on it.

## 4. Time: one clock for the whole chain

projectaria_tools 2.3.0 (`TimeTypes.h`) defines `DeviceTime` (capture time on the device clock, shared by all its sensors), `RecordTime` (VRS index time, "not guaranteed which time domain"), `HostTime` (arrival time on the host, "may not be accurate"), `TimeCode`, `TicSync` and `SubGhz` (TimeSync-server domains for multi-device capture) and `Utc` ("only seconds accurate"). The header says: "We strongly recommend to always work with the device timestamp when dealing with single-device Aria data." Lookups take `TimeQueryOptions { Before, After, Closest }`.

A camera frame rarely falls exactly on a trajectory sample, so interpolate the pose at the frame time (linear translation, spherical rotation: `interpolate_pose`) instead of taking the nearest sample. The error from a time offset \( \Delta t \) is about distance × angular rate × \( \Delta t \).

**Tested here** (synthetic trajectory with an assumed head turn of 1 rad/s; DeviceTime in microseconds):

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import math
from ego_geometry import SE3, quat_from_axis_angle, interpolate_pose
pose = lambda t_us: SE3(quat_from_axis_angle((0, 0, 1), 1.0 * t_us / 1e6), (0.0, 0.0, 1.5))  # 1 rad/s yaw
t0, t1, frame = 0, 100_000, 33_333                 # trajectory samples and a camera frame, DeviceTime in us
p, truth = (0.5, 0.0, 0.0), pose(frame).apply((0.5, 0.0, 0.0))
for name, T in (("interpolated", interpolate_pose(pose(t0), pose(t1), (frame - t0) / (t1 - t0))),
                ("nearest-before", pose(t0))):
    print(f"{name:15s} error {100 * math.dist(T.apply(p), truth):.2f} cm")
EOF
```

```text
interpolated    error 0.00 cm
nearest-before  error 1.67 cm
```

That matches the estimate: 0.5 m × 1 rad/s × 0.0333 s ≈ 1.67 cm. Mixing HostTime and DeviceTime shows up as timestamps that go backwards, which the Lecture 05 audit counts.

## 5. EgoDex: a new world frame every episode

EgoDex HDF5 files store `transforms/<joint>` as N×4×4 SE(3) matrices for the camera and 68 joints (25 per hand) in the **ARKit origin frame, which changes per episode**. Positions from two episodes are not comparable, so express each episode relative to an anchor of its own (its first camera pose, for example) before pooling. The lab's `SE3` holds a quaternion, so converting a 4×4 matrix is one more boundary where you state the order. Lecture 07 adds the last link, `T_base_world`, from robot calibration.

## 6. Exercise

**Exercise / pseudocode** (CPU only):

1. **Worked example → full chain.** Run the three tested snippets and copy their outputs into your record. Add a known-point check: the camera origin must land at `(T_world_device @ T_device_camera).t`.
2. **Guided variation.** Inject one fault at a time: focal length +25%, principal point +16 px, swapped quaternion order, `T_device_camera` applied to a device-frame landmark, nearest-sample pose instead of interpolation. Record each error at 0.5 m and which of your checks catches it.
3. **Independent transfer.** Apply the chain to one recording you may use (Lecture 02): an Aria recording with `projectaria-tools==2.3.0` and its MPS outputs, or one EgoDex episode re-anchored to its first camera pose. **CPU alternative:** write synthetic `T_device_camera` and trajectory values for a rig you sketch yourself, and run the same checks.

**Deliverable:** `geometry_chain.md`, the verified camera → device → world chain: each transform as `T_a_b` with its source file, the camera model and where its parameters came from, the `graphUid` values used, the clock, the quaternion order of every source, and the check outputs. **Acceptance:** the inverse round trip and the known-point check pass within a stated tolerance (for example 1e-6 m); every transform names its source; one clock is used throughout; at least three injected faults are detected, each with its error at 0.5 m.

## Failure clinic and self-check

**Failure clinic:** intrinsics come from the wrong camera, or from a full-resolution calibration while the video is a downscaled variant. Poses from two `graphUid` values are concatenated and the hand appears to jump. Open-loop odometry poses are used as world poses. Device-frame hand landmarks go through `T_device_camera` again. Poses are looked up by HostTime. EgoDex wrist positions are pooled across episodes without re-anchoring. An xyzw quaternion is read as wxyz.

1. **Your project/unproject round trip passes, but hand points land 6 cm off. How?** A round trip with one model is self-consistent even when the intrinsics are wrong. Check against ground truth: a known target, or detected points reprojected into the image.
2. **Two MPS poses have different `graphUid` values. Can you difference them for a velocity?** No. They are in different world frames, so the difference has no physical meaning without an alignment.
3. **Why DeviceTime instead of HostTime for single-device Aria data?** The header recommends DeviceTime, which every sensor on the device shares; HostTime is arrival time on the host and "may not be accurate".
4. **MPS reports a wrist landmark. Which transforms take it to world?** Only `T_world_device(t)` at the landmark's timestamp. It is already in the device frame.
5. **Why can't you average EgoDex wrist positions across episodes directly?** Each episode has its own ARKit origin, so the same coordinates mean different places in different episodes.

[Previous: datasets and licenses](Lecture-02.md) · [Course index](README.md) · [Next: the data pipeline](Lecture-04.md)
