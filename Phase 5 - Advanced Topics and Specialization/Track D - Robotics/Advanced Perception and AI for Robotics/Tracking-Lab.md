# From a pixel to a track: preserve time, frame and uncertainty

A detector can be accurate in image coordinates while the robot's object estimate is wrong. The missing steps are geometry, capture-time transforms, uncertainty, prediction and association. This lab works through those steps with small numbers before adding a neural detector.

Allow **90–120 minutes**. The main path uses Python 3.10+ and its standard library. An optional check executes the actual ROS 2 Jazzy TF2 buffer. Start with [camera geometry](../Robotics%20Foundations/Lecture-04.md) if focal length, transforms or variance are unfamiliar. Run commands from this lesson directory.

## 1. Recover a stationary object's position from a moving camera

Use a planar robot moving forward at 1 m/s. Its camera is aligned with the base, at the same origin. A fixed object is at world `(5, 1)` metres. The camera captures an image at **t=0.2 s**, but inference finishes at **t=0.4 s**.

At capture, the camera has moved 0.2 m. The object is 4.8 m forward and 1 m left of it. With horizontal focal length `fx=400 pixels` and principal point `cx=320 pixels`, its horizontal image coordinate is:

\[
u=320+400\frac{-1}{4.8}\approx236.667\text{ pixels}.
\]

Why the minus sign? Optical x points right, while body y points left. In this planar fixture, optical z is forward, so `(x_body, y_body)=(depth, -(u-cx) depth/fx)`. Vertical position, roll, pitch, distortion and a separate camera mounting transform are omitted deliberately. A real system needs those calibrated quantities. [ROS coordinate conventions](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst).

The supplied `depth=4.8 m` is **axial optical depth**, not Euclidean distance along the ray. A depth estimator, range sensor and stereo disparity output need different conversions; do not assume a field named “depth” already has the required semantics.

```bash
python3 labs/tracking.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

[The recorded output](labs/tracking-reference.json) recovers world `(5,1)` using the capture-time camera position. Deliberately relabeling the same pixels as captured at 0.4 s gives `(5.2,1)`: **20 cm of false object motion**. It also makes the reported age falsely zero. More accurate neural inference cannot repair the wrong timestamp.

The implementation interpolates a bracketed planar pose history, using the short angular arc across ±π. It rejects missing history, future/stale detections, nonfinite values and nonpositive depth/focal length. Exactly half a turn between two orientation samples is ambiguous in this model; a faster sample rate or an additional motion assumption is needed. Interpolation itself is an approximation to motion between samples.

## 2. See the same time distinction in actual TF2

If Jazzy's `tf2_ros`, `geometry_msgs` and `rclpy` packages are installed:

```bash
source /opt/ros/jazzy/setup.bash
python3 labs/tf2_history.py --output my-tf2-run.json
```

This script inserts two generated transforms directly into the actual TF2 buffer: camera x=0 at t=10 s and x=1 at t=11 s. It checks:

| Query | Expected and recorded result |
| --- | --- |
| world ← camera at 10.2 s | Translation +0.2 m |
| world ← camera at time zero | Latest available translation +1 m |
| camera ← world at 10.2 s | Translation −0.2 m |
| Query t=9 s | Past extrapolation rejected |
| Query t=12 s | Future extrapolation rejected |
| Query an absent frame | Missing-frame exception |

[All six checks executed](labs/tf2-reference.json). This is real TF2 library behavior with direct buffer insertion; it does not exercise `/tf` transport, a camera driver or synchronization between clocks. Time zero is a special “latest available” request. Use the image's original timestamp when transforming its geometry; replacing it with latest is a different question. [Official TF2 time tutorial](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Tutorials/Intermediate/Tf2/Learning-About-Tf2-And-Time-Cpp.rst).

## 3. Carry uncertainty through the geometry

Let `r=(u-cx)/fx`, yaw be θ, and camera position be `(tx,ty)`. The planar world point is

\[
x=t_x+z(\cos\theta+r\sin\theta),\quad
y=t_y+z(\sin\theta-r\cos\theta).
\]

Small independent errors propagate through the Jacobian: `Σ_world ≈ J Σ_input Jᵀ`. The code includes pixel, depth, x/y translation and yaw uncertainty. Each derivative tells you how one error moves the world point. For example,

\[
\frac{\partial x}{\partial u}=\frac{z\sin\theta}{f_x},\qquad
\frac{\partial y}{\partial u}=-\frac{z\cos\theta}{f_x}.
\]

When the camera faces along world x, horizontal pixel error mainly changes world y. Rotate it 90° and the same pixel error mainly changes world x. The tests check this rotation, covariance symmetry and known values. The covariance entries are in **m²**, not metres.

**Variation:** double depth while holding pixel uncertainty constant. The pixel-induced standard deviation doubles and its variance quadruples. Now double yaw uncertainty and inspect the result. A detector confidence score is not a substitute for either calibrated variance.

This first-order calculation assumes small errors and independent inputs. Multiple detections can share the same uncertain camera pose; fusing them as independent world measurements can become overconfident. A full estimator must account for shared errors, nonlinearities and correlations. This fixture does not estimate those quantities from data.

## 4. Predict position before matching detections

Track one world axis with state `[position, velocity]`. For interval Δt,

\[
F=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix},\qquad
P^-=FPF^T+Q,
\]

where the supplied continuous white-acceleration model uses

\[
Q=q\begin{bmatrix}\Delta t^3/3&\Delta t^2/2\\\Delta t^2/2&\Delta t\end{bmatrix}.
\]

Here q is a continuous acceleration-noise spectral density, with units m²/s³. It is not the variance of a single constant acceleration drawn for the whole step; that alternative produces different powers and coefficients.

Start at position 0 m, velocity 1 m/s, covariance `diag(0.25,0.25)` and q=0.2. After one second:

```text
predicted position = 1 m
predicted velocity = 1 m/s
P = [[0.5666667, 0.35],
     [0.35,      0.45]]
```

A measurement of 1.2 m with variance 0.04 m² gives innovation 0.2 m and innovation variance `S=0.5666667+0.04`. The gains are `K_position≈0.9341` and `K_velocity≈0.5769`. The corrected state is approximately **1.1868 m, 1.1154 m/s**. Position information changes velocity because prediction has created cross covariance.

The implementation uses the Joseph covariance update and tests positive semidefiniteness over a trace. Negative time, duplicate measurement timestamps and invalid covariance are rejected. Prediction and measurement updates check their numerical results before mutating the state. This is a known-answer Gaussian estimator, not a learned tracker. For a full derivation and extensions, see Särkkä's [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf).

## 5. Gate outliers and keep object identities separate

The scalar normalized innovation is

\[
d^2=\frac{(z-\hat x)^2}{P^-_{xx}+R}.
\]

The fixture rejects updates with `d²>9`. For a correctly specified one-dimensional Gaussian innovation, this corresponds to an absolute innovation beyond three standard deviations. A multi-dimensional gate has a different chi-squared distribution; do not reuse the same threshold without stating dimension and coverage. A bad noise model can make this gate misleading.

For two targets, predict both to the same capture timestamp, calculate all gated costs, and assign each detection to at most one track. The executable enumerates tiny assignments, first maximizing the number of gated matches and then minimizing summed d². It supports at most six tracks/detections and is for inspection, not fleet-scale throughput or a complete probabilistic data-association objective. Production systems also model missed detections, clutter, births, deletion and appearance.

A worked crossing:

| Track | Position at t=0 | Velocity | Prediction at t=1.2 |
| --- | --- | --- | --- |
| A | 0 m | +1 m/s | 1.2 m |
| B | 2 m | −1 m/s | 0.8 m |

The detections arrive in order `[0.8,1.2]`. Correct prediction assigns A→second and B→first. Matching to the old positions instead swaps identities in this fixture. At exact overlap or through long occlusion, geometry alone may not distinguish identities; a deterministic tie-break is not additional evidence.

## 6. Transfer the result to a ROS detector

Publish the original capture stamp, a meaningful `frame_id`, named coordinate/size conventions and the appropriate detection type. `message_filters` can pair messages by timestamps within a chosen rule; it does not synchronize unsynchronized hardware clocks or turn sequential exposures into simultaneous ones. Query TF at capture time, retain transform provenance, and define how missing transforms, stale output and lost tracks affect the consuming task.

A bounding box alone does not establish a grasp pose, free space or object identity. Keep the image-quality, tracking-identity and task-success evaluations separate. [ByteTrack's paper](https://arxiv.org/abs/2110.06864) is a useful next step for association using high- and low-confidence detections; running this small Kalman exercise is not an implementation or benchmark of ByteTrack.

**Completion:** explain the 20 cm timing error, reproduce the Kalman update by hand, run the **16 portable tests**, and retain one rejected outlier and one ambiguous-identity example. Mark the optional TF2 run separately. No trained model, live sensor, SLAM system, 3D pose estimator or robot task was executed here.

**Self-check answers:** A receive timestamp cannot recover when pixels were exposed. Doubling an input standard deviation quadruples its variance contribution. A rejected outlier must not change the state. Two visually identical objects at the same predicted position need information beyond this one-dimensional motion model to establish identity.

Return to the [perception survey](Lecture-01.md), or continue to [multi-robot coordination](../Multi-Robot%20Systems%20and%20Swarm%20Robotics/Coordination-Lab.md).
