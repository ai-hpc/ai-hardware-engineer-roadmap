# Module 1 — Understand a vehicle before choosing its AI

**Before:** Python, vectors, trigonometry and [robotics foundations](../../Track%20D%20-%20Robotics/Robotics%20Foundations/README.md). **First milestone:** 3–5 hours including the CPU lab. **Later depth:** perception, planning and dynamic vehicle models each need separate projects. A GPU is optional for the first milestone.

A driving system needs estimates of the world, predictions of how it may change, a feasible trajectory, and commands that the actuators can follow. A visually plausible lane overlay does not prove any of the later steps. We will start with a known geometric state, make a small controller work, then expose the timing and model-contract failures that a learned perception system introduces.

## 1. Define the quantities before the algorithm

Use a ground-fixed planar world and a rear-axle vehicle reference. In the vehicle frame x points forward and y left. The state is `(x, y, yaw, speed)` in metres, radians and metres/second. A road-wheel steering angle is `delta`; wheelbase is `L`. Steering-wheel angle, commanded motor torque, curvature and normalized simulator steering are different quantities.

| Boundary | Example output | Required interpretation |
|---|---|---|
| Perception | A lane curve, object box, or depth estimate | Coordinate frame, calibration, capture time, uncertainty |
| Prediction | Several possible future trajectories | Time samples, probabilities, agent identity |
| Planning | A selected time-parameterized path | Collision checks, limits, start time, feasibility |
| Control | Steering/acceleration requests | Units, saturation, rate limits, update age |
| Vehicle interface | Platform-specific actuator commands | Encoding, actuator dynamics, independent limits |

A path is geometry; a trajectory also specifies when positions are reached. A collision-free geometric path can still require impossible steering, acceleration or stopping behavior.

## 2. Work the bicycle model

At low slip, the rear-axle kinematic bicycle gives:

```text
x_dot = v cos(yaw)
y_dot = v sin(yaw)
yaw_dot = v tan(delta) / L
v_dot = acceleration
curvature = tan(delta) / L
```

At `L=2.8 m`, `v=5 m/s` and `delta=atan(2.8/10)≈0.273 rad`, the turn radius is 10 m and yaw rate is 0.5 rad/s. Approximate lateral acceleration is `v²/radius=2.5 m/s²`. Doubling speed while keeping steering produces four times the lateral acceleration. Geometric steering alone does not enforce tire-force limits.

The [CPU implementation](labs/vehicle_lab.py) integrates a held-angle arc and held acceleration, with no reversing. It has no slip, tire relaxation, load transfer, road bank or actuator lag. A dynamic bicycle model instead includes lateral velocity/yaw dynamics and tire forces. Use one when the experiment requires those effects; adding parameters without identification does not make the model accurate.

**Check:** with zero steering, initial speed 2 m/s, acceleration 1 m/s² and duration 2 s, displacement is 6 m and final speed 4 m/s. With braking −4 m/s² from 2 m/s, the vehicle stops after 0.5 s and 0.5 m; it must not continue backwards for the remainder of the step.

## 3. Derive pure pursuit with a signed example

Transform a lookahead target into the rear-axle frame, obtaining forward distance `x_t>0` and left distance `y_t`. The circle through that point tangent to the vehicle's heading has:

```text
curvature = 2*y_t / (x_t² + y_t²)
delta = atan(L * curvature)
```

For `(x_t,y_t)=(4,2)` m and `L=2.8`, curvature is 0.2/m and steering is `atan(0.56)≈0.510 rad` before saturation. A target to the right has negative `y_t` and produces negative steering. A point behind the vehicle is outside this forward-tracking exercise's contract.

A longer lookahead often smooths response but can cut corners; a shorter one responds more strongly and can oscillate with delay. Neither setting supplies obstacle avoidance. The original [pure-pursuit report](https://www.ri.cmu.edu/publications/implementation-of-the-pure-pursuit-path-tracking-algorithm/) explains the geometric construction.

For the lab's straight lane, use a world target 8 m forward on `y=0`, then transform it with the current yaw. Limit the requested road-wheel angle and its rate. The speed loop requests `a=clamp(v_target-v, -3, 2)` in this chosen fixture. It is proportional speed tracking, not a full adaptive-cruise controller.

## 4. Account for stopping and observation age

Under constant speed during delay `tau`, followed by constant braking magnitude `b>0` on level ground:

```text
stopping distance = v*tau + v²/(2*b)
```

At 10 m/s, 0.1 s delay and 4 m/s² braking, this is **13.5 m**, including 1 m traveled during the delay. The formula excludes brake buildup, grade, friction variation and moving obstacles. It is a useful lower-complexity calculation, not a stopping guarantee.

A constant-velocity time-to-collision baseline for bumper gap `g` is `g/(v_ego-v_lead)` only when the closing speed is positive. At 20 m gap and speeds 10 versus 6 m/s, TTC is 5 s. No positive closing speed means no finite collision time under that model; it does not mean the situation is safe under future braking or turns.

Keep capture time, inference completion, publication and application time separate. A fresh publication can contain an old image. At a 20 Hz model rate, an observation arrives every 50 ms in the ideal schedule; a 100 Hz controller may reuse it across several updates. Neither rate alone is an acquisition-to-actuation bound.

Run the [worked CPU lab](Lab-Guide.md), then the optional [CARLA frame synchronization lab](CARLA-Lab.md). The latter explicitly converts simulator coordinates and distinguishes normalized `steer` from wheel radians.

## 5. Add perception only after the state contract works

Begin with a metric baseline and a fixed split. For lane estimation, annotate the representation: image pixels, ground-plane points or polynomial coefficients in metres. An inverse-perspective transform assumes a calibrated camera and a road surface model; hills violate the flat-ground assumption. Monocular depth outputs may need a metric-scale calibration and can be ambiguous.

Compare a classical lane detector with a learned segmentation model on whole held-out drives, not neighboring video frames randomly divided between train and test. Report lateral error at named lookahead distances, missing-lane frequency and acquisition age. A segmentation IoU or detector mAP does not directly measure whether a controller can follow the road.

For obstacles, retain track identity, frame, covariance or another calibrated uncertainty representation, and time. Predict a constant-velocity baseline before a learned multimodal predictor. Score displacement error and missed modes separately. The average of “pass left” and “pass right” trajectories can pass through the obstacle; a mean prediction need not be a feasible behavior.

## 6. Progress from paths to feasible trajectories

A*/RRT can establish a geometric route under their chosen state space and collision model. Vehicle steering constraints need a suitable state representation and motion primitives, such as a lattice or hybrid search. Then assign timing and check steering, speed, acceleration, curvature and collision along the swept footprint between samples.

For MPC, define the state, control, dynamics, horizon, objective and constraints before selecting a solver. A failed optimization needs an explicit outcome; returning zero steering without reasoning is not a generic solution. Study the [executed LQR/MPC and trajectory lessons](../../Track%20D%20-%20Robotics/Robot%20Dynamics%20and%20Task%20Execution/README.md) for worked feasibility counterexamples.

**Independent transfer:** replace the straight target with a gentle arc. Establish the expected curvature and steering analytically, then test low speed, higher speed and observation delay. Report tracking, steering-rate saturation and lateral acceleration. Keep the tire model's validity as an explicit limitation.

**Ready for Module 2:** explain the first steering sign, reproduce both delay runs, distinguish a successful fallback from successful lane following, and interpret a model output's shape/units. Continue with [openpilot source ownership](../2.%20openpilot%20Reference%20Stack/Guide.md).
