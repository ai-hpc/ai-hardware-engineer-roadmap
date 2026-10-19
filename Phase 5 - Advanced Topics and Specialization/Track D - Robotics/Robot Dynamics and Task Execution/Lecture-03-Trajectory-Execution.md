# 3. Turn a free path into motion the robot can follow

**Before:** [planning foundations](../Robotics%20Foundations/Lecture-05.md) and
[actuator dynamics](Lecture-01-Dynamics.md). **After:**
[grasp and object-level success](Lecture-04-Grasp-and-Task-Success.md).
**Build:** reject a collision between free endpoints, time a segment, and detect
a tracking error that invalidates a nominally free route.

## A path, a trajectory, and an execution are different evidence

A path describes configurations without assigning time. A trajectory specifies
configuration and motion over time. An execution is what the robot actually did.
The same path can be executed slowly, too fast for its actuators, or with enough
tracking error to hit an obstacle.

For a manipulator, collision checks must consider every link, attached object,
and relevant self-collision pair over the motion. An end-effector point is not a
complete arm model. Our exercise uses a disk translating in a 2-D map, so it can
check a swept region analytically while keeping these limits visible.

## The gap between free endpoints

A robot disk of radius `0.1 m` moves from `(-1,0)` to `(1,0)`. A circular obstacle
of radius `0.2 m` is centered at `(0,0)`. Both endpoints are clear, but the straight
motion passes through the obstacle.

Find the closest point on the segment to the obstacle center. With start `a`, end
`b`, center `c`, and direction `d=b-a`:

```text
fraction = clamp(dot(c-a, d) / dot(d,d), 0, 1)
closest = a + fraction × d
clearance = distance(closest,c) - obstacle_radius - robot_radius - margin
```

If the segment has zero length, use its single endpoint. A clearance of zero is
contact and is treated as blocked here. For the straight crossing, closest point
is `(0,0)` and clearance is `−0.3 m`.

The detour `(-1,0) -> (-1,0.6) -> (1,0.6) -> (1,0)` has segment clearances
`0.7, 0.3, 0.7 m` for the specified disk and obstacle. This is an exact geometric
calculation for these shapes, not a statement about a real map's accuracy.

**Self-check:** why clamp the fraction? **Answer:** without clamping, you find the
closest point on the infinite line. An obstacle behind the start could appear
closer than every point on the actual segment.

## Timing must satisfy the motion's derivatives

Passing a sharp corner at nonzero speed changes velocity direction instantly in
an ideal piecewise-linear path. That would require unbounded acceleration. One
simple teaching schedule stops at each waypoint and uses a cubic blend for each
straight segment:

```text
s = t / T,    0 <= s <= 1
h(s) = 3s² - 2s³
position(t) = start + h(s) × (end-start)
```

Velocity is zero at both endpoints. Differentiate the blend:

```text
h_dot = (6s - 6s²) / T
h_ddot = (6 - 12s) / T²

peak speed = 1.5 × segment_length / T
peak acceleration magnitude = 6 × segment_length / T²
```

For speed limit `v_max` and acceleration limit `a_max`, this particular cubic
therefore requires:

```text
T >= max(1.5 × length / v_max, sqrt(6 × length / a_max))
```

For a one-metre segment, `v_max=0.5 m/s`, and `a_max=1 m/s²`, choose `T=3 s`.
Peak speed is `0.5 m/s`; peak acceleration is about `0.6667 m/s²`. This is the
minimum duration for the chosen cubic under those bounds, not the globally
time-optimal trajectory among every possible profile.

The [Modern Robotics time-scaling lesson](https://modernrobotics.northwestern.edu/nu-gm-book-resource/9-1-and-9-2-point-to-point-trajectories-part-2-of-2/)
connects geometric paths to polynomial timing. A cubic's endpoint acceleration
is nonzero. Joining segments can create acceleration jumps and does not impose
a jerk bound. More demanding drives need a suitable higher-order or bounded-jerk
profile, followed by checks against joint torque and other physical limits.

## Check the interpolated motion after processing

The cubic above remains on exactly the same straight segment because `h(s)`
increases from zero to one. Its disk collision result therefore carries over
under the model. A different smoothing or interpolation operation may change the
geometric path. Recheck the resulting motion rather than assuming the original
waypoint clearance still applies.

In a production stack, distinguish planner output, time parameterization,
controller interpolation, and measured feedback. Consult the selected MoveIt
release's [time-parameterization documentation](https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html)
and controller configuration; this lab does not execute those APIs.

## Planned clearance must cover actual errors

The horizontal detour is at `y=0.6 m` with `0.3 m` geometric clearance. If the actual
reference point drifts to `y=0.25 m`, clearance becomes `0.25−0.2−0.1=−0.05 m`:
the observed motion collides despite the valid nominal path.

An inflation margin can account for a declared bound on tracking and map error,
but the bound itself needs evidence. The maximum error in a few recorded trials
is not a guarantee for every future disturbance. Correlated localization errors,
stale maps, moving obstacles, and unmodeled geometry require additional treatment.

During execution, compare the measured state with the timed reference, not just
the nearest waypoint. Record tracking error, command age, limit violations, and
progress. If execution is invalidated, the controller needs a specified stopping
or recovery trajectory with its own dynamics and collision checks. Setting
velocity to zero in a message does not make a moving physical robot stop instantly.

## Exercise and transfer

Run `python3 labs/control_lab.py motion` from this course directory. Inspect
`segment_clearance`, `cubic_duration`, and `cubic_sample` in
[task_execution.py](labs/task_execution.py).

**Guided variation:** halve the speed limit while retaining the acceleration
limit. Predict the duration before computing it. For the one-metre example,
the speed term becomes `6 s`, which dominates the unchanged acceleration term.
The geometry stays the same; timing changes.

**Independent transfer:** implement the quintic blend
`10s³−15s⁴+6s⁵`, which has zero endpoint velocity and acceleration. Differentiate
it, find its peak speed and acceleration, and select a duration under the same
limits. Compare the two profiles using equal endpoints and declared timing.
Do not compare arbitrary durations and attribute the result only to polynomial
degree.

**Exit artifact:** a free-endpoint/colliding-segment example, one hand-derived
duration, and a trace or synthetic case where actual tracking invalidates the
plan. Explain which checks would need full robot geometry and real feedback.
