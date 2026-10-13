# Lecture 06 — Geometry, robot state, and control contracts

Make frame transforms, orientation conventions, timing, and action semantics explicit before any learned output reaches a robot.

**Prerequisites:** [Lecture 05](Lecture-05.md), matrices, trigonometry, rigid-body basics. **Effort:** 30 minutes setup, two hours lab, 90 minutes analysis. **Hardware:** CPU laptop; optional simulator visualization. **Stack:** state estimation, control, middleware, model integration. **Roles:** robotics systems engineer, simulation engineer, deployment engineer.

[Course](README.md) · [Previous](Lecture-05.md) · [Next](Lecture-07.md)

## Geometry is part of the model interface

A VLA returns six numbers and a gripper command. Without a convention, those six numbers might mean a position and Euler angles, a displacement and rotation vector, a spatial velocity, or a target pose in some normalized space. These interpretations can all fit the same array shape.

The misconception is that small actions make convention errors harmless. A 2 cm motion in the wrong frame can move into an object, and repeatedly applying a small absolute target as a delta creates a large trajectory error.

ROS REP-103 defines standard SI units, right-handed coordinate conventions, and a camera optical-frame convention distinct from the common body frame. Use its explicit conventions when integrating with ROS; do not infer conventions from an axis color in a visualization. [Official REP-103 source](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)

REP-105 distinguishes mobile-platform frames such as `map`, `odom`, and `base_link`. In particular, locally continuous odometry and globally corrected localization have different properties. A target expressed in a globally corrected frame may jump when localization updates. [Official REP-105 source](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst)

## Rigid transforms with an unambiguous notation

Let \(T_{B\leftarrow C}\) map coordinates expressed in camera frame \(C\) into base frame \(B\):

$$
T_{B\leftarrow C}=\begin{bmatrix}R_{B\leftarrow C}&t_{B\leftarrow C}\\0&1\end{bmatrix},
\qquad p_B=R_{B\leftarrow C}p_C+t_{B\leftarrow C}.
$$

A point uses rotation and translation. A free displacement vector uses only rotation: \(\Delta p_B=R_{B\leftarrow C}\Delta p_C\). Accidentally adding camera translation to every predicted displacement creates a constant, often enormous, motion bias.

Transforms compose by matching adjacent frame labels:

$$
T_{B\leftarrow E}=T_{B\leftarrow C}T_{C\leftarrow E}.
$$

The inverse is not obtained by negating every number:

$$
T_{C\leftarrow B}=\begin{bmatrix}R^\top&-R^\top t\\0&1\end{bmatrix}.
$$

Check \(R^\top R=I\) and \(\det R=1\) within numerical tolerance. A matrix with determinant \(-1\) includes a reflection and is not a proper rigid-body rotation.

## Worked example: camera motion is not base motion

For an educational camera rotated 90 degrees about the base z-axis, choose

$$
R=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix},
\qquad t=\begin{bmatrix}0.4\\0.1\\0.6\end{bmatrix}\ \mathrm m.
$$

A point \(p_C=(0.2,0,0.5)^\top\) maps to \(p_B=(0.4,0.3,1.1)^\top\) m. A camera-frame displacement \(\Delta p_C=(0.02,0,0)^\top\) maps to \((0,0.02,0)^\top\) m. Adding \(t\) to that displacement would produce \((0.4,0.12,0.6)^\top\), an obvious contract violation.

This matrix is a deliberately simple fixture, not a prescribed mounting transform. Real calibration must estimate the actual camera pose and quantify uncertainty. A transform can be mathematically valid and physically wrong.

## Orientations require more than three labels

Euler angles require an axis order and intrinsic/extrinsic convention. Quaternions require component ordering, normalization, and a defined multiplication convention. Rotation vectors encode an axis times an angle; they are not interchangeable with Euler angles just because both have three components.

For unit quaternions \(q\) and \(-q\), the physical orientation is the same. An ordinary componentwise difference can therefore report a large error for identical rotations. A useful orientation discrepancy is

$$
\theta_{\mathrm{err}}=2\arccos\left(\min(1,|q_1^\top q_2|)\right),
$$

after normalization. Clamp for floating-point roundoff. Reject a near-zero quaternion instead of dividing by its norm and producing nonfinite values.

Local and global rotation updates also differ. For a current orientation \(R\), a body-frame increment commonly composes as \(R\exp([\delta\phi]_{\times})\), while a world-frame increment composes on the left. In general these products are unequal. The action contract must specify which update the controller applies.

## Robot state and the meaning of a target

Joint state includes position \(q\), velocity \(\dot q\), and sometimes torque or estimated effort. A Cartesian target requires forward/inverse kinematics, joint limits, and an end-effector definition. “Tool center point” and “last link origin” can differ by several centimeters.

An absolute target remains fixed if sent repeatedly. A delta target accumulates. A velocity target depends on how long it is held. A normalized action requires its decoder before any of these meanings emerge. For grippers, define whether a scalar means opening width, normalized position, velocity, force, or a binary command and specify the sign.

A compliant controller may intentionally allow position error to regulate contact. Evaluating it only by positional tracking error can reward excessively stiff behavior. Separate policy target error, controller tracking error, and task outcome.

## Time belongs in the transform chain

A wrist camera moves with the robot. Its transform at image exposure time can differ from its transform when inference completes. Combining an old image with the newest joint state without compensation produces a geometric inconsistency.

If the tool translates at 0.15 m/s and image/state synchronization differs by 40 ms, the translational mismatch can be 6 mm before any visual prediction error. Angular motion can make the error larger for distant points. This is a first-order analytical estimate, not a calibration measurement.

Record sensor timestamps on a known clock basis and define interpolation limits. Extrapolating transforms indefinitely hides dropped observations. A stale-data check should happen before transformation and again before execution when computation takes appreciable time.

## Lab: a frame and action conformance suite

**Exercise / pseudocode; no robot motion is performed.** Implement rigid transform application, inversion, composition, and displacement rotation using double precision. Use the worked matrix as a known fixture. Add random valid rotations and verify round trips within a declared tolerance, such as \(10^{-9}\) for the small synthetic examples.

Build a command decoder with separate entry points for absolute pose, relative pose, and velocity. Apply the same numerical vector twice and show the resulting trajectories differ. Keep the timestep explicit for velocity integration.

Required negative cases are a millimeter value mislabeled as meters, a quaternion with swapped component ordering, a zero quaternion, an improper rotation, a stale camera transform, and translation accidentally added to a displacement. Some errors cannot be inferred from values alone: the unit swap must be caught by schema provenance or known fixtures, not a claim that every plausible number is self-describing.

For guided variation, attach a 0.1 m tool offset and recompute the target pose. For independent transfer, convert a bimanual policy whose two action vectors use different tool frames into a common base frame. Explain which transforms must be synchronized and which can remain static.

Measure round-trip error, orientation discrepancy, timestamp skew, accepted/rejected commands, and worst-case decoder runtime. Stop after the fixtures and 1,000 randomized algebra checks. Deliver code, a transform diagram, schema definitions, and a report distinguishing numerical correctness from calibration correctness.

CPU algebra cannot validate real kinematics, contact, collision avoidance, or the stability of a controller. If adding simulator visualization, keep those results separate from physical calibration evidence.

## Self-check and exit criteria

1. **Why omit translation for a displacement?** A displacement is a difference of points; the shared frame-origin translation cancels.
2. **Why can valid quaternions still yield wrong motion?** Their component ordering, frame, multiplication side, or physical calibration may be wrong.
3. **What makes a transform time-dependent?** Motion between its frames, such as a wrist-mounted camera relative to the base.

Exit when a reviewer can reconstruct every commanded pose from named frames, timestamps, units, and decoder rules without guessing.

[Previous](Lecture-05.md) · [Next: encoders and multimodal fusion](Lecture-07.md)
