# 1. Frames, units, and time

**Purpose:** turn an unlabelled vector into a physical statement a controller can use. **Prerequisite:** vectors and sine/cosine. **Next:** [kinematics](Lecture-02.md).

## A coordinate is an address relative to a frame

`(1, 0)` does not identify a unique place. It means one unit along some frame's x axis, starting at that frame's origin. A **frame** is an origin and oriented axes. A **pose** describes a frame's position and orientation relative to another frame. A **point** has position but no orientation.

For a planar robot we write `T_AB = (tx, ty, yaw)`: the pose of frame B expressed in frame A. It converts a point expressed in B into a point expressed in A:

\[
p_A = R(\theta)p_B+t,\qquad
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.
\]

The subscript labels keep the direction visible. A different library may name the transform differently, so verify its API with one known point. Do not rely on an ambiguous phrase such as “camera-to-world pose.”

To read the matrix equation as ordinary arithmetic, it says
`x_A = cos(theta)*x_B - sin(theta)*y_B + tx` and
`y_A = sin(theta)*x_B + cos(theta)*y_B + ty`. At zero angle, sine is zero and
cosine is one, so only translation remains. At a quarter-turn, sine is one and
cosine is zero, so `(x_B,y_B)` becomes `(-y_B,x_B)` before translation. The
[notation aid](../../../Beginner-Reading-Aid.md#3-read-the-notation-without-changing-its-meaning)
explains radians and transpose.

### Worked example: an object in front of the camera

The base is at `(1, 2)` metres, facing `pi/2` radians in the world. A planar camera frame has the same orientation as the base and is 0.2 m ahead of it. An object is at `(1, 0)` in that camera frame.

1. Camera → base: `(1, 0) + (0.2, 0) = (1.2, 0)`.
2. Base → world: rotate `(1.2, 0)` by 90°, obtaining `(0, 1.2)`.
3. Add the base translation: world point is `(1, 3.2)` m.

Composition is `T_world_camera = T_world_base T_base_camera`. Apply the rightmost transform first. Swapping their order changes the result. This example uses an aligned **planar sensor frame**, not a 3D optical frame.

To invert a transform, subtract the translation and then undo the rotation:

\[
p_B=R^\top(p_A-t),\qquad T_{BA}=(R^\top,-R^\top t).
\]

Negating translation alone is wrong when the frames are rotated. In `labs/foundations.py`, compare `inverse_pose` with that tempting mistake and use a nonzero yaw to expose it.

## Extend the idea to real sensors

In 3D, translation has three components and rotation becomes a 3×3 orthonormal matrix, with `R.T @ R = I` and determinant +1. Homogeneous 4×4 matrices combine both operations. A quaternion is another rotation representation; its component order and normalization are part of the interface. Do not subtract quaternion components to obtain an angular error, and remember that `q` and `-q` represent the same orientation.

ROS body frames conventionally use x forward, y left, z up; camera optical frames use x right, y down, z forward. Lengths use metres and angles radians. Imported datasets and devices still need explicit adapters. [ROS REP-103 source](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)

For a camera aligned forward with a body frame, a direction `(x_opt, y_opt, z_opt)` becomes `(z_opt, -x_opt, -y_opt)` in the body frame, before adding mounting translation. Verify this by mapping each basis vector, not by guessing Euler angles.

An illustrative mobile manipulator tree is:

```text
map -> odom -> base_link -> camera_link -> camera_optical_frame
                       \-> arm_base -> ... -> tool_frame
```

The camera and the arm are separate branches. `odom` supports a continuous local estimate that can drift. `map` supports a globally corrected estimate that may jump when localization updates. Neither is a guarantee of exact physical truth. [ROS REP-105 source](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst)

## A correct frame at the wrong time is still wrong

A transform belongs to a time. If the camera captured at 10.000 s and the base has moved by the time inference finishes at 10.080 s, transforming the old detection with the current base pose mixes states. Ask for the transform at the capture timestamp, or use an explicitly documented prediction model.

**Age = time used − time captured**, with both timestamps in one clock domain. In a simulation, wall time and simulation time are different clocks. A simulation pause is not a giant physical timestep. On distributed hardware, record clock synchronization assumptions and uncertainty.

At 0.5 m/s, 80 ms of unmodelled age corresponds to 4 cm of translation under a constant-velocity assumption. That is a geometric estimate, not measured grasp error. If this exceeds your task tolerance, faster inference, better scheduling, prediction, or a slower motion may be necessary.

## Practice and check your reasoning

Run `python3 labs/foundations.py frames` from this course directory.

**Guided variation:** replace the base yaw by zero and predict the world point before rerunning. It becomes `(2.2, 2)` m. Replace the camera translation by `(0, 0.2)` and explain which direction the object moves in world coordinates.

**Independent task:** use two nonzero rotations and translations. Verify composition against two sequential point transforms, then invert the result. Require a round-trip error below `1e-10` m for these small double-precision inputs. Add a stale timestamp and a future timestamp; both must be rejected. This numerical tolerance is a software check, not your physical calibration tolerance.

**Self-check:** Why does a round-trip test alone fail to prove the convention is right? A forward and inverse implementation can share the same mistaken convention. Include a hand-computed basis-vector example as an independent check. Why should you not silently clamp a future timestamp's age to zero? It hides a clock mismatch and falsely makes the observation look fresh.

**Handoff:** save one labelled frame diagram, the transformed point and round-trip error, and the capture/use timestamps. Bring those fields into the sensor contract in [lesson 6](Lecture-06.md).
