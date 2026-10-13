# Lecture 15 — Frames, units, and asset validation

**Prerequisites:** Lectures 02–04 and 14. **Time:** 2–3 hours. **Outcome:** establish a robot contract that survives an API or asset change.

## Write the contract before moving the robot

For each signal record units, frame, ordering, shape, and timestamp meaning. Example:

| Signal | Contract |
|---|---|
| Joint position | Radians, named joint order, measured at simulation step k |
| End-effector position | Metres, robot-base frame |
| Camera optical pose | Named optical axes and transform to base |
| Quaternion | Explicit component order and source API |
| Action | Position target, velocity, torque, or normalized delta; scale and limits |

Never infer action meaning from shape alone. Two seven-element vectors can encode seven joint targets or a pose command, and two position controllers can differ in whether a value is absolute or incremental.

Isaac versions and interfaces can use different quaternion orders. Inspect the [pinned Isaac Lab source](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA) and the relevant Isaac Sim API at the boundary. Even within one installation, an API wrapper and raw USD data may differ.

## Worked transform

Let the base frame have a +90° rotation about world z. A unit vector along base x maps to world y. The CPU fixture uses xyzw = (0, 0, sqrt(1/2), sqrt(1/2)) and verifies the result.

The identity quaternion is a weak convention test because some code paths special-case it. Use a nontrivial rotation, its inverse, and a known axis. A round trip alone is also insufficient if both conversions share the same error; include the independently known world-y result.

## Asset checks

Before tuning a controller, inspect scale, mass, center of mass, inertia, collision approximation, joint axes, limits, actuator gains, and default pose. Rendered geometry can look right while collision geometry is wrong. A tiny scale error can cause a large dynamics change.

Use a gravity-only drop, a single-joint sweep inside limits, and a static-contact check. Inspect joint names rather than assuming import order. Keep asset overrides separate from the source asset and save both revisions.

## Lab

1. Run the quaternion exercise in [robotics_contracts.py](labs/robotics_contracts.py).
2. Draw world → environment origin → base → tool → camera.
3. Log one robot pose through the simulator, observation builder, and ROS/export boundary.
4. Check the known rotated axis and inverse transform.
5. Compare visual and collision geometry in a small scene.
6. Deliberately swap two joint names in a test mapping and prove an assertion fails before actuation.

**Independent task:** import a second robot or mount the camera at a different pose. Reuse the contract tests rather than rewriting the policy.

## Self-check

- Why is a plausible rendered pose insufficient? The policy may receive values in another frame or order.
- Why map joints by name? Source/import order can change without changing vector length.
- Does normalization remove unit errors? It can hide them while leaving physically incorrect meaning.

**Exit artifact:** frame diagram, unit/action table, known-answer transform checks, and an asset validation log.

[Previous](Lecture-14.md) · [Lab sequence](Lab-Guide.md) · [Next: task correctness](Lecture-16.md)
