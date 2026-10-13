# Lecture 20 — Integrating a VLA with Isaac Sim and Isaac Lab

[Previous](Lecture-19.md) · [Course overview](README.md) · [Next](Lecture-21.md)

**Prerequisites:** Lectures 06, 14, and 19; Isaac course Lectures 14–16 or equivalent simulator experience.

**Effort:** 90 min teaching, 1–3 h environment setup, 3–5 h GPU lab; CPU contract exercise takes 45 min.

**Evidence cutoff:** October 13, 2026. Numerical examples below are illustrative unless explicitly identified as executed lab results.

## Separate the simulator from the policy contract

Isaac Sim supplies simulation capabilities; Isaac Lab supplies robot-learning workflows and task abstractions. A VLA adapter is the boundary between a specific task's observations/actions and a specific checkpoint's training contract. Installing the simulator does not create that adapter. A policy trained on end-effector deltas cannot directly consume joint torque action spaces merely because both use floating-point arrays.

This course connects to the repository's [Isaac engineering lab sequence](../Isaac%20Sim%20and%20Isaac%20Lab/Lab-Guide.md). Its selected teaching stack is Isaac Sim 6.1 with Isaac Lab 3.0.0-EA, where applicable to the chosen backend. Use the [versioned installation documentation](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html), record exact commits and compatible packages, and treat early-access APIs as version-specific. Do not transplant an older command into a different release without checking its documentation.

**Outcome:** specify and implement a task-to-policy adapter with reproducible resets, verified transforms, and a closed-loop recording.

## Define one concrete task

Choose a tabletop reach-and-grasp task with one arm, a gripper, two distinguishable objects, and a fixed RGB camera. The instruction chooses the object. Begin with a fixed camera and robot base, then vary object poses. This is a curriculum design, not the name of a preinstalled Isaac environment.

Write a manifest before training or importing a model:

```yaml
# Exercise schema: these are fields to implement, not Isaac Lab API keys.
observation:
  rgb: {layout: HWC, channels: RGB, timestamp: simulation_seconds}
  proprioception: {order: [joint_positions, gripper_width], units: [rad, m]}
action:
  representation: end_effector_delta
  frame: robot_base
  translation_unit: m
  rotation: rotation_vector_rad
  gripper: width_m
  action_period_s: 0.1
  predicted_horizon: 8
  executed_prefix: 2
reset:
  initial_state_source: recorded_task_seed_and_asset_revision
```

The adapter maps this task schema to the checkpoint schema. If the checkpoint expects a different rotation convention, action cadence, camera count, or gripper sign, document and test the conversion. Some mismatches require retraining rather than a reshape. Padding absent joints is only valid if the model was trained to interpret the corresponding mask.

## Trace the time bases

Let physics timestep be \(\Delta t_p\), controller decimation \(d\), and target decimation \(k\) controller steps. The predicted-action interval is

\[
\Delta t_a = k d\Delta t_p,\qquad \Delta t_{\mathrm{replan}}=K\Delta t_a.
\]

With physics at 200 Hz, a controller every two physics steps, and an action target every ten controller steps, controller frequency is 100 Hz and target frequency is 10 Hz. Here \(K\) is the number of targets executed before requesting a new chunk. An eight-step policy chunk predicts 0.8 seconds of targets. Executing two targets before replanning makes the nominal replan interval 0.2 seconds (5 Hz); it does not make each target a physics-step command.

Do not multiply an already integrated displacement by timestep again. A velocity command has units m/s; an end-effector delta has units m per action interval. If the dataset changes frequency, resampling requires interpreting those semantics. Simply keeping every second delta halves the integrated motion for many trajectories.

## Build the adapter in stages

First use a scripted reach command to verify the simulator's axes, controller, reset, and termination. Second record a demonstration and replay its physical actions through the exact adapter. Third evaluate the trained policy offline on that recording. Only then close the loop.

**Adapter pseudocode — intentionally not an Isaac API recipe:**

```python
observation = capture_timestamped_observation(task)
policy_input = encode_images_language_and_state(observation, checkpoint_contract)
normalized_chunk = policy.predict(policy_input)
physical_chunk = decode_and_denormalize(normalized_chunk, training_statistics)
validated_chunk = validate_frames_units_limits_and_finiteness(physical_chunk)
scheduler.submit(validated_chunk, observation.timestamp, task.reset_epoch)
```

Each named function needs implementation and a contract test. The simulator should advance under the controller while inference runs if the intended deployment is asynchronous. Pausing physics during inference can be useful for initial debugging, but it removes latency from the control problem; label such results accordingly.

Image handling deserves its own trace. Store raw camera samples, crop coordinates, resize algorithm, channel order, value range, and encoder normalization. A mirrored image can still look plausible to a human. Include an asymmetric calibration object and verify its pixel movement when the robot moves along a known axis.

## Reset isolation and language leakage

Vectorized environments must reset policy state per environment. When environment 7 terminates, discard its chunk, previous action, temporal ensemble, and history cache without resetting its neighbors. Give every reset a monotonically increasing local epoch; a delayed response with an earlier epoch must be rejected even if its timestamp looks recent.

A task's target object ID may be needed by the reward or success function. Do not accidentally expose that ID as privileged policy input when claiming visual-language grounding. An object-state oracle is useful for debugging, but report it as a separate baseline. Test the same scene with two different instructions; a language-conditioned controller should change its target appropriately.

## Guided lab and measurements

On CPU, construct two short episodes and use `chunk_with_mask` from the [lab tools](Lab-Guide.md) to prove that action chunks cannot cross resets. Add an intentionally stale reset epoch to your adapter design and specify the rejection path. This verifies data bookkeeping, not Isaac integration.

On a supported GPU machine, follow the versioned setup and task-template instructions linked in the Isaac course. Record exact GPU, driver, asset hashes, simulator/backend, renderer, camera mode, and task commit. Stop after one scripted success and one intentionally failed rollout if resource limits prevent policy training. Do not claim a VLA result from a scripted baseline.

The full lab artifact contains a schema, one demonstration replay, per-axis action plots, two paired instruction rollouts, reset logs, and a camera-to-action timing trace. Measure success under fixed seeds, wall-clock/simulation-time ratio, sensor age, and GPU memory with the simulator and policy running together. Memory measured in separate processes at separate times underestimates their simultaneous requirement.

## Failure clinic and self-check

If offline predictions look right but the arm moves in the wrong direction, inspect frame and gripper conventions before changing the learning rate. If only some vector environments fail, inspect per-environment history and reset masks. If all tests pass while physics is paused but fail at wall-clock speed, investigate observation age and action scheduling.

1. Why is a reward target ID different from an observation? The evaluator may use privileged state that the deployed policy cannot access.
2. Does a 10 Hz action-target cadence imply a 10 Hz motor controller? No; a faster controller can track targets between policy updates.
3. What does a CPU reset test establish? Correct episode bookkeeping under its fixture, not simulator behavior or physical control quality.
4. What makes a simulated instruction test convincing? Hold the scene fixed, change the requested target, and measure the resulting task behavior without privileged target leakage.

---

[Previous](Lecture-19.md) · [Lab guide](Lab-Guide.md) · [Next](Lecture-21.md)
