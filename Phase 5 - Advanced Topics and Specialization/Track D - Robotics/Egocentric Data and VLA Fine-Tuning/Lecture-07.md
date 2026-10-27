# Lecture 07 — From Human Hands to Robot Actions: Retargeting and Feasibility

Convert a tracked human hand into a 7-D robot action stream in a stated base frame, and report which parts of the clip a given arm cannot execute.

**Prerequisites:** [Lecture 03](Lecture-03.md) (frames and clocks), [Lecture 06](Lecture-06.md) (hand-track quality), [VLA Foundations Lecture 06](../VLA%20Foundations%20to%20Frontier%202026/Lecture-06.md) (geometry, robot state, control contracts) and [Lecture 08](../VLA%20Foundations%20to%20Frontier%202026/Lecture-08.md) (action representations). **Effort:** reading 1.5 hours, exercise 2–3 hours. **Hardware:** CPU track. **Roles:** robot learning data engineer, robot foundation-model (post-training) engineer.

**Misconception this lecture corrects:** "retargeting is a coordinate change, so a clean hand track gives valid robot actions." A coordinate change moves the motion into the robot's frame. It does not make the motion reachable, slow enough, or graspable with the robot's hand, so the feasibility report is part of the output.

## 1. The pipeline and its frames

`retarget_episode` in [`labs/retarget.py`](labs/retarget.py) runs four steps: `resample` the hand track (`HandFrame`: device-clock time, `T_world_wrist`, thumb tip, index tip) onto the control-rate clock, with slerp for rotation; compute virtual-gripper poses in the base frame with `eef_poses`; turn them into `relative_actions` plus a `GripperMap` opening; and check them with `feasibility_report` against `Limits`. With b = robot base, w = SLAM world, h = wrist and e = virtual gripper:

$$
T_{be}(t) = T_{bw}\, T_{wh}(t)\, T_{he}
$$

| Transform | Code name | Where it comes from |
|---|---|---|
| \( T_{wh}(t) \) | `HandFrame.T_world_wrist` | Hand tracking chained through closed-loop `T_world_device` (Lectures 03 and 06) |
| \( T_{he} \) | `T_wrist_eef` | Your choice: the virtual gripper (tool center point) in the wrist frame, oriented like the robot's tool frame |
| \( T_{bw} \) | `T_base_world` | Calibration between the SLAM world and the robot base; for human-only clips with no robot present, a canonical frame you define and document |

## 2. Run it on the synthetic reach

`synthetic_reach` is copied from [`labs/test_geometry_retarget.py`](labs/test_geometry_retarget.py) (lines merged): over 1 s at 30 fps the wrist moves 0.15 m along +x while yawing at 0.3 rad/s, and the thumb–index gap closes from 8 cm to 2 cm at 0.6 s. The fingertips sit at \( (0.05, \pm\text{gap}/2, 0) \) in the wrist frame, so a 5 cm offset puts the virtual gripper at their midpoint. Run from the repository root with `EGO_COURSE` set as in the [Lab guide](Lab-Guide.md).

**Tested here:**

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import math
from ego_geometry import SE3, quat_from_axis_angle
from retarget import HandFrame, eef_poses, retarget_episode
def synthetic_reach(n=31, fps=30.0, dx_per_s=0.15, yaw_per_s=0.3, close_at=0.6):
    """Copied from test_geometry_retarget.py: a hand moving along +x while yawing, closing at 0.6 s."""
    frames = []
    for i in range(n):
        t = i / fps
        wrist = SE3(quat_from_axis_angle((0, 0, 1), yaw_per_s * t), (0.4 + dx_per_s * t, 0.0, 0.2))
        gap = 0.08 if t < close_at else 0.02
        frames.append(HandFrame(t, wrist, wrist.apply((0.05, gap / 2, 0.0)), wrist.apply((0.05, -gap / 2, 0.0))))
    return frames
frames = synthetic_reach()
T_wrist_eef = SE3(t=(0.05, 0.0, 0.0))                  # virtual gripper at the thumb-index midpoint
f0 = frames[0]
mid = tuple((a + b) / 2 for a, b in zip(f0.thumb_tip, f0.index_tip))
print("eef", eef_poses([f0], T_wrist_eef, SE3())[0].t, "midpoint", mid)
out = retarget_episode(frames, 10.0, T_wrist_eef, SE3())
print(out["report"])
for k in (0, -1):
    print(k, [round(x, 4) for x in out["actions"][k]])
EOF
```

```text
eef (0.45, 0.0, 0.2) midpoint (0.45, 0.0, 0.2)
{'poses': 11, 'outside_workspace': 0, 'too_fast_steps': 0, 'too_fast_rotation_steps': 0, 'feasible_fraction': 1.0, 'ok': True}
0 [0.015, 0.0015, 0.0, 0.0, 0.0, 0.03, 0.8667]
-1 [0.0144, -0.0025, 0.0, 0.0, 0.0, 0.03, 0.0667]
```

* **The chain checks out for one frame:** the virtual gripper lands on the fingertip midpoint. Do the same check on real data with a frame where you can see both tips.
* **Rate.** 31 frames at 30 fps become 11 poses at 10 Hz and 10 actions. Resample to the control rate of the robot data you will co-train with (`svla_so101_pickplace`, for example, records at 30 fps).
* **Actions.** Each is `(dx, dy, dz, rx, ry, rz, gripper)`: translation and rotation vector in the current gripper frame, then the opening. 0.015 m per 0.1 s is 0.15 m/s and 0.03 rad per step is 0.3 rad/s. The 1.5 mm sideways term is the lever arm, since yaw swings a point 5 cm ahead of the wrist. The last translation has turned with the accumulated yaw because it is expressed in the gripper's own frame.
* **Gripper.** The opening is \( g = \min(1, \max(0, (d - d_{\text{closed}}) / (d_{\text{open}} - d_{\text{closed}}))) \) for thumb–index distance \( d \). With the default `GripperMap(closed_m=0.015, open_m=0.09)`, the 8 cm gap maps to 0.8667 and the closed 2 cm gap to 0.0667. A hand closed on an object does not bring thumb and index together (the fixture keeps about 2 cm), so "closed" maps near 0, not to 0. Calibrate the two distances per person and object size, or threshold to a binary command if the gripper is binary.

## 3. Relative versus absolute actions

The relative action is \( \Delta_k = T_{be}(t_k)^{-1}\, T_{be}(t_{k+1}) \). Multiplying every pose on the left by a constant \( T_{bw} \) cancels: \( (T_{bw} P_k)^{-1} (T_{bw} P_{k+1}) = P_k^{-1} P_{k+1} \). Relative actions therefore do not depend on the world origin or the base calibration, which suits data from many rigs: UMI uses relative-trajectory actions, GR00T N1.7 uses a relative end-effector action space shared by robot and human embodiments, and EgoDex's per-episode ARKit origin does not matter for them. Absolute actions (target poses in the base frame) carry the calibration inside every label; a robot needs them to reach a specific place, and so does any absolute-pose state you feed the policy. The same cancellation hides calibration errors, which is the first negative case.

## 4. Negative cases: swapped calibration and too-fast motion

The lab's `Limits` defaults (workspace 0.2–0.8 m in x, ±0.4 m in y, 0–0.6 m in z; 0.5 m/s; 2.0 rad/s) are placeholders. Replace them with your arm's documented limits and your table's real workspace.

**Tested here:**

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
import math
from ego_geometry import SE3, quat_from_axis_angle
from retarget import HandFrame, Limits, retarget_episode
from test_geometry_retarget import synthetic_reach      # the same function as above
T_base_world = SE3(quat_from_axis_angle((0, 0, 1), math.pi / 2), (0.0, -0.6, 0.0))
box = Limits(workspace_min=(-0.2, -1.0, 0.0), workspace_max=(0.2, 0.0, 0.6))
ok = retarget_episode(synthetic_reach(), 10.0, SE3(), T_base_world, limits=box)
bad = retarget_episode(synthetic_reach(), 10.0, SE3(), T_base_world.inverse(), limits=box)
print("correct:", ok["report"]["outside_workspace"], "outside | swapped:", bad["report"]["outside_workspace"], "outside")
print("max action difference:", max(abs(a - b) for x, y in zip(ok["actions"], bad["actions"]) for a, b in zip(x, y)))
wide = Limits(workspace_min=(-2, -2, -2), workspace_max=(2, 2, 2))
fast = synthetic_reach(dx_per_s=1.5)                     # 1.5 m/s: three times the lab's 0.5 m/s limit
for s in (1, 2, 3, 4):
    slowed = [HandFrame(f.t * s, f.T_world_wrist, f.thumb_tip, f.index_tip) for f in fast]
    r = retarget_episode(slowed, 10.0, SE3(), SE3(), limits=wide)["report"]
    print(f"time x{s}: poses {r['poses']}, too_fast_steps {r['too_fast_steps']}, ok {r['ok']}")
EOF
```

```text
correct: 0 outside | swapped: 11 outside
max action difference: 2.7755575615628914e-16
time x1: poses 11, too_fast_steps 10, ok False
time x2: poses 21, too_fast_steps 20, ok False
time x3: poses 31, too_fast_steps 19, ok False
time x4: poses 41, too_fast_steps 0, ok True
```

* **Swapped calibration** (`T_base_world.inverse()` where `T_base_world` is needed). The actions agree to 3e-16, so nothing in the action stream reveals the error. The workspace check puts all 11 poses outside. Always run the report in the base frame you will deploy in.
* **Too-fast motion.** A 1.5 m/s reach breaks the 0.5 m/s limit on every step. Stretching time by a factor \( s \) divides every speed by \( s \), so you need \( s \ge v_{\text{peak}} / v_{\max} = 3 \). At exactly 3 the speed sits on the limit and floating-point rounding fails 19 of 30 steps; 4 passes. Leave margin. Time scaling also stretches the episode (11 → 41 poses) and the timing of contact, and the policy then learns slower-than-human motion.
* **Report, do not clip.** Clipping positions or speeds silently produces a different motion than the human made. Mark infeasible steps, then decide whether to drop, time-scale or keep them for a different arm.

## 5. What the report does not check

| Not checked | Why it matters | Where it is handled |
|---|---|---|
| Inverse kinematics | A pose inside the box may be unreachable in that orientation, near a singularity or past a joint limit | IK on the robot model ([VLA Foundations Lecture 06](../VLA%20Foundations%20to%20Frontier%202026/Lecture-06.md)) |
| Collisions | The robot's arm sweeps a different volume than a human arm | Collision checking in simulation ([Isaac Sim and Isaac Lab](../Isaac%20Sim%20and%20Isaac%20Lab/README.md)) |
| Contact and force | Grasp success depends on force and friction, which no hand track observes (Lecture 06) | Robot data or simulation |
| Policy outcome | Feasible actions can still train a poor policy | Paired evaluation (Lecture 12) |

## 6. Five-finger hands versus a parallel jaw

| Target | What maps from the hand | Examples |
|---|---|---|
| Parallel-jaw gripper | Wrist pose plus one scalar (thumb–index gap) | This lab. UMI avoids the mapping: the person holds a gripper and its width is recorded (`robot{i}_gripper_width`) |
| Dexterous hand | Finger joints from hand keypoints, by IK or optimization | DexCap (IK plus point-cloud imitation from glove mocap); EgoScale (a 22-DoF hand); EgoVLA (IK and retargeting of predicted wrist and hand actions) |
| Humanoid | A unified human/humanoid state–action space | Humanoid Policy ~ Human Policy, which retargets differentiably |

A parallel jaw throws away almost all finger information, which makes it forgiving of Lecture 06's finger failures. A dexterous hand needs the full skeleton, and every occlusion and swap now corrupts many joints instead of one gap.

## 7. Exercise

**Exercise / pseudocode** (CPU only):

1. **Worked example → your arm.** Run both tested-here blocks. Replace `Limits` with your arm's workspace and speed limits and rerun the synthetic reach.
2. **Guided variation.** Swap the calibration and confirm that the report catches it while the actions do not change. Set `dx_per_s=1.5` and find the smallest time-scaling factor that passes with at least 10% speed margin. Set `GripperMap(closed_m=0.02)` and check that the closed value drops to 0 (up to floating-point rounding).
3. **Independent transfer.** Retarget a real track: your own Aria recording (MPS `T_Device_Wrist` and fingertip landmarks chained through closed-loop `T_world_device`) or the track from your Lecture 06 report. State `T_wrist_eef` and `T_base_world` and where each came from. A retargeted stream is a derived dataset, so check the source license first (EgoDex's CC-by-NC-ND terms restrict sharing derivatives, and the CC version is unverified; a reading of the license, not legal advice).

**Deliverable:** `retargeted_actions.jsonl` (one 7-D action per control step, with its timestamp) and `retarget_report.md` with the frames, the calibration source, the limits and their source, the control rate, the time-scaling factor and the `feasibility_report` output pasted verbatim. **Acceptance:** the report says `ok: True`, or every violation is explained and its steps are marked; the swapped-calibration check was run and caught; the gripper calibration values are stated.

## Failure clinic and self-check

**Failure clinic:** an inverted calibration produces a clean-looking action stream, and the error only shows when the robot reaches for the wrong place. Positions are clipped to the workspace and the "retargeted" motion no longer matches the human's. A track is resampled with host-clock times, so steps have uneven durations. A closed hand maps to 0.07, and on a thin object the gripper never closes far enough to hold it. A Lecture 06 left/right swap reaches the stream as a single huge step.

1. **Why did the swapped calibration leave every action unchanged?** A constant left-multiplied transform cancels in \( T_k^{-1} T_{k+1} \). Only checks in the base frame (workspace, absolute state) can catch it.
2. **When do you need absolute actions?** When the target is a specific place in the robot's frame, or the policy consumes absolute poses as state. Then the calibration is inside every label and must be verified.
3. **A reach peaks at 1.2 m/s and the arm allows 0.5 m/s. What time-scaling factor?** At least 2.4; choose about 2.7 or more for margin. Record it, because it also stretches contact timing.
4. **Why does a closed hand map to about 0.07 rather than 0?** The thumb–index gap stays near 2 cm around an object while `closed_m` is 1.5 cm. Calibrate per person and object, or threshold.
5. **The report says `ok: True`. Can the robot execute the stream?** Not established. IK, collisions and contact are unchecked (Section 5); the report is a necessary condition only.

[Previous: hands and objects](Lecture-06.md) · [Course index](README.md) · [Next: learning from video without robot actions](Lecture-08.md)
