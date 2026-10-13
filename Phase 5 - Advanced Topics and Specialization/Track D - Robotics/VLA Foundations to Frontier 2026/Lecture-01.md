# Lecture 01 — What a VLA policy is

Build a precise contract between a vision-language-action model and the robot that executes its decisions.

**Prerequisites:** probability, vectors, basic feedback control, and Python. **Effort:** 30 minutes setup, 90 minutes lab, 60 minutes analysis. **Hardware:** CPU laptop; no robot required. **Stack:** algorithms, runtime, system integration, application. **Roles:** robot learning engineer, inference engineer, robotics systems engineer.

[Course](README.md) · [Next: historical foundations](Lecture-02.md)

## The engineering problem

A camera image and the instruction “put the cup in the tray” do not specify motor currents. They leave object position, grasp choice, approach direction, contact forces, timing, and recovery unresolved. A VLA policy learns some of that mapping from observations and language to an action representation. Its output becomes meaningful only when a controller interprets it using the same units, frames, and timing conventions used during training.

In this course, a VLA is a learned policy conditioned on visual observations and language that produces executable robot actions or action chunks. The boundary is operational: inspect the output. A model that writes “grasp the cup” still needs a skill executor; a model that returns an end-effector displacement needs an inverse-kinematics or Cartesian controller. Neither output is automatically a motor torque. This distinction is visible in the different roles of [SayCan](https://arxiv.org/abs/2204.01691) and [RT-2](https://arxiv.org/abs/2307.15818).

The misconception to remove is that a sufficiently capable multimodal model replaces the complete robotics stack. Sensors still have timestamps, actuators still have limits, and a delayed correct action can become incorrect.

## State, observation, and memory

Let the physical state \(s_t\) include robot configuration, object poses, velocities, contact state, and unobserved properties such as friction. The robot receives an observation \(o_t\), which may contain images, joint measurements, and force signals. An instruction is \(l\); an action is \(a_t\).

One useful abstraction is a partially observable Markov decision process: dynamics \(P(s_{t+1}\mid s_t,a_t)\), observation model \(P(o_t\mid s_t)\), and reward \(r(s_t,a_t,l)\). A belief distribution \(b_t(s)=P(s_t=s\mid o_{\leq t},a_{<t})\) summarizes uncertainty when the models are known. Most VLA implementations approximate useful memory through a finite observation history or learned latent state rather than computing an exact belief. The formal background is [Kaelbling, Littman, and Cassandra](https://www.cassandra.org/arc/papers/aij98.pdf).

Write a history-conditioned chunk policy as

$$
\pi_\theta(A_t\mid o_{t-k+1:t},a_{t-k+1:t-1},l,e),
\qquad A_t=(a_t,\ldots,a_{t+H-1}),
$$

where \(e\) identifies the embodiment or its configuration. This notation exposes choices hidden by the phrase “the model takes an image”: history length \(k\), horizon \(H\), which previous actions are known, and whether the robot identity is explicit.

Consider a cup temporarily hidden behind a gripper. The current image can be identical when the cup was successfully grasped and when it slipped away. A memoryless image policy must average or guess between incompatible actions. A previous image, measured gripper width, or contact signal can resolve the ambiguity. More model parameters cannot recover information that no input contains.

## Policy, planner, and controller

A planner selects goals or searches over candidate futures. A policy maps its available information to decisions. A controller converts targets into actuator commands while rejecting disturbances. These roles can be learned, analytical, or combined; their distinction is about responsibility rather than neural-network size.

An illustrative stack uses a task planner when a subtask ends, a VLA at 10 Hz, a Cartesian tracking controller at 250 Hz, and motor control at a faster device-specific rate. These are teaching choices, not universal requirements. The VLA should not block the tracking loop while token generation waits for GPU scheduling.

A command envelope should therefore include the action value, representation identifier, reference frame, units, creation timestamp, intended execution interval, validity duration, and sequence number. The adapter must reject malformed and stale messages. A confidence score can inform policy selection, but a model's token probability is not a certified probability of physical success.

## Learning objectives versus task objectives

Behavior cloning learns from demonstrations \(D\):

$$
\mathcal L_{\mathrm{BC}}(\theta)
=-\mathbb E_{(h,l,A)\sim D}\log\pi_\theta(A\mid h,l).
$$

This is an optimization objective over recorded examples. Deployment instead cares about rollout return, success, time, collisions, interventions, and resource consumption. Low imitation loss does not prove that a robot recovers after an object moves or that it respects a new workspace boundary.

Even the action likelihood depends on representation. A millimeter error and a one-radian error cannot share an unexamined scalar loss. Define normalization and per-dimension metrics before comparing policies. A policy that improves average loss by predicting the common “do nothing” action may become worse at rare grasp transitions.

## Worked example: a correct action arriving late

Assume a 10 Hz policy receives an image at time zero. Image processing takes 12 ms, policy inference 64 ms, and command transport 8 ms. Total observation-to-command delay is 84 ms, leaving 16 ms before the next 100 ms policy deadline. At an object speed of 0.20 m/s, the object can move

$$
0.20\times0.084=0.0168\ \mathrm m=16.8\ \mathrm{mm}.
$$

This is a kinematic estimate, not a measured grasp error. It excludes exposure time and queuing, so it may underestimate state age. Doubling camera frequency does not remove the 84 ms processing path. Reducing inference to 32 ms lowers that path to 52 ms and the displacement estimate to 10.4 mm.

Now predict ten actions at 20 Hz. The chunk covers 0.5 s, but executing all ten without refreshed feedback exposes the last action to substantial scene changes. Executing only two actions before replanning reduces open-loop exposure at the cost of more inference requests. Prediction horizon and execution horizon are separate configuration values.

## Lab: specify and break the boundary

**Exercise / pseudocode; not a tested robot integration.** Create `policy_contract.json` describing a seven-dimensional action: three base-frame position deltas in meters, three small rotation-vector increments in radians, and an explicitly defined gripper command. Include the policy period, maximum image age, controller interface, and execution horizon.

Create a synthetic CSV with columns `image_time_ms`, `policy_done_ms`, `arrival_ms`, `sequence_id`, and `dx_m`. Use 20 messages at 100 ms intervals. Set the normal delay to 84 ms; make message 7 arrive after 160 ms and duplicate message 12. Implement a validator with this structure:

```python
# Exercise pseudocode: monotonic clock and limits must be supplied.
if not finite(action) or action.schema != configured_schema:
    reject("invalid action")
elif arrival_time - image_time > max_image_age:
    reject("stale observation")
elif sequence_id <= last_accepted_sequence:
    reject("duplicate or reordered action")
else:
    accept(action)
```

Use 120 ms as this fixture's maximum image age. The intended result rejects the delayed and duplicate messages. Record rejection counts, maximum accepted age, deadline misses, and a timeline of accepted commands. Distinguish “accepted” from “executed”; a later controller can still reject a command.

For a guided variation, change the age limit to 80 ms and explain why every normal message now fails. For independent transfer, extend the envelope to a bimanual chunk with two cameras whose timestamps differ. Define whether synchronization failure rejects the entire observation or masks one camera. Stop after the finite fixture and two variations; no training is needed.

The CPU exercise validates interface reasoning only. It cannot establish controller stability, GPU tail latency, contact behavior, or real-world success. Its final artifact is the contract, validator, raw CSV, and one-page failure report.

## Self-check and exit criteria

1. **Can a single image always determine the next action?** No. Explain a pair of physically different states with the same image and identify an additional sensor or memory signal that distinguishes them.
2. **Does a 50-step chunk imply 50 steps of open-loop execution?** No. A receding-horizon executor may consume a short prefix and request a replacement.
3. **Why can a faster model lower task success?** It may sacrifice useful observations, change action precision, or increase command frequency without preserving the training-time control contract.

You are ready to continue when a reviewer can identify exactly what your policy owns, reproduce your rejected-message cases, and trace every accepted command back to the observation that produced it.

[Previous: course entry](README.md) · [Next: historical foundations](Lecture-02.md)
