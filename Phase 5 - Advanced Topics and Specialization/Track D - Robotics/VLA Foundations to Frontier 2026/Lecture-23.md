# Lecture 23 — Robustness, System Identification, and Sim-to-Real Transfer

[Previous](Lecture-22.md) · [Course overview](README.md) · [Next](Lecture-24.md)

**Prerequisites:** Lectures 06, 13–16, and 19–22; rigid transforms and basic dynamics.

**Effort:** 90 min teaching, 1 h fixture setup, 3–5 h analysis; hardware identification is an optional supervised extension.

**Evidence cutoff:** October 13, 2026. Numerical examples below are illustrative unless explicitly identified as executed lab results.

## Identify the source of the gap

A simulation-trained policy may fail on a physical robot because of appearance, geometry, dynamics, timing, action semantics, or task-distribution differences. Calling all of these “domain shift” is too coarse to guide an experiment. Split the observation-to-command path into measurable components and locate the first mismatch.

**Outcome:** produce a transfer hypothesis, an identification experiment, and a held-out robustness matrix that distinguishes calibration errors from policy limitations.

An accurate renderer cannot repair the wrong gripper zero point. More demonstrations cannot reliably compensate for a silently inverted coordinate axis. Conversely, a perfectly calibrated camera does not teach the model how to handle an unseen object category. Match the intervention to the failure mechanism.

## From coordinates to behavior

Represent a camera-to-base transform by \(T_{BC}\in SE(3)\), mapping camera coordinates to robot-base coordinates. For a point written in homogeneous coordinates \(p_C\), transformation gives \(p_B=T_{BC}p_C\). A calibration error affects actions through both image interpretation and downstream geometry. Explicitly state transform direction; \(T_{BC}\) and \(T_{CB}\) are inverses, not interchangeable labels.

Consider an illustrative camera translation error of 1 cm and a grasp tolerance of 5 mm. A highly accurate image-conditioned policy can still miss because the geometry error alone exceeds task tolerance. Repeated misses in a consistent direction suggest systematic bias. Broad, state-dependent error may implicate dynamics, perception, or multimodal actions instead.

For rotations, compose transformations on the correct side. A tool-relative delta and a base-relative delta produce different motion when the gripper rotates. Quaternion component ordering, normalization, handedness, and angle conventions belong in the adapter contract. The [ROS coordinate convention reference](https://www.ros.org/reps/rep-0103.html) is a useful shared convention; individual datasets may use others and require explicit conversion.

## Estimate a small dynamics model

Before randomizing everything, identify a simple actuator response. As a teaching model, use

\[
y_{t+1}=\alpha y_t+(1-\alpha)u_{t-d}+\epsilon_t,
\qquad 0\leq\alpha<1,
\]

where \(y\) is measured velocity, \(u\) is commanded velocity, and \(d\) is integer delay. This is an approximation for analysis, not a complete robot dynamics equation. Fit \(\alpha\) over candidate delays using one trajectory and compare prediction error on another. If a different input amplitude changes the fitted parameter dramatically, the linear approximation may be missing saturation, friction, or mode changes.

For \(\alpha=0.8\) with 10 ms sampling, the equivalent first-order time constant is approximately \(-0.01/\log(0.8)=44.8\) ms. Adding a 30 ms command delay changes phase and tracking even if steady-state velocity is correct. A simulator that matches final pose but not transient response can train a policy that acts too aggressively.

Use low-risk, supervised identification motions appropriate to the device and its controller. The CPU version generates data from the equation, adds seeded noise, searches candidate delays, and checks parameter recovery on held-out samples. It establishes understanding of estimation, not calibration of a real actuator.

## Design domain randomization around hypotheses

Randomization samples simulator parameters \(\phi\) from a distribution \(p(\phi)\). Training optimizes expected performance over that distribution. It does not guarantee robustness outside it, and unrealistic independent combinations may waste capacity or teach artifacts.

Group parameters by mechanism: visual appearance and lighting; camera pose and intrinsics; object shape and placement; mass, friction, and compliance; sensor delay and noise; controller gains and command delay. Start with measured uncertainty ranges where available. Preserve known correlations, such as camera mounting geometry or linked actuator limits.

Use three sets: training randomization, validation perturbations used for decisions, and held-out transfer conditions. A domain-randomization range repeatedly expanded after looking at the test set makes that test part of development. Log the history of these decisions. Separate interpolation within the training range from extrapolation beyond it.

## Robustness requires language and interaction tests

Construct counterfactual pairs: same scene with different target instruction; same instruction with swapped object locations; distractor objects with similar appearance; modest paraphrases; occluded target; and physically impossible requests. A model that always grasps the nearest object can perform well on a biased dataset while failing language grounding.

For temporal robustness, disturb the object after a chunk starts and measure reaction delay, recovery rate, and inappropriate persistence. Long open-loop chunks can look smooth while ignoring the new state. For long tasks, distinguish successful primitive execution from correct sequencing and from recovery after a failed primitive.

Do not reduce every result to average success. Include failure type, intervention, maximum physical error, latency, and performance by perturbation severity. A perturbation sweep reveals thresholds and graceful degradation better than one arbitrarily “hard” setting.

## Guided transfer lab

Follow the [Isaac system-identification lecture](../Isaac%20Sim%20and%20Isaac%20Lab/Lecture-20.md) for a simulator-focused companion. For this course, create a 3×3 grid of synthetic delay and actuator response settings around a nominal model. Generate one identification trajectory, fit parameters, then evaluate a fixed controller on an independent command trajectory.

Next choose one visual perturbation and one timing perturbation for the Isaac VLA adapter. Measure their individual effects and their combination. Interactions matter: a slightly inaccurate grasp pose may become a failure only when combined with stale commands. Use paired initial states and keep the success function unchanged.

The artifact contains the nominal contract, measured or assumed uncertainty ranges, parameter-estimation code, residual plots, task outcomes, and a short explanation of the next experiment. A professional report marks which numbers are measured on hardware, measured in simulation, synthetically generated, or merely proposed.

## Failure clinic and self-check

Training loss improves after aggressive randomization but nominal success falls: inspect unrealistic combinations, capacity limits, and action normalization. A policy transfers only when rendering is slowed: inspect timing and controller behavior. Failures cluster at one workspace edge: inspect calibration and kinematic reachability before adding global noise.

1. Why identify before randomizing? Measurements constrain plausible ranges and identify systematic errors that should be corrected rather than learned around.
2. Is visual realism sufficient for transfer? No; dynamics, timing, sensing, and command semantics also affect the closed loop.
3. Why hold out command trajectories during identification? A model can fit one excitation pattern without predicting a different operating regime.
4. Does passing nine perturbation settings prove general robustness? It supports those tested conditions and motivates broader testing; it does not establish an unrestricted guarantee.

---

[Previous](Lecture-22.md) · [Lab guide](Lab-Guide.md) · [Next](Lecture-24.md)
