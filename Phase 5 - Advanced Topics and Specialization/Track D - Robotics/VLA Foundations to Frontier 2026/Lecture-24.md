# Lecture 24 — Deployment Boundaries, Recovery, and Human Intervention

[Previous](Lecture-23.md) · [Course overview](README.md) · [Next](Lecture-25.md)

**Prerequisites:** Lectures 19–23; state machines and basic feedback control.

**Effort:** 75 min teaching, 30 min fixture setup, 2–3 h offline fault-injection lab.

**Evidence cutoff:** October 13, 2026. Numerical examples below are illustrative unless explicitly identified as executed lab results.

## Put authority in an explicit place

A VLA predicts a desired action or trajectory. A deployed robot also needs controllers, command validation, operating boundaries, and a recovery process. Model confidence is not a substitute for these mechanisms. Treat the model as one fallible component in a system whose behavior remains defined when inference fails.

**Outcome:** build an offline supervisor specification with testable transitions, fault injection, and an intervention log. This lecture teaches engineering design; it does not certify a robot or establish compliance with an application-specific safety standard.

Separate high-level task authority from low-level actuation authority. A language instruction may choose a task, but it should not silently change controller limits, disable monitoring, or redefine the physical workspace. External scene text and language-model outputs are observations and proposals, not trusted configuration changes.

## Start with an operating envelope

Write down the intended robot, end effector, workspace, object classes, payload range, camera configuration, task types, and human interaction assumptions. These define the domain in which the policy's evidence applies. “Generalist” describes training or capability ambitions; it does not remove operating assumptions.

For each boundary, define an observable condition and response. Examples include invalid action shape, nonfinite values, excessive command age, camera loss, impossible joint target, repeated unsuccessful grasps, and unexpected contact. The proper physical response depends on the robot controller and task. For a loaded manipulator, abruptly removing actuation can itself be hazardous; choose and validate responses with the device's documented control behavior.

A command validator can enforce array shape and finite values without understanding task success. A controller can enforce joint and velocity limits without understanding language. A task supervisor can detect repeated failures without directly commanding motor currents. These layers provide different evidence and should remain independently testable.

## Understand what uncertainty can tell you

A generative action distribution can be broad because multiple actions are valid. It can also be confidently wrong under distribution shift. Entropy, ensemble disagreement, and sampled-action variance are useful signals only after calibration against the failure modes of interest. An action distribution's numerical scale depends on representation and normalization, so do not compare raw variance across incompatible models.

A simple selective-execution experiment chooses an uncertainty score \(u(o)\), accepts predictions below threshold \(\tau\), and sends others to a fallback. Plot **coverage**—the fraction accepted—against **risk**, such as failure rate among accepted episodes. Choose thresholds on validation data; evaluate the resulting policy and intervention load on held-out data. A low-risk system accepting only trivial tasks may have limited usefulness, which coverage makes visible.

For a worked example, suppose a supervisor accepts 60 of 100 attempts and 54 of those succeed. Accepted success is 90%, but autonomous completion across all requests is 54%. If every rejection requires a person, intervention demand is 40%. Reporting only 90% hides the operational cost. Count recovery attempts and retries consistently as well.

## Recovery is a policy with a state

A naive retry repeats the same action from a changed physical state. A better recovery procedure observes the result, identifies the failed stage, and decides whether to reobserve, reposition, replan, request assistance, or terminate. Repeated failures should have a bounded retry budget, not an infinite loop.

**Exercise transition design:**

```text
IDLE -> task accepted within scope -> OBSERVE
OBSERVE -> synchronized valid input -> EXECUTE
EXECUTE -> progress confirmed -> OBSERVE or COMPLETE
EXECUTE -> recoverable failure -> RECOVER
RECOVER -> new state verified -> OBSERVE
ANY -> invalid command / expired state / controller fault -> controller-defined response
RECOVER -> retry budget exhausted -> ASSIST or ABORT
```

Make progress predicates task-specific. “Gripper closed” does not prove “object grasped.” Use available observations such as object motion, grasp contact, or task-state estimates, while documenting their limitations. A privileged simulator predicate is useful for evaluation; a deployed recovery trigger must use information actually available on the robot.

Log each transition with timestamp, request ID, model revision, observation reference, reason, and human action if any. Human correction data can support later learning, but it changes the data distribution and annotation requirements. Record whether a demonstration is uninterrupted expert behavior, a recovery from learner failure, or an intervention that overrides the model.

## Ground language in permitted operations

A task planner can output a structured proposal such as a known skill, target object, and parameter bounds. Validate the proposal before dispatch. Free text that resembles a command is insufficient to authorize physical execution. Reject unknown skills and schema changes; resolve ambiguity through the task interface or a supervising operator.

For a hierarchical VLA, verify both the high-level plan and low-level execution. A correct plan can fail through poor grasping; a capable controller can successfully execute the wrong target. Evaluation must retain this distinction. Language-driven recovery also needs a stale-plan rule when the environment changes.

## Guided fault-injection lab

Start from the runnable [dataset/dispatch replay](Dataset-and-Execution-Lab.md), then the trained-policy [timed robot loop](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md). The latter distinguishes commands accepted from commands actually applied and includes a case with terminal successes but no applied model commands. A fallback-assisted terminal success must not become evidence that the policy controlled the task. Both are synthetic environments, not physical supervisor validation. Create offline fixtures for a NaN action, swapped schema identifier, old reset epoch, sensor timeout, duplicate result, unreachable target, and a repeated unsuccessful grasp. Each fixture should produce a deterministic transition and a reason code. Do not connect the fixture runner to actuators.

Then add a synthetic uncertainty score correlated imperfectly with failures. Sweep an acceptance threshold and plot risk, coverage, and intervention demand. Include a counterexample in which the score is confidently wrong. The goal is to expose what your gate can and cannot detect, not to manufacture a perfect score.

The required artifact is a supervisor diagram, transition table, executable offline fixture tests, and a one-page operating-envelope document. An advanced Isaac exercise injects missing camera frames and delayed inference while recording the controller response. Physical testing belongs to a separate supervised integration process using the selected device's procedures.

## Failure clinic

Frequent unexplained stops often indicate thresholds chosen without measuring normal variation. Rare silent failures indicate gaps between monitored signals and actual task hazards. Rising intervention rates after an optimization can reveal a deployment regression even when average task success looks unchanged. A recovery loop that succeeds by ignoring the original instruction is not successful recovery.

## Self-check and answer guidance

1. Does low model entropy establish a safe action? No; calibration can fail under shift, and the score does not encode all physical constraints.
2. Why report coverage with accepted success? Selective rejection can inflate success while reducing useful autonomous work.
3. What belongs in the retry state? Failure stage, current observation, attempt count, relevant history, and the allowed recovery actions.
4. Why keep command validation separate from the model? Model errors, crashes, and malformed outputs still require defined handling by independently testable components.

---

[Previous](Lecture-23.md) · [Lab guide](Lab-Guide.md) · [Next](Lecture-25.md)
