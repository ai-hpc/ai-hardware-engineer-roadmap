# Robotics AI engineering: explain the whole loop

Practice turning a robot failure into a testable explanation. These are original
engineering exercises, not reports of a particular employer's interview process.
All numbers below are illustrative inputs or calculations, not robot benchmarks.

**Prerequisites:** [Robotics Foundations](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Robotics%20Foundations/README.md)
and [AI evaluation](../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/02-Evaluation-and-Robot-Learning.md).
**Artifact:** a ten-minute design explanation, one timestamped trace, and a small
reproducible failure case from your own work.

## How to practice

For each scenario, spend five minutes on your own answer before opening the
guidance. State the assumptions, do the calculation, and propose one experiment.
Use the follow-up to test whether you understand the reasoning beyond the numbers.
Two sessions of 60–90 minutes are enough for an initial pass; return after a lab.

## 1. The robot reaches to the wrong side of a table

The robot base is at `(1, 2)` metres in the world and rotated +90° about world z.
A point is `(0.5, 0)` metres in the base frame. Where is it in the world?
Why would adding the translation directly to the point be wrong?

<details markdown="1"><summary>Answer guidance</summary>

Rotate first: `(0.5, 0)` becomes `(0, 0.5)`. Then translate: the world point is
`(1, 2.5)`. The expression is `p_world = R_world_base p_base + t_world_base`.
Direct addition would give `(1.5, 2)`, treating base-frame components as if they
were already expressed in world axes.

Check a known point and a transform/inverse round trip. Then check frame names,
metres versus millimetres, angle units, and the time at which the transform is
valid. A correct transform at the wrong time can still place a moving object
incorrectly.

**Follow-up:** the point comes from a camera whose optical axes differ from the
base axes. Add the camera-to-base transform in the correct composition order;
do not repair this by swapping coordinates without stating the frame convention.

</details>

## 2. A 20 ms model still produces 44 ms old actions

All timestamps use one synchronized clock. A camera observation is stamped at
1000 ms, reaches the application at 1008 ms, completes preprocessing at 1010 ms,
completes model inference at 1030 ms, reaches a controller tick at 1040 ms, and
takes effect at 1044 ms. Define the relevant latency measurements.

<details markdown="1"><summary>Answer guidance</summary>

Model execution takes `1030 − 1010 = 20 ms`. Application-arrival-to-actuation
takes `1044 − 1008 = 36 ms`. Observation age at actuation is
`1044 − 1000 = 44 ms`. The last is what describes how old this action's input is.
Reducing model time cannot automatically remove capture/transport delay,
controller alignment, or actuation delay.

Trace observation IDs through every stage. Record timestamp meaning and clock
domain, then measure tail latency under the actual sensor and model workload.
If the camera stamp marks the beginning of exposure, say so; another timestamp
definition produces a different age.

**Follow-up:** halve model execution to 10 ms. The result depends on whether the
earlier completion catches an earlier controller tick. Draw the timeline before
promising a 10 ms reduction in observed behavior.

</details>

## 3. The policy predicts demonstrations well but drifts in a rollout

You have 20 recorded episodes with 100 adjacent frames each. Randomly splitting
all 2,000 frames gives a low test action error. The robot nevertheless drifts away
from the demonstrated path. Explain two distinct problems and how to test them.

<details markdown="1"><summary>Answer guidance</summary>

First, neighboring frames from one episode share scene and trajectory information.
A random frame split can put near-duplicates on both sides of the evaluation.
Split by episode and, where relevant, by scene, object, operator, or collection
session. Fit normalization only on training data. A simple initial split could
use 12 training, 4 validation, and 4 test episodes, but four test episodes offer
little precision and should not support a broad generalization claim.

Second, behavior cloning predicts actions on demonstrated states. Once a small
error moves the robot elsewhere, the policy may encounter states absent from the
demonstrations. Evaluate closed-loop rollouts with controlled perturbations,
compare a simple controller, and inspect where trajectories first diverge.
Better data coverage or corrective demonstrations may help; only an experiment
can establish the improvement for this task.

**Follow-up:** keep the same split but add camera bias. Distinguish a robustness
test from an untouched estimate of nominal performance.

</details>

## 4. Success increased from 80 to 84 out of 100 trials

Can you conclude the new policy is better? What additional evidence would you ask
for before changing the deployed model?

<details markdown="1"><summary>Answer guidance</summary>

A four-point observed increase is not enough on its own. For 80 successes in 100
independent Bernoulli trials, a 95% Wilson interval is approximately 71.1–86.7%.
That describes uncertainty in one success rate, not a confidence interval for the
difference between policies. Overlapping intervals are not a formal paired test.

Use the same held-out scenario IDs for both policies and retain per-scenario
outcomes. Report task strata, failures, uncertainty, intervention frequency, and
latency. Use paired analysis when the evaluation design is paired. Correlated
trials from the same scene require a suitable grouping strategy rather than
pretending each video frame is an independent trial.

**Follow-up:** the new policy wins on easy scenes and loses on rare crowded ones.
Report the strata and deployment requirements; a higher aggregate mean can hide
a regression that matters to the actual task.

</details>

## 5. A reliable message is too late to use

A command arrives successfully but was generated from an old camera frame. The
team proposes reliable transport as the fix. What belongs in the command
contract and the receiver?

<details markdown="1"><summary>Answer guidance</summary>

Reliable delivery does not establish freshness. Define the command's units,
frame, observation timestamp, validity interval, sequence, and episode/session
identity. The receiver checks finite values, permitted bounds, clock assumptions,
age, ordering, and whether the command belongs to the current session.

Expiry behavior must be explicit: stop, hold, or a task-specific fallback according
to the controller contract. Test stale, future-dated, reordered, duplicated, and
pre-reset commands. Queue length and transport reliability are separate decisions.

For actual middleware behavior, read the selected ROS distribution's
[QoS documentation](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Quality-of-Service-Settings.html)
and reproduce publisher/subscriber compatibility in the ROS lab.

**Follow-up:** the sender and receiver clocks differ by 100 ms. A timestamp
comparison needs a clock synchronization/error model; sequence numbers alone do
not recover physical observation age.

</details>

## 6. A path exists for a point, but the robot clips a doorway

The planner searches a free-cell grid and returns a route through a one-cell
opening. Why can that route be unusable? What does an inflated occupancy grid
approximate, and what does it leave out?

<details markdown="1"><summary>Answer guidance</summary>

A point path ignores the robot's footprint. Mark positions whose footprint
would overlap an obstacle as blocked, then plan the robot's reference point in
the remaining space. Inflation depends on footprint, map resolution, and chosen
clearance. A square or circular approximation can be conservative or inaccurate
for a rotating noncircular body.

The path still needs trajectory timing and checks for curvature, dynamics,
tracking error, and moving obstacles. An unreachable goal or no-path result is a
valid outcome, not a reason to disable collision checks.

**Follow-up:** add a long rectangular robot that must rotate in a corridor. A 2-D
position grid cannot express every orientation-dependent collision; explain what
state representation you would need next.

</details>

## 7. A larger VLA does not fit the edge computer

What would you measure before choosing quantization, offloading, or a new board?

<details markdown="1"><summary>Answer guidance</summary>

Identify the exact model and runtime revisions, input shapes, history/action
lengths, supported operators, weight storage, activation peaks, caches, workspace,
and nonmodel services. Trace preprocessing, model stages, transfers, and action
validation. Peak parameter memory alone does not establish that the application
fits.

Define a task-quality reference and a closed-loop acceptance criterion before
changing precision. Measure sustained timing and observation age under realistic
thermal and communication conditions. Offloading adds transport and availability
dependencies; a smaller local model may change task quality. Present the measured
tradeoff instead of treating TOPS or parameter count as the decision.

**Follow-up:** a numerical action-parity test passes but task success drops.
Investigate timing, accumulated deviations, and scenario coverage; numerical
agreement is one part of the evidence, not the whole robot evaluation.

</details>

## Final design exercise and self-check

Explain one of your own projects using this sequence: task, sensor/action
contract, simple baseline, learned policy if useful, timing trace, failure test,
and evidence supporting the next design decision. Let a peer double the sensor
rate or add one policy-period of delay. Revise your design aloud.

Score each dimension 0–2: correct calculation, explicit assumptions, useful
measurement, concrete failure test, and reproducible evidence. A score of 0 means
missing or incorrect; 1 means plausible but incomplete; 2 means justified with a
worked example or your own artifact. Revisit every zero before moving on.

[Interview overview](../README.md) · [Physical AI Design Studio](../../Physical-AI-Design-Studio.md)
