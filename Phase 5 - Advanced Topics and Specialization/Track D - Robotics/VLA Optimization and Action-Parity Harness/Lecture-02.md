# 2. Build an action-parity gate that says what it knows

**Purpose:** separate a numerical regression, a task result and a statistical
claim. **Before:** [optimization boundaries](Lecture-01.md). **Run next:**
[the connected CPU lab](CPU-Action-Parity-Lab.md).

A harness compares a reference and candidate under a declared contract. It is
not a proof of physical safety, and its reference can be wrong. Verify the task
and the reference first. Report **pass**, **fail**, **inconclusive**, and **not
run** distinctly; a skipped required check must not become a release pass.

## 1. Preserve the complete policy contract

A useful adapter needs more than `step(observation) -> array`. Stateful policies
also need episode reset, session identity, timestamp handling, random-state or
noise control, and action-chunk consumption rules. Record these even when the
current policy is stateless. A cached chunk must not survive an episode reset.

| Boundary | Fields that must agree |
|---|---|
| Observations | Episode/step identity, raw sample, preprocessing revision, feature/camera order, units, frame and capture clock |
| Policy | Weight/checkpoint identity, normalization, action decoder, sampling configuration, reset/history state |
| Returned command | Semantics, physical units, frame/joint order, chunk shape/spacing, source capture time and expiry |
| Consuming system | Controller period, clipping/rate limits, fallback, simulation timestep and integration order |

Hash both raw input identity and the transformed model inputs when feasible.
A hash match establishes byte identity under the serialization you used; it
does not establish correct calibration, semantics or freshness. A mismatch
means the comparison boundary differs. It can be a capture, preprocessing,
ordering or manifest problem; it does not uniquely diagnose a preprocessing bug.

For stochastic policies, freeze noise draws where the API supports them for a
controlled diagnostic. Also evaluate the intended sampling distribution. Seeds
do not guarantee cross-backend bit identity, and replacing sampling with argmax
changes the policy. Reference variability is something to characterize, not a
number to multiply by an arbitrary factor and call statistical equivalence.

## 2. Compare decoded actions with their units

On a frozen observation tape, compare the same episode and sample identities.
Reject missing, duplicate and nonfinite records; do not silently compare the
shorter of two lists. The CPU bridge implements these checks.

For scalar acceleration, errors `+0.2` and `−0.2 m/s²` have zero signed mean but
MSE `(0.04+0.04)/2 = 0.04 m²/s⁴` and maximum absolute error `0.2 m/s²`.
Signed bias, RMSE and maximum error answer different questions. A 99th
percentile is not the maximum, and neither alone establishes safety.

Compare raw predictions and the actions after decoding/limits. If the reference
requests `2 m/s²`, the candidate `3 m/s²`, and both are clipped to `1 m/s²`,
applied-action MSE is zero while raw-action MSE is one. Clipping has hidden a
large model difference that may matter under another limit or state.

Use metrics suited to the action representation:

- Translation errors have metres or metres/second units according to the
  contract. Their squared errors have squared units.
- Orientations need a representation-aware distance. Yaw `+179°` versus `−179°`
  differs by 2° on the circle, not 358°. For general 3D rotations use relative
  rotation angle; do not subtract arbitrary Euler triples.
- Gripper commands may be discrete. Values `0.499` and `0.501` straddle a
  threshold of 0.5 despite tiny squared error. Count open/close disagreement and
  compare the task consequences.
- Token KL is optional and requires aligned token distributions, masks and
  conditioning history. It does not apply to every continuous action head and
  is not a universal early-warning metric.

Fix tolerances before inspecting candidate scores. Derive them from the
numerical approximation and task/controller sensitivity; label educational
thresholds as such. The CPU lab's `0.01 m/s²` maximum action difference and
`0.02 m` trajectory difference are **fixture regression budgets**, not universal
robot tolerances.

## 3. Distinguish replay from a rollout

**Saved-input replay:** both policies see the same saved observations. Their
outputs do not change the next recorded observation. This isolates numerical
and adapter differences.

**Open-loop action playback:** send each fixed action sequence through a
specified dynamics/controller model. This reveals how that sequence moves the
model, but does not let the policy react to its new states.

**Closed-loop rollout:** each policy receives observations from its own evolving
environment. Pair initial conditions, simulator settings and relevant random
streams; the subsequent observations should usually diverge after different
actions. Requiring their hashes to remain equal would invalidate this test.

Do not use forward kinematics as a universal state integrator. The correct
update depends on the action:

```text
joint velocity: q_next = q + qdot * dt; tool_pose = FK(q_next)
joint position target: controller/plant tracks q_target over time
cart acceleration: v_next = v + a*dt; x_next = x + v_next*dt
Cartesian displacement: apply the specified frame/composition rule, then solve/control
```

The first line neglects joint dynamics and is appropriate only for the stated
kinematic model. None models contact merely by adding an `FK` call.

**Worked failure:** the CPU bridge's unchanged model with FIFO delay has zero
saved-input action difference, but its own outdated commands change its path
and future observations. All twelve reference successes are lost. With
newest-only pending work it regains twelve terminal successes, yet exceeds the
fixed suite's 2 cm trajectory-difference budget. A bounded trajectory or equal
success count is not automatically equivalent behavior.

## 4. Grade timing and model participation separately

For each applied interval, retain capture, delivery, inference start/end,
command application and expiry. Compare timestamps in the same clock domain.
Checking only completion age misses a command held beyond its lifetime.

The [timed-policy environment](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md)
checks the entire integration interval. A result can arrive exactly at its
capture deadline, pass message acceptance, and still apply to zero subsequent
intervals. Record `distinct_commands_applied` and `applied_policy_intervals`,
as well as accepted/rejected counts.

When all 350 ms results expire under a 250 ms budget, the fixture applies no
policy actions. Two episodes still satisfy its terminal condition because of
initial conditions and the independent damping fallback. Crediting those two
successes to the policy would misstate what happened.

For real target timing, use sustained workloads, queue/thermal conditions and a
stated clock boundary. The CPU lab's 10/120/350 ms values are imposed parameters,
not benchmarks of the model or an accelerator.

## 5. Paired outcomes and noninferiority

Use the same episode identifiers for both policies. Count four outcomes:

| Reference | Candidate | Interpretation |
|---|---|---|
| Success | Success | Shared success |
| Failure | Failure | Shared failure |
| Success | Failure | Harmed pair: candidate loses a reference success |
| Failure | Success | Helped pair: candidate gains a success |

The observed success-rate difference is `(helped - harmed)/N`. Equal aggregate
rates can hide a swap of which episodes fail. Retain failure causes, task groups,
trajectory constraints and time-to-success. Report timeouts alongside completion
times; comparing times only among successes can select different subsets.

A **noninferiority** claim states that the population difference is greater
than a prespecified negative margin, such as `−0.02` (two percentage points).
A nonsignificant difference test does not establish that claim. Ten inconclusive
trials do not become evidence of equivalence because `p > 0.05`.

### An exact, conservative bound you can calculate

Let `H` be the probability of a harmed pair and `G` that of a helped pair.
The success difference is `G-H >= -H`. An upper confidence bound on `H` therefore
gives a conservative lower bound on the difference. This ignores beneficial
pairs and can be less powerful than a dedicated paired analysis, but its
assumptions and arithmetic are easy to inspect.

For **zero harmed pairs in N independent paired episodes**, the one-sided 95%
exact upper bound solves `(1-H)^N = 0.05`:

```text
H_upper = 1 - 0.05**(1/N)
N = 12  -> 22.09% upper bound
N = 50  ->  5.82% upper bound
N = 150 ->  1.98% upper bound
```

Thus zero losses in twelve trials does not establish a two-percentage-point
margin. For nonzero harmed counts, the lab inverts the binomial CDF to compute
the one-sided Clopper–Pearson upper bound. Its numerical tests include the
independent solution for one harmed pair in two trials:
`P(X<=1)=1-H²=0.05`, giving `H_upper=sqrt(0.95)`.
[NIST exact binomial confidence limits](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm).

This is a claim about independent episodes from the specified distribution.
Repeated frames, repeated deterministic trials, or 150 invented booleans do not
supply 150 independent physical experiments. Choosing the best of many candidates
using these outcomes also changes the inference: use validation for selection,
then fresh final episodes or an appropriate multiple-comparison procedure.

### Why there is no universal episode count

For one success proportion near 0.8, the normal-approximation 95% half-width at
N=400 is `1.96*sqrt(0.8*0.2/400)=0.0392`, about **3.9 percentage points**. Even
one proportion needs roughly 1,537 independent trials for an approximate 2-point
half-width under those assumptions. Two independent proportions have additional
variance; a paired design depends on discordant outcomes. These are planning
calculations, not universal minimum counts or exact final intervals.

The bridge therefore reports **fixed-fixture PASS** separately from
**population INCONCLUSIVE**. Its population calculation is deliberately
conservative. It does not pretend that the fixed twelve-case regression suite
certifies a VLA or real robot.

## 6. A concrete gate and its scope

The implemented command is:

```bash
python3 'Phase 5 - Advanced Topics and Specialization/Track D - Robotics/VLA Optimization and Action-Parity Harness/labs/action_parity.py' --gate-candidate folded_normalization --output /tmp/parity-folded.json
```

Exit 0 means this candidate passed the **declared CPU fixture regression**.
The report still says the two-point population claim is inconclusive. This is
an intentional distinction, not an ignored failed release gate. A deployment
pipeline that requires population noninferiority must require that additional
result and all target checks explicitly.

The fixture rejects missing normalization, excessive applied age, a policy
that never contributes an action, lost paired successes, and excessive
trajectory divergence. Unknown candidate names and invalid counts fail the
CLI. A run without `--gate-candidate` generates a comparison report and exits
successfully even when some candidates fail; it is not a promotion command.

## 7. Move to a supported task environment

Use a benchmark only with the policy's matching task adapter and trained or
fine-tuned checkpoint. Freeze the simulator, assets, camera definitions, action
rate, controller, reset protocol and task success definition. LIBERO, RoboCasa,
and Isaac Lab supply different environments and contracts; an Open X-Embodiment
dataset is not an automatically installed Isaac Lab manipulation suite.
[LIBERO source](https://github.com/Lifelong-Robot-Learning/LIBERO),
[RoboCasa source](https://github.com/robocasa/robocasa),
[Isaac Lab environment catalog](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html).

Simulator success is evidence about that model and task. Physical evaluation
adds sensing, actuator, calibration, contact and scheduling differences. Plan it
with the actual platform's qualified controller, limits, stop/recovery behavior
and operator procedure. No Python parity flag can certify those mechanisms.
This course's executable lab has no physical command endpoint.

## Practice and handoff

Run the [CPU lab](CPU-Action-Parity-Lab.md), then explain three cases:

1. Why does normalization folding preserve the mathematical function while
   simply removing normalization fails?
2. How can zero replay MSE coexist with zero task successes?
3. How can 12/12 matched successes coexist with both a trajectory failure and
   an inconclusive population claim?

For independent transfer, choose the numerical or timing change in the lab,
write its expected failure before running, and retain all candidate outcomes.
Hand over the immutable reference/contract, candidate description, raw paired
results, gate decision and limitations. That is the starting point for a
pretrained-policy experiment, not a substitute for one.
