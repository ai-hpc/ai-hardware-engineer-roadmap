# Lecture 08 — Energy, task quality, and honest comparisons

**Previous:** [Scheduling](Lecture-07.md) · **Next:** [Prototype to silicon](Lecture-09.md) · **Practice:** [CPU labs](Lab-Guide.md)

**Prerequisites:** a fixed task suite, a reference policy, and observation/action timestamps. **Deliverable:** a quality-constrained comparison with raw energy, latency, and paired task outcomes.

## 1. Define the question before the experiment

Run two different comparisons and label them:

| Comparison | Keep fixed | What the result can support |
|---|---|---|
| Hardware/runtime | Checkpoint, numerical contract, observation/action settings, data, task, quality criteria | Benefit of mapping the same computation onto a different implementation |
| Co-designed system | Task distribution, quality/latency requirements, system power boundary | Benefit of a changed model/algorithm plus the proposed hardware |

Changing denoising steps, action horizon, resolution, precision, and hardware at once cannot isolate a silicon advantage. Exact bitwise results may be inappropriate across floating-point implementations; declare numerical tolerances and behavior requirements before timing.

Use the [VLA action-parity course](../../Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) for policy-level evaluation. The present course adds the accelerator and energy boundaries.

## 2. Power boundaries determine the meaning of efficiency

Record where energy is measured: accelerator rails, module input, full compute system, or wall socket. Include host CPU, DRAM, interconnect, and required sensor preprocessing consistently. Actuator energy can be important to a product, but must be reported separately if the claim concerns compute-system efficiency.

For samples of power `P_i` at timestamps `t_i`, trapezoidal integration is:

```text
E_joules ≈ sum(0.5 * (P_i + P_(i+1)) * (t_(i+1) - t_i))
```

Use seconds and watts, record sensor accuracy/sample rate, and synchronize the measurement window with attempts. A module's configured wattage limit is not its measured energy. Multiplying a short kernel duration by advertised TDP does not establish energy per inference.

Report gross energy first. An optional incremental-energy estimate may subtract a stated idle baseline, but include its method and uncertainty; idle subtraction can obscure required host/platform costs.

## 3. Worked example: lower energy per attempt can fail to improve the product

A baseline uses 15 J on each of three attempts and succeeds on all three. A candidate uses 10, 12, and 8 J, with the middle attempt failing.

| Quantity | Baseline | Candidate |
|---|---:|---:|
| Total compute energy | 45 J | 30 J |
| Successful attempts | 3 | 2 |
| Energy per attempt | 15 J | 10 J |
| Energy per successful task | 15 J | 15 J |

The candidate saves energy per attempt but has no advantage on this energy-per-success metric and fails the baseline's observed task quality. Three attempts are a teaching fixture, not enough to establish a population success rate.

In general:

```text
energy_per_success = total energy across all attempts / successful attempts
```

Include failures, timeouts, retries, and interventions according to rules declared before collection. If there are no successes, the ratio is undefined; do not report zero energy per success. Also report energy per policy evaluation and task duration, because longer tasks can hide a faster or slower inference path.

## 4. Timing needs more than one column

Collect per-action timing independently of per-episode outcomes:

| Field | Reason |
|---|---|
| Source observation ID/time and clock | Measures freshness and connects to sensor data |
| Queue, model, and usable-output times | Separates service time from waiting and validation |
| Consumption time | Detects a late action even if inference completed quickly |
| Chunk ID, index, interval, and expiry | Distinguishes playback frequency from fresh replanning |
| Warm/cold, thermal state, concurrent load | States the conditions that affect timing |
| Deadline miss/drop/fallback reason | Keeps rejected work in the evidence |

Report sample count, median, tail percentiles, maximum observed latency, misses, and observation age. State the quantile estimator. A p99 from a tiny pilot is unstable; the CPU fixture intentionally makes that limitation visible.

## 5. Quality is a paired closed-loop experiment

Use matched tasks, initial states, scene/seed IDs, timeouts, and success predicates. Save the candidate and reference outcome for each pair. After policies choose different actions, their observations can diverge: the experiment is paired by initial condition, not by assuming identical future frames.

Declare an allowed quality loss `delta` and a statistical procedure before running the final test. For success-rate difference `candidate - reference`, a noninferiority claim needs a suitably defined lower confidence bound above `-delta`. A point estimate within the tolerance is insufficient. State whether the interval is one-sided or two-sided and how tasks, seeds, or collection sessions are resampled; correlated frames are not independent trials.

**Worked decision:** permitted loss is 2 percentage points. The estimate is -1 point, but the interval is [-5, +3]. The result does not establish noninferiority because the plausible loss exceeds the permitted budget. It also does not prove the candidate is worse. Preserve this as inconclusive evidence rather than selecting a more favorable subset after the fact.

Use [the VLA evaluation lecture](../../Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Lecture-19.md) and [paired CPU exercises](../../Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Lab-Guide.md) for the statistical implementation. Check long-horizon tasks, changed lighting, occlusion, disturbance recovery, and stale observations according to the intended operating domain.

## 6. An ablation matrix that explains the result

Build the comparison in stages:

1. Reference policy on a tuned existing platform.
2. Same policy on the proposed mapping, preserving the numerical contract.
3. One precision change on both mappings where supported.
4. One iteration/reuse change with its own behavioral evaluation.
5. Combined candidate, measured again rather than multiplying individual speedups.

For each row retain quality, energy, full-path timing, memory peak, rejected requests, and unsupported operators. Report kernel-only measurements separately. Interleave paired performance runs where practical to reduce drift from temperature or clock state. A vendor table remains attributed context until its configuration is independently reproduced.

The proposed **2× energy-efficiency target** means half as much energy per successful task at matched quality and required latency. It is a design objective, not a conclusion from a lower-bound model. Several policy families should test the claim; do not average away a failure on one required workload.

## Lab and acceptance

Run the `evaluate` CPU fixture. Derive its 30 J total, 15 J per success, one deadline miss, and one stale-consumption event. Explain why the same event can belong to more than one failure category.

**Negative controls:** set every outcome to failure; reverse a timestamp; duplicate an episode ID. Confirm that undefined efficiency and malformed records cannot silently become a favorable benchmark.

**Extension:** collect a small pilot on your available CPU/GPU with a real power boundary. Save per-action traces separately from episode energy, and use the pilot to estimate the budget for a larger evaluation. Do not label the fixture a model benchmark.

**Acceptance:** the comparison identifies what changed, preserves raw failed attempts, declares uncertainty, and states which product requirement passed or remains unproven.

## Self-check with answers

1. **Why is energy per kernel insufficient?** The product may spend more energy on preprocessing, transfers, waiting, the host, or repeated failed tasks.
2. **Does close action MSE establish task parity?** No. Errors can accumulate in closed loop, and a small average can conceal a critical action dimension.
3. **When is a 2× figure misleading?** When measurement boundaries, quality, workloads, or inference settings differ without an explicit system-level comparison.
4. **Can an inconclusive result pass the course?** Yes, with a correct protocol, reproducible evidence, and a justified next experiment. It cannot pass the product's unproven quality gate.
