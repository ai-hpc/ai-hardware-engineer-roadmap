# CPU lab: carry a trained policy through the action-parity boundary

**Time:** 90–120 minutes. **Needs:** Python 3.12, the full course checkout and a
CPU; standard library only. First complete the [small-policy lab](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-01-CPU-Policy.md)
and [timed-loop lab](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md).

This lab imports those two existing implementations. It trains the same model,
keeps the episode split and selected parameters, then compares seven candidate
cases. It does not load a pretrained VLA, process images/language, install a
simulator or command a robot. Its timing values are imposed simulation inputs.

## 1. Predict the comparisons

The reference has two normalized features, three learned parameters, a bounded
acceleration output and a 20-second cart task. Read the unchanged contract in
[lecture 1](Lecture-01.md#4-preserve-arithmetic-before-changing-precision).

| Candidate | What changes? | Predict before running |
|---|---|---|
| `identity` | Nothing | What should every numerical difference be? |
| `folded_normalization` | Algebraically fold mean/scales into weights/bias | Should final floating-point bits necessarily match? |
| `int8_weight_arithmetic` | Round learned weights to one signed INT8 scale, then dequantize | How large is an error relative to half a scale step? |
| `missing_normalization_fault` | Feed physical inputs to weights trained for normalized features | Why is this not the same as folding normalization? |
| `same_weights_slow_fifo` | Same model; 120 ms inference and unchecked FIFO | Can replay action error reveal observation staleness? |
| `same_weights_slow_latest` | Same model; 120 ms inference, latest pending input, capture deadline | Must matching final success mean matching trajectory? |
| `same_weights_over_budget` | Same model; 350 ms inference exceeds 250 ms lifetime | Can the model deserve credit if no action is applied? |

The first four use the fast 10 ms/zero-transport setting. Slow settings use
20 ms transport; all share the same initial conditions, 10 ms physics timestep,
20 Hz sensor, outage, acceleration limit and independently observed velocity
fallback defined in the timed lab. A changed timing configuration is a system
candidate, not a numerical model optimization.

## 2. Execute and retain the report

From the repository root:

```bash
python3 'Phase 5 - Advanced Topics and Specialization/Track D - Robotics/VLA Optimization and Action-Parity Harness/labs/action_parity.py' --output /tmp/action-parity-report.json
python3 -m unittest discover -s 'Phase 5 - Advanced Topics and Specialization/Track D - Robotics/VLA Optimization and Action-Parity Harness/labs' -p 'test_*.py' -v
```

The script locates the Phase 3 modules relative to its own file, so its absolute
path also works from another directory. Keep the full course tree; the imports
are part of this connected exercise. Save reports you want to retain outside
`/tmp`.

**Actual run, 18 October 2026, Python 3.12.3:**

| Candidate | Max raw action difference, m/s² | Task successes | Fixed fixture gate |
|---|---:|---:|---|
| Identity | 0 | 12/12 | PASS |
| Folded normalization | `1.11e-16` | 12/12 | PASS |
| INT8 weight arithmetic | `0.00162145` | 12/12 | PASS |
| Missing normalization | `0.465194` | 5/12 | FAIL |
| Same weights, slow FIFO | 0 | 0/12 | FAIL |
| Same weights, slow latest | 0 | 12/12 | FAIL: trajectory difference |
| Same weights, over budget | 0 | 2/12 | FAIL: no policy actions applied, lost successes and trajectory difference |

Every twelve-episode population gate is **INCONCLUSIVE** at the declared
one-sided 95% confidence and two-percentage-point noninferiority margin.
That includes the identity regression: the fixed observations and episodes are
not a population proof. These results are retained in [sample-run.json](labs/sample-run.json),
with checks and source hashes in [validation.json](labs/validation.json).

## 3. Read the layers of evidence

The report's `reference_tape_frames` is 180 held-out demonstration observations.
`numerics` compares candidate actions against the reference actions on those
same inputs; it is not the error against teacher labels. The latter appears
separately as `held_out_teacher_rmse_m_s2`, approximately `0.02193`.

The rollouts use twelve fresh paired initial states from seed 501, the same
seed as the timing lesson. `max_position_divergence_m` compares complete
trajectories at identical simulation timestamps. `successes` checks final
position and velocity, so a candidate can pass that condition while moving
along a different route. The report includes every episode's outcome and
applied-command metrics.

The quantized candidate stores `integers` and a scale. It multiplies the
integers by that scale, then predicts in Python floating point. This is an
arithmetic perturbation experiment, **not an optimized INT8 runtime**. The
folded candidate removes normalization operations algebraically; this tiny
script does not claim a measured inference speedup for either candidate.

## 4. Exercise the actual gate

Run a passing fixed-fixture candidate:

```bash
python3 'Phase 5 - Advanced Topics and Specialization/Track D - Robotics/VLA Optimization and Action-Parity Harness/labs/action_parity.py' --gate-candidate folded_normalization --output /tmp/parity-pass.json
```

It exits 0. Then run the deliberately wrong normalization:

```bash
python3 'Phase 5 - Advanced Topics and Specialization/Track D - Robotics/VLA Optimization and Action-Parity Harness/labs/action_parity.py' --gate-candidate missing_normalization_fault --output /tmp/parity-fail.json
```

It writes the report and exits **2**, intentionally. Inspect `failure_reasons`;
do not repair the exercise by weakening its tolerance. Running without a gate
name merely generates a report and exits 0, even when the report includes failed
candidates. Neither command grades a physical deployment.

## 5. Inspect one function per question

| Function in [action_parity.py](labs/action_parity.py) | Reason to inspect it |
|---|---|
| `fold_normalization` | Derive the changed bias term by hand; dropping it is not equivalent |
| `make_tape`, `replay_metrics` | Match identities/contracts and reject missing, duplicate or nonfinite data |
| `quantized_weights` | Understand rounding, scale and actual computation dtype |
| `paired_outcomes` | Count harmed/helped pairs rather than only aggregate success |
| `binomial_upper` | Reproduce the exact zero-harm bound and two-trial hand example |
| `run` | See actual imported training and timed simulation, fixed initial states and separate gates |

The tests check hand-computed arithmetic, hidden raw errors after clipping,
contract mismatches, empty/missing/duplicate data, nonfinite/overflow rejection,
exact confidence-bound examples and the actual trained-policy integration.
They deliberately do not label invented Boolean outcomes as robot experiments.

## 6. Change one thing

Choose one variation, working in a copy of this course:

1. **Numerical:** replace 127 in `quantized_weights` with 7 for a signed
   `[-7,7]` rounding fixture. Predict action error and trajectory change. Keep
   all acceptance budgets unchanged; report both passing and failing cases.
2. **Timing:** change only the slow-latest transport from 20 to 60 ms. Predict
   which commands expire, then compare applied intervals, trajectories and
   success counts. A fresh result can expire while being held.
3. **Contract fault:** use `dataclasses.replace(CONTRACT, action_unit='cm/s^2')`
   as the candidate contract in a replay comparison. Even identical numeric
   arrays must be rejected. Explain why merely relabeling units does not convert
   a model.

Do not retune thresholds on these test results and still call the original
suite held out. If selecting an improved candidate from variations, treat these
results as development evidence and reserve new evaluation episodes.

**Finish:** retain the baseline and variation reports, one hand calculation, a
caught fault, an explanation of the over-budget model-credit trap, and the
strongest claim your evidence actually supports. For a pretrained VLA extension,
replace the policy adapter and environment deliberately while preserving
identity, normalization, reset, timing and outcome checks.
