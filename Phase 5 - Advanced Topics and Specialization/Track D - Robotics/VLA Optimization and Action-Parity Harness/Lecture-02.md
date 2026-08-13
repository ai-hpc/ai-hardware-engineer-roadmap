# Lecture 2: The Action-Parity Harness

## Overview

**Optimization without measurement is folklore.** Lecture 1 listed eight rungs of VLA optimization; every one of them can **silently destroy a policy**. This lecture is the measurement framework that decides what is safe to ship.

A useful action-parity harness answers four questions, in order, and refuses to answer the next one until the previous one passes:

1. **Per-tick parity** — given the same observation, does the candidate policy emit the same action as the reference, within a tolerance you can defend?
2. **Trajectory parity** — over an open-loop replay of logged observations, does the candidate's action sequence stay close to the reference?
3. **Closed-loop sim parity** — when the candidate actually drives the simulator, does it reach the same task success rate as the reference?
4. **On-robot parity** — does the candidate behave the same on the real robot, under canary protection, across a held-out task set?

Per-tick parity is **cheap and necessary but not sufficient**. Closed-loop sim parity is what catches the failures Lecture 1's quantization can cause. On-robot parity is what catches the failures that **sim cannot model**.

By the end of this lecture you should be able to:

* implement a deterministic replay loop that feeds the same observation tape into N candidate policies and emits a per-step diff
* set per-axis tolerance budgets for action MSE, and explain why those budgets came from the *robot*, not the model
* run a closed-loop sim parity sweep on LIBERO / RoboCasa / Isaac Lab with a fixed seed list
* write a CI gate that another engineer can run on a new checkpoint and get a pass/fail
* design a 50-episode on-robot canary that fails *closed* without breaking hardware

---

## 1. The mental model: reference, candidate, harness

```text
                ┌──────────────────────┐
   observation  │   Reference policy   │  ──►  a_ref(t)
   tape  ─────► │   (fp16/bf16, full)  │
                └──────────────────────┘
                                                ┌──────────────┐
                ┌──────────────────────┐        │              │
                │   Candidate A        │  ──►   │              │
                │   (TRT + INT4)       │        │   Harness    │  ──►  parity_report.csv
                └──────────────────────┘        │              │
                                                │  per-step Δ  │
                ┌──────────────────────┐        │  trajectory  │
                │   Candidate B        │  ──►   │  closed-loop │
                │   (INT4 + FP8 KV)    │        │  sim rollout │
                └──────────────────────┘        │              │
                                                └──────────────┘
                                                        │
                                                        ▼
                                                 pass / fail gate
```

Three principles:

* **The reference is the policy you would actually ship if compute were free.** Usually fp16/bf16 of the published checkpoint. Pin its exact weights, exact tokenizer, exact preprocessing pipeline. The reference is the contract.
* **Determinism is non-optional.** Disable cuDNN nondeterminism, fix all RNG seeds, fix sampling temperature (or replace sampling with argmax), pin CUDA Graph capture, log every randomness source. If two runs of the reference disagree, the harness is broken.
* **Every metric has a budget *the roboticist* defines, not you.** "0.01 rad of joint error" is not inherently good or bad; the gripper's task tolerance is the answer.

---

## 2. Per-tick parity (Question 1)

The cheapest, most useful, most over-trusted check.

### 2.1 What the harness records

For each tick `t` in a frozen observation tape:

| Field | Type | Why |
|-------|------|-----|
| `t` | int | tick index |
| `obs_hash` | sha256 | guarantees both policies saw the same input |
| `a_ref[t]` | float32[A] | reference action, length = action dim |
| `a_cand[t]` | float32[A] | candidate action |
| `Δ[t] = a_cand[t] − a_ref[t]` | float32[A] | per-axis error |
| `logits_kl[t]` (optional) | float32 | KL of action-token distributions if tokenized |
| `latency_ms[t]` | float32 | candidate wall-clock |

The harness must reject any tick where `obs_hash` differs between runs. If hashes disagree, you have a preprocessing bug, not a model bug, and per-tick parity is meaningless until you fix it.

### 2.2 Metrics

* **Action MSE, per-axis:** `mean(Δ[:, k]^2)` for each axis `k`. The per-axis breakdown matters — a model that is great on translation but bad on gripper-close is a different deployment problem from one that drifts uniformly.
* **Action max-error, per-axis:** `max(|Δ[:, k]|)`. P99 is more useful than mean for safety arguments.
* **Logit KL** (only if the action head is tokenized): `mean(KL(p_ref || p_cand))`. The most sensitive early-warning of backbone quantization damage.
* **Latency P50 / P95 / P99:** required for the deployment story; also used to reject candidates that pass parity but blow the control budget.

### 2.3 Tolerance budgets

This is the table the harness uses as a pass/fail gate. The numbers below are *examples for a 7-DoF arm doing tabletop manipulation* — your robot's numbers will differ.

| Axis | Unit | MSE budget | P99 max budget | Source of the number |
|------|------|------------|-----------------|----------------------|
| x, y, z (end-effector) | m | 1e-5 (≈3 mm RMS) | 8e-3 (8 mm) | gripper finger half-width minus object margin |
| roll, pitch, yaw | rad | 1e-4 (≈0.6° RMS) | 5e-2 (≈3°) | orientation tolerance of typical grasp |
| gripper | normalized [0, 1] | 1e-4 | 5e-2 | threshold beyond which "open" becomes "close" |
| action-token logit KL | nats | 0.01 mean, 0.05 P99 | — | empirical — values above this correlate with rollout drift |

How to derive *your* budgets:

1. Run the reference policy twice with different RNG seeds and the same observation tape. The non-determinism of the reference itself is your *floor*.
2. Multiply that floor by 3-5× to get a "candidate is statistically indistinguishable from reference" budget.
3. Independently, ask the robot integrator the maximum joint / pose error that the task can tolerate (the *physical* budget).
4. Take the **min** of the statistical budget and the physical budget. That is your pass threshold.

If your candidate passes step 4 budgets, it has passed Question 1. It has not yet earned the right to drive a robot.

---

## 3. Trajectory parity (Question 2)

Per-tick parity is misleading because **tick errors compound**. The same 5 mm per-tick error can integrate to 5 cm or to 5 mm depending on whether the policy **self-corrects**.

### 3.1 Open-loop replay

Feed the candidate the same observation tape as the reference, tick by tick, but accumulate a synthetic state estimate by integrating its actions:

```text
state_cand(t+1) = forward_kinematics( state_cand(t) + a_cand(t) * dt )
state_ref(t+1)  = forward_kinematics( state_ref(t)  + a_ref(t)  * dt )

drift(t) = || state_cand(t) − state_ref(t) ||
```

Plot `drift(t)` per episode. The shape of the curve matters more than its peak:

* **Bounded drift**: the candidate is statistically equivalent. Ship it.
* **Linear drift**: there is a per-axis bias in the candidate. Re-investigate per-axis MSE; the bias was hiding under a small mean.
* **Late-spike drift**: the candidate is fine on the cruise phase and fails on the precise phase (final approach, grasp closure). This is the most common quantization failure mode. The fix is usually keeping the action head at higher precision.

### 3.2 Tolerance

Budget the same way as in §2.3: take the reference-vs-reference drift envelope and multiply by 3-5×; intersect with the physical task envelope (e.g. "the end-effector must be within 1 cm of where the reference policy would have placed it at the same task phase").

### 3.3 Why open-loop is not enough

Open-loop replay assumes the candidate's actions **do not change what it would see next tick**. In a real rollout, every action changes the observation, and small per-tick errors get **amplified by closed-loop dynamics**. That is Question 3.

---

## 4. Closed-loop sim parity (Question 3)

The metric that actually predicts whether your optimized policy will pick up the cup.

### 4.1 Sim choice

The harness should support at least one of:

* **LIBERO** (Spatial / Object / Goal / Long) — fast, well-defined success criteria, the de facto eval for OpenVLA-class policies.
* **RoboCasa** — kitchen-scale tasks, useful for π₀-class policies.
* **Isaac Lab** with the OpenX manipulation suite — slower but matches real-robot dynamics better, supports domain randomization.

Whichever you pick, **freeze the sim version, asset set, and seed list**. The harness commits these to the repo as a manifest file. A "parity report" generated against an unspecified sim build is worthless.

### 4.2 Protocol

For each candidate (including the reference, re-run as a sanity check):

1. Fix a seed list of N episodes per task. N = 50 is a workable minimum; N = 200 is what you publish in a tech report.
2. Roll out the policy in sim, recording success/failure, time-to-success, and the full action+observation trace.
3. Compute:
   * **success-rate parity**: `success_rate(cand) − success_rate(ref)` per task. Acceptance: candidate must be within `Δ_SR` of reference (typically Δ_SR ≤ 2 percentage points absolute, with the per-task confidence interval reported).
   * **time-to-success parity**: median episode length difference. Catches policies that "succeed but slowly," which usually means a quantization that made the policy timid.
   * **failure-mode breakdown**: classify each failure (collision, dropped object, missed grasp, timeout). A candidate that has the same success rate as reference but different failure modes is suspicious and should not pass.

### 4.3 Chunk-cache and speculative-decode modes

If Lecture 1's rungs 5 or 6 are in play, the harness must run a *separate* sim sweep with chunk caching enabled. Per-tick parity will trivially pass (the cached actions are bit-exact); closed-loop success rate is where the staleness shows up. The deliverable is a table of `K` (cache stride) vs success-rate parity, per task. The largest `K` that stays inside the parity budget is the deployment setting.

### 4.4 Statistical confidence

With N = 50 episodes per task and a binomial outcome, the 95% CI on a 0.80 success rate is roughly ±11 pp. That means a 2 pp parity budget cannot be claimed at N = 50 — you need N ≥ ~400 per task for a tight 2 pp band, or you accept a wider band and report it.

The harness's pass/fail gate should not lie about confidence. If the budget is tighter than the CI, the gate reports "inconclusive at N = 50, re-run at N = 200" rather than "pass."

---

## 5. On-robot canary (Question 4)

The cheapest sim is **not the real robot**. The harness must define a canary protocol that **fails closed**.

### 5.1 Canary set

Pick 5-10 tasks from the deployment scope that:

* span the failure modes you care about (grasping, placing, transit, fine alignment)
* include at least one task that is *known* to be near the policy's capability edge — easy tasks pass everything and tell you nothing
* are recoverable if the policy fails (no crash, no dropped fragile object, no person in the workspace)

### 5.2 Protocol

For each candidate:

1. Run the **reference** policy on the canary set, M = 10-20 trials per task, behind the same safety wrapper you will use for the candidate. Record success/failure and any safety-stop events.
2. Run the **candidate** with the same setup, same operator, same scene reset protocol, on the same day if possible.
3. Compare success rates with a binomial test or Fisher's exact test per task. The pass criterion is: candidate is not statistically worse than reference at α = 0.05 on any individual task, *and* the aggregate success rate is within Δ_SR of reference.

### 5.3 The safety wrapper

The harness assumes the robot side has:

* a force / torque watchdog that triggers a controlled stop on unexpected contact
* a workspace-boundary monitor that prevents the policy from commanding the end-effector outside the safe envelope
* an action-rate-of-change limiter that rejects discontinuous commands
* a human-accessible e-stop within reach for the whole canary

If any of these are missing, the harness should refuse to grade an on-robot candidate. This is a code-level check, not a checklist item.

### 5.4 What "fails closed" means

A bug in the harness, a missing seed, an undefined budget, a sim version mismatch — any of these should cause the gate to *fail* the candidate, never to silently pass. The **default for unknown is "do not ship."**

---

## 6. Implementation skeleton

The harness is a small library plus a CLI. The shape, in pseudocode:

```text
harness/
├── manifest.yaml              # sim version, seed list, task suite, tolerance budgets
├── tape/                      # frozen observation tapes for per-tick + open-loop runs
│   ├── tape_001.npz
│   └── ...
├── adapters/
│   ├── openvla_ref.py         # loads reference checkpoint, exposes step(obs) -> action
│   ├── openvla_int4.py        # candidate
│   └── ...
├── metrics/
│   ├── per_tick.py            # MSE, max, logit-KL
│   ├── trajectory.py          # open-loop FK drift
│   ├── closed_loop.py         # sim rollout + success rate
│   └── canary.py              # on-robot grading with stats tests
├── gate.py                    # reads manifest, runs metrics, emits pass/fail
└── report.py                  # markdown + CSV + plots
```

CLI surface:

```text
parity tape --episodes 50 --out tape/                       # record reference observation tape
parity per-tick   --ref openvla_ref --cand openvla_int4     # answers Q1
parity trajectory --ref openvla_ref --cand openvla_int4     # answers Q2
parity sim        --suite libero-spatial --episodes 200     # answers Q3
parity canary     --robot panda-1 --tasks canary_set.yaml   # answers Q4
parity gate       --candidate openvla_int4 --manifest manifest.yaml  # CI entry point
parity report     --out reports/openvla_int4.md
```

Two design choices worth defending:

* **Adapters expose `step(obs) -> action`, nothing else.** No streaming, no internal state visible to the harness. This is what lets you grade a TRT-compiled policy and a vLLM-served policy with the same code.
* **The gate is the only thing CI runs.** Individual metrics are for debugging; the gate is the contract. If the gate passes, the candidate is shippable per the manifest's definitions. If the manifest is too lax, that is a manifest review, not a metric bug.

---

## 7. CI gating

The gate script is the artifact that makes the harness real. A useful contract:

```text
$ parity gate --candidate openvla_int4 --manifest manifest.yaml
[ok]   per_tick:    action_mse_within_budget=true   logit_kl=0.008
[ok]   trajectory:  drift_within_envelope=true
[ok]   sim:         success_rate_parity Δ=-0.7pp (CI ±2.1pp at N=200)
[skip] canary:      no robot configured (manifest.canary.required=false)
GATE: PASS
```

```text
$ parity gate --candidate openvla_int4_kv_fp8 --manifest manifest.yaml
[ok]   per_tick:    action_mse_within_budget=true   logit_kl=0.012
[FAIL] trajectory:  late-phase drift exceeds envelope on tasks: pick_butter, close_drawer
[skip] sim:         skipped because trajectory failed
GATE: FAIL — see reports/openvla_int4_kv_fp8.md
```

CI hooks:

* gate runs on every checkpoint pushed to the model registry
* gate is required for the `production` tag; a checkpoint without a passing gate cannot promote
* gate runs against the *current* manifest; if the manifest changes, all production candidates re-gate

---

## 8. Failure modes the harness exists to catch

These are the failures Lecture 1's optimizations actually produce in the wild, and how the harness catches them:

| Failure | Symptom | Caught by |
|---------|---------|-----------|
| Backbone INT4 over-suppresses rare action tokens | per-tick MSE looks fine but task success drops on tasks needing edge-case grips | sim parity (Question 3), not per-tick (Question 1) |
| Vision tower FP8 calibration trained on wrong distribution | per-tick fine on calibration-like scenes, fails on lighting changes | sim parity on a domain-randomized seed list |
| Chunk-cache K too aggressive | per-tick bit-exact, fails on fast sub-trajectories (final grasp closure) | sim parity with chunk-cache mode enabled |
| Action-head INT8 collapsed a continuous axis | per-axis MSE fine in mean, P99 max blows up on one axis | per-axis P99 budget in per-tick (Question 1) |
| Speculative decode rejects the wrong tokens | per-tick MSE fine, latency *worse* than expected | latency P95 vs. logit-KL diff in the gate |
| Sim parity passes but real-robot fails | success on LIBERO, drops on Panda | on-robot canary (Question 4) — sim is necessary not sufficient |
| Reference shifted under you | candidate parity drifts on the same code | manifest pins the reference; harness refuses to grade if the reference hash changed without a manifest bump |

If your harness does not catch all of these, list which ones it misses in `manifest.yaml` so the next engineer knows what is *not* validated.

---

## 9. Lab — Stand the harness up on Lecture 1's candidates

Continues directly from Lecture 1's lab. Assumes you have a reference + at least one candidate checkpoint.

1. Implement `adapters/` for the reference and one candidate. Each exposes `step(obs) -> action`.
2. Record an observation tape of 50 episodes from the reference on LIBERO-Spatial. Commit the tape (or a manifest pointing to a stable URL).
3. Implement `per_tick.py` with the metrics in §2.2. Run it on the candidate. Commit `reports/per_tick.csv`.
4. Implement `trajectory.py` with simple forward kinematics + drift accumulation. Plot `drift(t)` per episode. Commit `reports/trajectory.png`.
5. Implement `closed_loop.py` against LIBERO-Spatial with a fixed 200-seed list. Run reference and candidate. Compute success-rate parity with CIs. Commit `reports/sim_parity.md`.
6. Write `manifest.yaml` with your tolerance budgets and the seed list.
7. Write `gate.py` that ties it all together and exits 0 on pass, non-zero on fail.
8. Run the gate. Iterate the candidate until it passes, or document why it cannot.

Pass criterion for the lab: another engineer can clone the repo, run `parity gate --candidate openvla_int4 --manifest manifest.yaml`, and reproduce your pass/fail result on the same hardware class. The harness's value is reproducibility, not the metrics themselves.

---

## Self-check

1. Your candidate passes per-tick MSE within budget on every axis, but closed-loop sim success rate is 12 pp below the reference. Name two physical mechanisms that could cause this and the change you would make to the harness (not the candidate) to detect them earlier next time.
2. Your gate is reporting `inconclusive at N=50` for sim parity. The team wants to ship. What is the right answer, and what does it cost to actually move the gate to `pass`?
3. You have on-robot canary results for 10 trials per task and the candidate "looks similar" to the reference. Why is "looks similar" not allowed in the gate's vocabulary, and what is the smallest defensible statistical statement you can make at N = 10?
4. Why does the harness pin a sim version in `manifest.yaml`? What goes wrong if it does not?
5. A teammate proposes "let's skip on-robot canary because sim passed." Refute this in two sentences using one specific failure mode the canary catches that sim does not.

---

## References

* OpenVLA evaluation protocol — [code](https://github.com/openvla/openvla/tree/main/experiments)
* LIBERO benchmark — [paper](https://arxiv.org/abs/2306.03310), [code](https://github.com/Lifelong-Robot-Learning/LIBERO)
* RoboCasa — [project](https://robocasa.ai/), [paper](https://arxiv.org/abs/2406.02523)
* Isaac Lab manipulation suite — [docs](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)
* "Evaluating Real-World Robot Manipulation Policies in Simulation" — [paper](https://arxiv.org/abs/2405.05941) — sim/real correlation arguments that justify Question 4
* CalVin benchmark — [project](http://calvin.cs.uni-freiburg.de/) — alternative for long-horizon tasks
* On binomial confidence intervals for small-N robot evaluation — Wilson score interval, see [Brown, Cai & DasGupta (2001)](https://projecteuclid.org/euclid.ss/1009213286)

---

## Next in this special course

* Previous: [Lecture 1 — VLA Optimization for Real-Time Control](Lecture-01.md)
* Back: [VLA Optimization and Action-Parity Harness — Overview](README.md)
* Up: [Phase 5 — Robotics](../Guide.md)
