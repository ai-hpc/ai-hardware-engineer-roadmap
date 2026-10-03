# Lecture 14: Capstone: RL Post-Training π0.5 on LIBERO

## Overview

Picture a basketball player who never misses in their home gym. Every shot goes in, but they have practiced from exactly the same spots ten thousand times. Move the hoop half a meter, change the lighting, or put a chair where they usually stand, and their percentage collapses. They did not learn to shoot. They learned *those* shots. Coaching this player is not about more reps from the same spots. It is about practicing from new spots, scoring every attempt honestly, and not breaking the form they already have.

That is the state of π0.5 on LIBERO. openpi publishes a π0.5 checkpoint fine-tuned on LIBERO, and its LIBERO example reports success in the high 90s on the Spatial, Object, and Goal suites and the low 90s on LIBERO-10. **The standard suites leave little headroom.** Robustness studies tell a different story: LIBERO-PRO and LIBERO-Plus perturb objects, initial states, layouts, cameras, and instructions, and report that VLAs scoring above 90% on standard LIBERO degrade sharply under such changes. Their authors conclude that much of the standard score reflects memorized position-to-motion mappings. **The headroom is in robustness, and this capstone targets it.**

This kind of work is also scored in public settings — for example, Bittensor subnet SN80 scores improved π0.5 checkpoints in simulation with hidden tests that move objects. The protocol below is what you need to make a claim like that credible whether or not you enter one.

**Project statement.** Starting from openpi's `pi05_libero` checkpoint, produce an RL-post-trained checkpoint that **improves success on perturbed-layout LIBERO tasks by a statistically significant margin without regressing any nominal suite beyond its confidence interval**, with a reproducible config, a seed-controlled evaluation report, a before/after failure taxonomy, and a GPU-hour cost ledger.

The project follows an eight-step game plan, and every step uses an earlier lecture:

| Step | What you do | Lectures |
|---|---|---|
| 1. Understand the failure | Measure the nominal → perturbed drop and classify every failure | 02, 13 |
| 2. Better practice data | Privileged scripted teacher / DAgger, recovery examples, unusual layouts | 02, 03, 12 |
| 3. Start simple | Copy good tries more strongly; compare each layout to its own average | 04, 10, 11 |
| 4. Add a smart critic | A value function that sees exact object poses | 05 |
| 5. Pick a flow-policy RL method | Noise-injected PPO/GRPO, weighted regression, or critic-guided selection — then bake into the weights | 06, 10, 11 |
| 6. Protect general skills | KL leash, small steps, a little randomness, balanced suites | 06, 08, 12 |
| 7. Keep tasks short | Chunk length and replan horizon | 03, 13 |
| 8. Test like a scientist | Seeds, per-suite CIs, held-out perturbations, gated claims | 13 |

By the end you should be able to:

* reproduce the published π0.5 LIBERO baseline within its confidence interval and measure its perturbed-layout gap
* build a failure taxonomy that tells you which training signal to add
* run at least two RL post-training methods on a flow-matching VLA and compare them at an equal rollout budget
* keep the policy close to its starting point and prove with measurements that general skills survived
* plan and account for the GPU-hours of the whole project, split into rollout, learner, and evaluation
* defend the final number to a skeptical reviewer

---

## 1. The system you are post-training

From the π0.5 paper and the openpi configs: a PaliGemma-class vision-language model (`gemma_2b` backbone with a SigLIP image encoder) plus a ~300M-parameter flow-matching **action expert** (`gemma_300m`). Together that is a ~3B-class model. Each call takes two 224×224 camera images (agent view and wrist), the proprioceptive state, and the instruction, and integrates a flow-matching ODE from Gaussian noise to an action chunk. openpi's sampler defaults to **10 Euler steps**.

Three details of the LIBERO setup matter for everything below, and are easy to get wrong:

* **Chunk and replan horizon.** The base π0.5 config uses 50-step chunks, but `pi05_libero` uses `action_horizon=10`, and openpi's `examples/libero/main.py` executes the first `replan_steps=5` actions before querying again.
* **Fixed initial states.** The eval script runs `num_trials_per_task=50` episodes per task, each starting from one of LIBERO's 50 stored initial states per task (`task_suite.get_task_init_states`), with a fixed environment seed (default `7`). These 500 episodes per suite are the **test set**. Never train on them.
* **Horizons.** Maximum steps per episode are 220 (Spatial), 280 (Object), 300 (Goal), and 520 (LIBERO-10), after 10 no-op steps that let objects settle. At 5 actions per call, a LIBERO-10 episode costs up to ~104 VLA calls.

---

## 2. Hardware view: the compute plan

This is regime C. The simulator is a CPU process. **The policy forward pass dominates rollouts**, and a backward pass through a 3B model dominates the learner.

### 2.1 Memory budget (estimates — measure yours)

| Item | Full-parameter | LoRA |
|---|---|---|
| Policy weights (bf16, ~3.3B params) | ~6.6 GB trainable | ~6.6 GB frozen + adapters (MBs) |
| Optimizer state (AdamW, fp32 master + 2 moments) | ~40 GB | adapters only — small |
| Gradients | ~7-13 GB | adapters only |
| Reference policy for the KL leash | +~6.6 GB frozen copy | **free**: reference = adapters disabled |
| Critic | Small MLP on privileged state: negligible. A head on the VLM: shares the trunk, adds activations | same |
| Activations | Grow with batch × (prefix tokens + chunk × denoising steps) | same, minus frozen-weight grads |
| Rollout storage | Host RAM / disk: 2 uint8 images ≈ 0.3 MB per call, ~18 MB per 60-call episode | same |

openpi's README lists >22.5 GB for LoRA fine-tuning and >70 GB for full fine-tuning (supervised). RL adds the reference policy, rollout inference, and stored per-call data, so plan for headroom.

### 2.2 Two paths

| | **LoRA path** | **Full-parameter path** |
|---|---|---|
| Hardware | 1 × 80 GB (or 2: one rollout, one learner) | 4-8 × 80 GB |
| Layout | Colocated: rollout and learner share the GPU, time-sliced | Disaggregated: rollout GPUs serve batched inference; learner GPUs train with FSDP (openpi's `TrainConfig` has an `fsdp_devices` field) |
| Weight sync | Adapters only (MBs) | Full weights (~6.6 GB bf16) every policy update |
| Expressiveness | Limited by rank; less forgetting by construction | Highest; forgetting must be controlled explicitly |
| Notes | openpi ships a LoRA config for π0 (`pi0_libero_low_mem_finetune`) that shows the pattern (`gemma_2b_lora` / `gemma_300m_lora` variants plus a freeze filter); write the π0.5 equivalent. At the time of writing, openpi's PyTorch port lists LoRA as unsupported — use the JAX path or an RL framework | Use an RL framework that already schedules rollout / inference / training placement |

Per iteration, with \(N\) parallel environments, \(T\) steps, \(H\) actions per call, and \(t_{\text{chunk}}(N)\) per batched call:

$$
t_{\text{iter}} \approx \underbrace{\frac{T}{H}\, t_{\text{chunk}}(N) + T\, t_{\text{env}}}_{\text{rollout}} \;+\; \underbrace{E \cdot \frac{N\,T/H}{B}\, t_{\text{learn}}(B)}_{\text{learner}} \;+\; t_{\text{sync}}
$$

The usual lessons apply: batch across environments until rollout throughput flattens, keep CPU simulator workers ahead of the GPU, and measure the rollout/learner split before optimizing either.

### 2.3 The cost ledger

Keep `COST_LEDGER.csv` from day one, one row per run:

| run_id | phase | GPUs | rollout GPU-h | learner GPU-h | eval GPU-h | episodes collected | learner steps | outcome |
|---|---|---|---|---|---|---|---|---|
| p0-baseline | 0 | 1×H100 | — | — | measured | 2,000 | — | reproduced? |
| p3-awr-s0 | 3 | 1×H100 | measured | measured | measured | … | … | gate passed? |

Derive GPU-hours from timestamps logged by each process, not from estimates. The final report quotes the totals and the cost of the *winning recipe alone*, so another team can price a rerun.

---

## 3. Phase 0 — Baseline protocol

Goal: reproduce the published `pi05_libero` numbers and wire the evaluation into `eval_harness/` from Lecture 13.

1. Clone openpi **with submodules** (`git submodule update --init --recursive`) and follow `examples/libero/README.md`. The Dockerized path is recommended. By default it serves the `pi05_libero` checkpoint (`gs://openpi-assets/checkpoints/pi05_libero`) and runs the client:

   ```bash
   SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
   ```

   Select a suite with `CLIENT_ARGS="--args.task-suite-name libero_10"`, and serve your own checkpoints with `SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir <dir>"`. The non-Docker path runs `scripts/serve_policy.py --env LIBERO` in one terminal and `examples/libero/main.py` (Python 3.8 venv) in another.
2. **Patch the client to log one row per episode.** The stock script logs running totals and treats a simulator exception as a failed episode. Separate the two:

   ```python
   # in examples/libero/main.py, eval_libero(): sketch of the additions
   rows = []                                   # before the task loop
   calls, crashed, t_start = 0, False, time.perf_counter()   # at the start of each episode
   # ... where the script calls client.infer(element): increment `calls`
   # ... in the except branch: set crashed = True
   rows.append(dict(policy=POLICY_TAG, suite=args.task_suite_name, task_id=task_id,
                    init_id=episode_idx, eval_seed=args.seed, success=bool(done) and not crashed,
                    crashed=crashed, steps=t, policy_calls=calls,
                    wall_s=time.perf_counter() - t_start))           # after each episode
   pd.DataFrame(rows).to_csv(out_csv, index=False)                    # after the task loop
   ```

   Also include the episode index in video filenames. The stock names overwrite each other per task and outcome.
3. Run all four suites. **Gate:** each suite's Wilson 95% interval contains the published number. If not, stop and debug before training anything — image rotation, resize, state layout, normalization statistics, and package versions are the usual suspects.
4. Measure throughput: per-call latency at batch 1 and at your planned batch size, calls per episode, and episodes per GPU-hour. These go into the cost model.

### 3.1 The perturbed baseline

Define three layout sources and **keep their roles separate**:

| Source | Role | Notes |
|---|---|---|
| Your own perturbation generator | **Training** distribution for RL | Sample new object placements. Calling `env.reset()` without `set_init_state` gives a fresh placement from the task's BDDL regions — verify by rendering a few. Or perturb object poses in the stored states, letting them settle during the no-op steps |
| Held-out draws from your generator (test-range seeds) | **Validation** for checkpoint selection | Never used for gradient updates |
| LIBERO-PRO and/or LIBERO-Plus variants | **Test**: a perturbation generator you did not write | Guards against overfitting to your own generator |

Run the base checkpoint on all of them. **Gate:** the nominal-to-perturbed gap is large enough to measure. With the episode counts from your `episodes_needed.md`, the CIs of the nominal and perturbed rates must not overlap. If the gap is within noise, choose harder perturbations — no improvement you make will be detectable otherwise.

---

## 4. Phase 1 — Failure taxonomy (step 1)

Watch videos and use privileged simulator state to label every failed episode of the base policy on the validation layouts. Start from this template and extend it:

| Category | Definition | Auto-detect from sim state | Base nominal | Base perturbed | Final nominal | Final perturbed |
|---|---|---|---|---|---|---|
| Memorized location | Reaches to where the object *usually* is | Gripper's closest approach is near the nominal pose, far from the actual one | | | | |
| Wrong object | Manipulates a distractor | First contact with a non-target object | | | | |
| Missed grasp | Reaches the target, closes on air | Gripper closed near target, object never lifted | | | | |
| Drop / slip | Lifted, then lost in transport | Object height rises then falls away from goal | | | | |
| Near-miss placement | Released close to but outside the goal region | Final object-goal distance under a threshold | | | | |
| Collision / knock-over | Disturbs the scene irrecoverably | Large non-target object displacement | | | | |
| Stall / oscillation | No progress for many calls | Low end-effector displacement over a window | | | | |
| Instruction ignored | Performs a different task from the same scene | Predicate of another task satisfied | | | | |
| Timeout near success | Correct behavior, too slow | Success predicate within a few steps of being true | | | | |
| Sim exception | Crash, not a policy failure | Logged `crashed=True` | | | | |

The taxonomy decides the training signal. "Memorized location" and "wrong object" call for layout diversity and RL on perturbed layouts. "Drop / slip" and "missed grasp" call for recovery data. "Instruction ignored" calls for multi-task balance and language checks. Report counts with CIs before and after.

---

## 5. Phase 2 — Better practice data (step 2)

The cheapest improvement is often supervised. Before any RL:

* **Privileged scripted teacher.** In simulation a teacher can read every object pose. Writing waypoint controllers (approach, grasp, lift, transport, place) for 40 tasks is real engineering. Start with the task families that fail most in your taxonomy. Use the teacher to (a) generate demonstrations on perturbed layouts and (b) label states the *student* visits — DAgger without a human (Lecture 02).
* **Recovery data.** Inject perturbations mid-episode (nudge the object, open the gripper, add DART-style action noise) and record the teacher's recovery.
* **Self-generated successes.** Run π0.5 itself on perturbed layouts and keep successful episodes. This needs no teacher, and it is the first step of Phase 3.

Fine-tune with openpi's standard flow-matching recipe on the base LIBERO data mixed with the new data, converted to LeRobot format the same way `examples/libero/convert_libero_data_to_lerobot.py` converts the original. Keep normalization statistics consistent with the checkpoint unless you deliberately recompute them and retrain. **This SFT-augmented checkpoint is the baseline every RL method below must beat at equal GPU-hours.**

---

## 6. Phase 3 — Start simple: group baselines and weighted copying (step 3)

For each training layout (a sampled seed), run a **group** of \(G\) rollouts (e.g., \(G = 8\)) from the identical initial state. Flow sampling starts from fresh Gaussian noise on every call, so rollouts differ. Verify that two calls with identical observations return different chunks; if they do not, control the sampler's noise explicitly. With binary success \(R_i \in \{0, 1\}\):

$$
A_i = R_i - \frac{1}{G}\sum_{j=1}^{G} R_j
$$

which is the per-initial-state Monte-Carlo baseline from Lecture 04. Groups where every rollout succeeds or every rollout fails give \(A_i = 0\) for all members: **zero signal, wasted rollouts.** Log their fraction, and use Lecture 12's success-rate-adaptive sampling to spend rollouts on "goldilocks" layouts.

```python
# capstone/groups.py — advantages and the dataset for weighted flow-matching regression
import numpy as np

def group_advantages(success: np.ndarray) -> tuple[np.ndarray, float]:
    """success: (num_layouts, G) in {0,1}. Returns per-rollout advantages and the zero-signal fraction."""
    adv = success - success.mean(axis=1, keepdims=True)
    zero_signal = float(np.mean(success.std(axis=1) == 0))
    return adv, zero_signal

def regression_weights(adv: np.ndarray, beta: float = 0.5, mode: str = "awr") -> np.ndarray:
    if mode == "filtered":                       # filtered BC: keep successes from informative groups
        return (adv > 0).astype(np.float32)
    # Zero-signal groups (all success / all fail) have adv = 0 -> weight 1: plain self-distillation.
    # Pass a mask (or drop those episodes) if you want only informative groups to move the policy.
    w = np.exp(adv / beta)                       # AWR-style exp(A / beta), Lecture 10
    return np.minimum(w, 20.0).astype(np.float32)
```

Train with the ordinary flow-matching loss, weighting each logged chunk by its episode's weight, mixed with a fixed fraction of the original LIBERO demonstrations. No log-probs are needed. This is the simplest correct form of "copy good tries more strongly" for a flow policy, and the same family as advantage-conditioned approaches such as Physical Intelligence's RECAP (π*0.6), which conditions the policy on an advantage indicator computed by a learned value function.

**Gate:** perturbed-validation success improves over the SFT-augmented baseline with non-overlapping CIs, and no nominal suite regresses beyond its CI. Iterate collect → weight → fine-tune for a few rounds before moving on.

---

## 7. Phase 4 — A critic that sees everything (step 4)

The group mean is a baseline for \(s_0\) only. A **privileged critic** \(V_\phi(s_t)\) trained on exact object poses, gripper state, and task ID (Lecture 05's asymmetric actor-critic) gives per-call advantages along the episode. That shows *which* chunk caused a drop and lets one rollout score many decisions. With binary success and \(\gamma \approx 1\), \(V_\phi\) estimates the probability of success from here, so it doubles as a progress monitor for the taxonomy.

* Train \(V_\phi\) on Monte-Carlo returns from Phase 3 rollouts first, then TD targets. Track explained variance on held-out episodes.
* The critic sees state, never pixels, so it costs almost nothing. The actor still sees only cameras, instruction, and proprio. **Nothing privileged may reach the policy at evaluation.**
* Use GAE over policy calls (one decision = one chunk) with \(\lambda \approx 0.95\).

---

## 8. Phase 5 — Pick a flow-policy RL method, then bake it in (step 5)

π0.5 has no cheap exact action log-likelihood (Lecture 11). Choose one primary method and compare it with Phase 3 at **equal rollout GPU-hours**:

| Method | How it gets a training signal | Learner cost | Main risk |
|---|---|---|---|
| **Noise-injected PPO / GRPO** (πRL's Flow-Noise / Flow-SDE, ReinFlow) | Make each denoising step stochastic so it is Gaussian with a tractable log-prob, then use clipped-ratio updates with group or critic advantages | Store the denoising trajectory per call; backprop through \(K\) expert steps | Training sampler (noisy) differs from deployment sampler (ODE) — evaluate with the deployment sampler |
| **Advantage-weighted / conditioned regression** (Phase 3, AWR, RECAP-style) | Flow-matching loss on self-generated data, reweighted or conditioned by advantage | Same as SFT | Improves slowly; bounded by behaviors the policy already samples |
| **Noise-space steering** (DSRL) | Train a small policy that chooses the input noise of the frozen flow model, using a critic | Tiny actor + critic; base frozen | The result lives *outside* the weights until distilled |
| **Critic-guided sample-and-rank** | Sample \(N\) chunks, execute the one the critic scores highest, then distill the selected behavior | \(N\)× rollout inference | Exploiting critic errors (Lecture 08 reward hacking) |

**Bake it into the weights.** Steering and sample-and-rank improve behavior at inference time with extra machinery. If the deliverable is a checkpoint — as it is here, and in any competition that accepts only the model — collect the improved behavior on training layouts and distill it into π0.5 with the flow-matching loss. Then evaluate the distilled checkpoint alone.

For the full-parameter path, frameworks such as **RLinf** (π0 / π0.5 support, LIBERO plus its PRO and Plus variants, PPO / GRPO and other algorithms; its README links the πRL report as its π0 / π0.5 RL fine-tuning report) handle rollout/inference/training placement. Read their current configs rather than relying on remembered flags.

---

## 9. Phase 6 — Protect general skills (step 6)

RL on perturbed layouts can destroy what made π0.5 work: language following, nominal tasks, smooth motion.

* **KL leash to the base policy.** Flow policies have no closed-form KL between action distributions. With noise injection, each denoising step is Gaussian, so the per-step KL to the reference is closed-form. When both share the step variance \(\sigma^2\) it reduces to \(\mathrm{KL} = \lVert \mu_\theta - \mu_{\text{ref}} \rVert^2 / (2\sigma^2)\); with learned noise (as in ReinFlow) use the general diagonal-Gaussian formula. Otherwise, penalize the squared difference between current and reference velocity predictions on the same noisy inputs, as a proxy. On the LoRA path the reference is the same network with adapters disabled.
* **Small steps.** A low learning rate, clipped ratios, few epochs per batch, and gradient-norm clipping.
* **A little randomness.** Keep sampling stochastic during collection. Collapse to one deterministic behavior kills group variance (more zero-signal groups) and robustness.
* **Balanced practice.** Sample training layouts across all four suites with a floor per suite, and mix original demonstrations into every update. Easy suites must not crowd out hard ones, or the reverse.
* **Regression tests every N iterations.** A small fixed nominal validation set per suite, plus an instruction-swap probe (same scene, a different valid instruction), since robustness studies report that VLAs often ignore language.

---

## 10. Phase 7 — Keep tasks short (step 7)

Chunking sets both the effective horizon (Lecture 13) and the rollout cost. Sweep `replan_steps` ∈ {2, 5, 10} on validation layouts and measure success, calls per episode, and failures by category. Shorter replan horizons react better to slips but cost proportionally more inference (5 → 2 is 2.5× the calls) and lengthen the credit-assignment chain. You may also cut denoising steps during rollouts to save compute. If you do, check that the training-time sampler still matches the deployment sampler on a parity set of observations (Lecture 03; the [action-parity harness](../VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-02.md)), and **evaluate the final checkpoint with exactly the settings you ship.**

---

## 11. Phase 8 — Test like a scientist (step 8)

The final evaluation is frozen *before* the last training run:

* Test sets: the four standard nominal suites (500 episodes each, fixed initial states, eval seed recorded), your held-out perturbed layouts, and at least one external robustness suite (LIBERO-PRO or LIBERO-Plus).
* ≥ 3 training seeds of the final recipe. Report per-seed results, mean, IQM, and stratified-bootstrap CIs from `eval_harness/`.
* Per-suite and per-perturbation-type success with Wilson CIs. A paired McNemar test against the base checkpoint on identical episodes.
* Checkpoints selected on validation layouts only; test sets touched once.
* **Claim rule:** the perturbed-test improvement CI excludes zero, and no nominal suite regresses beyond its CI. Anything weaker is reported as "no significant difference."

---

## 12. Risk register

| Risk | Symptom | Detection | Mitigation |
|---|---|---|---|
| Reward hacking via sim bugs | Success rises; videos show objects tunneling, launched, or the predicate triggered by knocking things over | Review random *successes* every round, not only failures; physics sanity checks on object velocities | Fix the predicate or the scene; add termination on implausible states |
| Forgetting | Nominal suites or instruction probes decline | Regression tests every N iterations | KL leash, demo mixing, lower LR, LoRA |
| Eval leakage | Large gains on standard states only | Audit which seeds and init states each run sampled | Disjoint seed ranges; standard init states eval-only |
| Overfitting to your perturbation generator | Gains on your generator, none on LIBERO-PRO / LIBERO-Plus | Always report the external suite | Diversify the generator; select checkpoints on validation, not test |
| Selection bias | Best-of-many checkpoint looks great, does not replicate | Re-evaluate on fresh test seeds | Fix the selection rule in advance |
| Sampler mismatch | Training metrics improve, deployment-sampler eval does not | Eval with the deployment sampler each round | Distill; match NFE and noise settings |
| Zero-signal collapse | Most groups all-0 or all-1 | `zero_signal` fraction in logs | Adaptive layout sampling, curriculum, larger \(G\) for hard layouts |
| Budget overrun | GPU-hours per gate far above plan | Ledger vs plan after each phase | Go/no-go gates; cut scope before cutting evaluation |

---

## 13. Milestones and go/no-go gates

| Milestone | Deliverable | Go criterion | If no-go |
|---|---|---|---|
| M0 Baseline | Per-episode CSVs, throughput numbers | Every nominal suite's CI contains the published number | Debug preprocessing / versions; do not train |
| M1 Headroom | Perturbed baseline + taxonomy | Nominal and perturbed CIs separated | Harder or different perturbations |
| M2 Data | SFT-augmented checkpoint | Perturbed validation improves; no nominal regression | Revisit teacher coverage / data mix |
| M3 Simple RL | Group-baseline weighted regression | Beats M2 at equal GPU-hours | Check zero-signal fraction, \(\beta\), mixing ratio |
| M4 Critic | Privileged critic, explained variance | Explained variance clearly above zero on held-out data; per-call advantages beat M3 | Use M3 with group baselines only |
| M5 Flow RL | One Phase 5 method, distilled into weights | Beats M3 at equal GPU-hours, or documented negative result | Ship M3 |
| M6 Final eval | Frozen protocol, ≥ 3 seeds | Claim rule satisfied | Report "no significant difference" honestly |
| M7 Ship | Repo, report, ledger | Another engineer reproduces within CI | Fix reproducibility gaps |

A documented negative result at M5 is a valid outcome. Shipping M3 with honest numbers beats shipping M5 without them.

---

## 14. Measure it

| Metric | Definition | Why it matters |
|---|---|---|
| **Per-suite success, nominal** | Wilson CI, 500 episodes per suite | Regression guard |
| **Per-suite / per-perturbation success** | Held-out and external suites, Wilson CI | The headline |
| **Paired improvement vs base** | McNemar on identical episodes | Whether the claim holds |
| **Seed spread** | IQM + stratified bootstrap over ≥ 3 seeds | Recipe, not luck |
| **Zero-signal group fraction** | All-0 or all-1 groups / total groups | Wasted rollout compute |
| **KL / action drift to base** | Per-step Gaussian KL or velocity-MSE proxy | Leash health |
| **Rollout throughput** | Episodes/GPU-h, calls/episode, latency at batch | Regime-C cost driver |
| **GPU-hour split** | Rollout / learner / eval from the ledger | Cost to reproduce |
| **Failure taxonomy shift** | Category counts before → after | What actually changed |

---

## 15. Ship it

Matching the course's "What you ship":

* **`capstone/`** — data-generation scripts (teacher, perturbation generator, group collector), training configs for every phase, and a `REPRODUCE.md` with exact commands, commit hashes, and seeds.
* **The checkpoint** — the final post-trained π0.5, loadable through openpi's `serve_policy.py` with your config, plus its normalization assets.
* **`EVAL_REPORT.md`** — per-suite nominal and perturbed results with CIs for base, SFT-augmented, and final checkpoints over ≥ 3 seeds; the paired test; the external robustness suite; and the seed-list hash.
* **`FAILURE_TAXONOMY.md`** — the Phase 1 table filled in before and after, with representative videos.
* **`COST_LEDGER.csv` + `COST_SUMMARY.md`** — GPU-hours split into rollout, learner, and evaluation, per phase and for the winning recipe alone.
* Links to `rollout_bench/`, `eval_harness/`, and the per-lecture implementations this capstone used.

---

## Exit criteria

You are done when:

* the baseline reproduction passes its gate, and the perturbed gap is measured with CIs
* at least two post-training methods have been compared at equal rollout GPU-hours, with the comparison in the report
* the final checkpoint satisfies the claim rule — or the report says clearly that it does not, and why
* nominal regression tests and instruction probes show general skills survived
* another engineer, given the repo and the ledger, can reproduce your headline number within its CI and predict the GPU-hours it will take

---

## Self-check

1. After two rounds of RL, perturbed validation success is up 12 points, but LIBERO-Goal dropped 4 points and its CI no longer overlaps the base. Which Phase 6 mechanisms do you check first, and what does the failure taxonomy tell you that the aggregate does not?
2. Your logs show 70% of groups are all-success or all-failure. What does this cost in GPU-hours per useful gradient, and name three changes (from Lectures 04, 12, and this lecture) that would reduce it.
3. Success on training layouts jumps from 55% to 90% in one round. Videos of randomly chosen *successes* show the bowl occasionally clipping through the plate and registering as placed. What is happening, why did the failure review miss it, and what do you change in the protocol?
4. You have one 80 GB GPU. Justify LoRA over full-parameter training using the memory table, and explain why the KL leash costs almost nothing on that path. What do you give up?
5. Your checkpoint gains 15 points on your own perturbation generator and 1 point (CI includes zero) on LIBERO-Plus layout perturbations. What do you claim, and what would you change in training before trying again?
6. DSRL steering gives your best validation numbers, but the deliverable must be a single checkpoint loaded by `serve_policy.py`. Describe how you convert it, which metric you re-measure after conversion, and which risk-register entry applies.

---

## References

* Black et al., "π0.5: a Vision-Language-Action Model with Open-World Generalization," 2025 — [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
* Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," 2024 — [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
* openpi — [repo](https://github.com/Physical-Intelligence/openpi), [LIBERO example](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero)
* Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning," 2023 — [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)
* "LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization," 2025 — [arXiv:2510.03827](https://arxiv.org/abs/2510.03827), [code](https://github.com/Zxy-MLlab/LIBERO-PRO)
* Fei et al., "LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models," 2025 — [arXiv:2510.13626](https://arxiv.org/abs/2510.13626), [code](https://github.com/sylvestf/LIBERO-plus)
* Chen et al., "πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models," 2025 — [arXiv:2510.25889](https://arxiv.org/abs/2510.25889)
* "ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning," 2025 — [arXiv:2505.22094](https://arxiv.org/abs/2505.22094)
* "Steering Your Diffusion Policy with Latent Space Reinforcement Learning" (DSRL), 2025 — [arXiv:2506.15799](https://arxiv.org/abs/2506.15799)
* Physical Intelligence, "π*0.6: a VLA That Learns From Experience" (RECAP), 2025 — [arXiv:2511.14759](https://arxiv.org/abs/2511.14759), [blog](https://pi.website/blog/pistar06)
* "RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models," 2025 — [arXiv:2510.06710](https://arxiv.org/abs/2510.06710), [RLinf repo](https://github.com/RLinf/RLinf)
* Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021 — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

---

## Next in this special course

* What comes after: [VLA Optimization and Action-Parity Harness](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) — make the checkpoint you just trained meet a real control-loop deadline without losing the success rate you earned
* Previous: [Lecture 13 — Theory, Evaluation Rigor, and Sim-to-Real](Lecture-13.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
