# Deep RL for Robot Learning — Special Course

<div class="course-identity robotics" markdown="1">
<div class="course-identity__icon">RL</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · Robotics · Special Course</p>
<p class="course-identity__title">Train robot policies by copying, trying, and improving — from behavioral cloning to RL post-training of a flow-matching VLA.</p>
<p class="course-identity__meta">Artifact: RL-post-trained policy + training/eval harness · Measure: success rate per suite, env-steps/s, samples-to-success, wall-clock-to-success</p>
</div>
</div>

> *Copying gets a robot onto the scoreboard. Practicing with a score gets it to the top.*

This course is the **training half** of Track D's robot-learning path: imitation learning, policy gradients, actor-critic, PPO, value-based methods, and model-based/offline RL. The companion [VLA action-parity course](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) includes an actual trained CPU policy bridge plus separately labeled pretrained-model deployment work. This course's included CPU lab checks return targets and mathematical snippets; simulator trainers and pretrained VLA RL remain implementation exercises unless accompanied by execution evidence.

The lecture sequence follows the topic order of **UC Berkeley CS 185/285, Deep Reinforcement Learning (Spring 2026, Sergey Levine)**, compressed from 25 lectures into 13 plus a capstone, with one production deep dive (06b, RSL-RL). Every lecture is rewritten for this roadmap's standard: build it, measure it, ship an artifact — and always ask where the compute goes.

**Comprehensive VLA companion:** [VLA Foundations to Frontier 2026](../VLA%20Foundations%20to%20Frontier%202026/README.md) provides 32 lectures spanning history, architectures, data, imitation/flow/RL training, evaluation, deployment, and the research frontier reviewed through October 18, 2026. Use its [CPU labs](../VLA%20Foundations%20to%20Frontier%202026/Lab-Guide.md) and capstones alongside this specialization.

**Scope:** algorithms, training systems, and evaluation for robot policies in simulation, with sim-to-real as the frontier. The runtime optimization of a *trained* policy for deployment belongs to the VLA course; general LLM post-training infra belongs to [Track G](../../Track%20G%20-%20ML%20Systems%20Engineering/Guide.md).

**Layer mapping:** L2-L8. Model architecture (flow-matching action experts, critics, world models), training systems (rollout/learner split, GPU-parallel simulation, replay buffers), and the closed-loop evaluation harness.

**Role targets:** Robot Learning Engineer · Robotics Foundation-Model Engineer (post-training) · RL Infrastructure Engineer · Simulation / Rollout Infra Engineer · Applied Research Engineer (embodied AI).

**Prerequisites:**

* Phase 3 — [Neural Networks](../../../Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Guide.md) and [PyTorch](../../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/Guide.md) — you can write a training loop and debug a loss curve.
* Phase 3 — [Transformer Fundamentals](../../../Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Transformer%20Fundamentals/Lecture-01.md) — VLA backbones are transformers.
* Phase 5 — Robotics — [Advanced Perception and AI for Robotics, Part C](../Advanced%20Perception%20and%20AI%20for%20Robotics/Lecture-01.md) — the 10-minute survey this course expands.
* Phase 5 — Track G — [Logprobs, Perplexity and KL Divergence](../../Track%20G%20-%20ML%20Systems%20Engineering/Logprobs,%20Perplexity%20and%20KL%20Divergence/README.md) — helpful before Lectures 06, 08, and 11.
* Comfortable with probability (expectations, log-likelihood, KL) at the level of an intro ML course.

**Hardware assumed:** one 24-48 GB NVIDIA GPU (RTX 4090 / L40S / A6000 class) for Lectures 01-13. The capstone needs either one 80 GB GPU (LoRA path) or 4-8 × 80 GB (full-parameter path).

**CPU entry exercise:** use [the return-target and PPO lab](CPU-Lab.md) before GPU training. It includes a complete hand calculation, standard-library implementation, 12 tests, and deliberate termination/truncation/reset mistakes. Completing this arithmetic milestone does not require the hardware budget above or claim a trained policy.

**What comes after:** [VLA Optimization and Action-Parity Harness](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) — take the policy you trained here and make it meet a real control-loop deadline without losing the success rate you just earned.

---

## Lecture map

<div class="lecture-map" markdown>

| # | Title | CS 185/285 topics | Focus |
|---|-------|-------------------|-------|
| 01 | [The RL Problem and the Rollout Engine](Lecture-01.md) | 1, 4 | MDP vocabulary · return, V, Q, advantage · try → judge → improve · where wall-clock goes: env step vs policy forward vs learner · GPU-parallel simulation |
| 02 | [Imitation Learning: Behavioral Cloning and DAgger](Lecture-02.md) | 2 | Demonstrations · BC as supervised learning · distribution shift and the T² compounding bound · DAgger with a privileged scripted expert · recovery data |
| 03 | [Modern Imitation: Multimodality, Action Chunks, Flow Matching](Lecture-03.md) | 3 | History and causal confusion · multimodal actions · tokenized vs diffusion vs flow heads · action chunking · pre-train → post-train · the π0.5 architecture |
| 04 | [Policy Gradients: Do More of What Worked](Lecture-04.md) | 5 | REINFORCE derivation · why dynamics cancel · baselines and reward-to-go · variance vs batch size · group (per-initial-state) baselines |
| 05 | [Actor-Critic, GAE, and Privileged Critics](Lecture-05.md) | 6 | Value functions as a learned baseline · TD error · GAE and the λ dial · discount γ · asymmetric critics that see simulator state |
| 06 | [PPO, Trust Regions, and the KL Leash](Lecture-06.md) | 9, 10 | Importance sampling · clipped surrogate · TRPO and the performance-difference bound · natural gradient · KL penalties to a reference policy · PPO at GPU-sim scale |
| 06b | [Actor-Critic PPO in Practice: Inside RSL-RL](Lecture-06b.md) | 6, 9 | The PPO behind Isaac Lab, read line by line · runner, models, storage · observation groups and asymmetric critics · time-out bootstrapping · KL-adaptive learning rate · clipped value loss · ablations on a custom vectorized env |
| 07 | [Value-Based and Off-Policy RL: From Q-Learning to SAC](Lecture-07.md) | 7, 8, 13 | Value iteration · fitted Q · target networks · replay and update-to-data ratio · double Q · DDPG / TD3 · max-entropy RL and SAC · sample-and-rank |
| 08 | [The Probabilistic Toolkit: ELBO, VAEs, Control as Inference, Inverse RL](Lecture-08.md) | 11, 12, 13 | Latent variables · ELBO and the KL gap · reparameterization · diffusion/flow as stacked VAEs · soft optimality · MaxEnt IRL · GAIL and learned success detectors |
| 09 | [Model-Based RL and World Models](Lecture-09.md) | 15, 16 | Learned dynamics · model exploitation · aleatoric vs epistemic uncertainty · ensembles · random shooting, CEM, MPC · Dyna/MBPO short branches · Dreamer / TD-MPC |
| 10 | [Offline RL and Offline-to-Online Fine-Tuning](Lecture-10.md) | 17, 18 | Fixed datasets · stitching · OOD action overestimation · AWR, CQL, IQL · RLPD · RL for diffusion/flow policies: IDQL, FQL, noise-space steering (DSRL) |
| 11 | [RL for LLMs and VLAs](Lecture-11.md) | 14 | Pre-train → SFT → RL · reward models vs verifiers · GRPO · KL to reference · why flow VLAs lack cheap log-probs and the workarounds · rollout systems where policy inference dominates |
| 12 | [Exploration, Curricula, Skills, and Multi-Task RL](Lecture-12.md) | 19, 23, 24 | Sparse rewards · optimism and novelty bonuses (RND) · curricula over initial states · DIAYN / Skew-Fit · hindsight relabeling · multi-task balance · hierarchy · meta-RL |
| 13 | [Theory, Evaluation Rigor, and Sim-to-Real](Lecture-13.md) | 20, 21-22, 25 | Contraction and error vs horizon · why chunks shorten the horizon · seeds and confidence intervals · per-suite reporting · domain randomization · the real-world frontier |
| 14 | [Capstone: RL Post-Training π0.5 on LIBERO](Lecture-14.md) | — | Baseline → diagnose failures → better data → RL post-training → protect general skills → gated, seed-controlled evaluation on nominal and perturbed-layout suites |

</div>

Each lecture develops theory, a worked mechanism, an implementation exercise, and evaluation criteria. Start with [CPU-Lab.md](CPU-Lab.md) for included runnable files. Names such as `ppo/`, `vla_rl/`, and `eval_harness/` in the lecture sketches are learner-created packages, not missing downloads. GPU/simulator recipes require their stated adapter, data, model, and hardware work and are not claimed as executed here.

---

## The environment ladder

The whole course climbs one ladder of environments, so measurements from early lectures stay comparable to later ones:

| Rung | Environment | Policy | Where time goes | Used in |
|------|-------------|--------|-----------------|---------|
| 1 | Gymnasium `CartPole-v1`, `Pendulum-v1` (CPU) | 2-layer MLP | Python env step overhead | 01, 04, 05, 07 |
| 2 | GPU-parallel manipulation — [ManiSkill3](https://github.com/haosulab/ManiSkill) `PickCube-v1` / `PushCube-v1` (or [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — see the [Isaac Sim and Isaac Lab](../Isaac%20Sim%20and%20Isaac%20Lab/README.md) course) | MLP on state, small CNN on pixels | Learner backward pass, GPU memory | 01-03, 05-07, 09, 10, 12 |
| 3 | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) suites (Spatial / Object / Goal / Long), plus perturbed layouts | π0.5 via [openpi](https://github.com/Physical-Intelligence/openpi) | **Policy forward pass** — billions of parameters per action chunk | 03, 10, 11, 13, 14 |

The move from small GPU-simulated policies to a large VLA can shift the bottleneck from learning or simulation to policy inference. The result depends on scene complexity, observation rendering, batch size, model, and hardware. Measure environment stepping, inference, learner updates, and synchronization separately; no fixed transitions-per-second range applies across these configurations.

---

## Why this is a hardware-first course

RL is usually taught as math. In practice, the binding constraint on a robot-learning team is almost always **throughput**:

* **Rollouts are inference at scale.** RL post-training of a VLA spends most of its GPU-hours generating actions, not computing gradients. Every trick from the inference world — batching across parallel envs, CUDA Graphs, reduced flow-matching steps, KV reuse — directly speeds up training.
* **Sample efficiency and wall-clock efficiency are different metrics.** An off-policy method that needs 10× fewer env steps but 20× more gradient updates per step loses on wall-clock whenever simulation is cheap. Whenever the policy forward pass is expensive, it wins. Which one holds depends on the hardware, not on the algorithm.
* **Memory constrains algorithm choices.** A reference may share an immutable base with adapters; a critic may be a separate network or a smaller shared-feature head. PPO stores old log-probabilities, while GRPO avoids a learned value network. Count the actual weights, optimizer state, activations, and rollout storage before choosing a configuration.
* **Evaluation is a compute budget.** A credible success-rate claim at ±3% needs on the order of a thousand episodes per suite. That is real GPU time, and it has to be planned.

Every lecture ends with a **Measure it** section that asks for env-steps/s, policy-inferences/s, learner-updates/s, GPU memory, and samples-to-success alongside the usual learning curves.

---

## What you ship

By the end of the course you should have, in one repo:

* **`rollout_bench/`** — the throughput characterization from Lecture 01 for all three ladder rungs (CSV + plot + a one-page "where does time go" note).
* **From-scratch implementations** (CleanRL-style, single file each) of BC + DAgger, a flow-matching chunked policy, REINFORCE with group baselines, PPO with GAE and an asymmetric critic, SAC, and one offline method (IQL or RLPD) — each with learning curves over ≥ 3 seeds.
* **A capstone checkpoint** — π0.5 post-trained with RL on LIBERO, with a reproducible training config.
* **An evaluation report** — success rate per suite on nominal *and* perturbed layouts, ≥ 3 seeds, confidence intervals, compared with the base checkpoint, plus a short failure taxonomy before and after.
* **A cost ledger** — GPU-hours split into rollout, learner, and evaluation, so another team can estimate what a rerun will cost.

---

## Exit criteria

You are done with this special course when you can:

* derive the policy gradient on a whiteboard and explain why the environment's dynamics do not appear in it
* explain, for a specific robot task and GPU, whether PPO, SAC, GRPO, or an offline method is the right first choice — and justify it with **throughput numbers**, not preference
* describe at least two ways to run RL on a flow-matching policy that has no cheap log-likelihood, and the trade-off each one makes
* defend a success-rate improvement to a skeptical reviewer: seeds, confidence intervals, per-suite breakdown, perturbed-layout results
* point to the repo above and have another engineer reproduce your capstone number within its confidence interval

---

## References for the whole course

* UC Berkeley CS 285 / CS 185 Deep RL — [course site](https://rail.eecs.berkeley.edu/deeprlcourse/) (lecture slides and videos from earlier offerings)
* Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. — [free online](http://incompleteideas.net/book/the-book-2nd.html)
* OpenAI Spinning Up — [docs](https://spinningup.openai.com/) (clear derivations for PG, PPO, SAC)
* CleanRL — [repo](https://github.com/vwxyzjn/cleanrl) (single-file reference implementations to diff against)
* π0 — [paper](https://arxiv.org/abs/2410.24164) · π0.5 — [paper](https://arxiv.org/abs/2504.16054) · openpi — [repo](https://github.com/Physical-Intelligence/openpi)
* LIBERO — [paper](https://arxiv.org/abs/2306.03310), [code](https://github.com/Lifelong-Robot-Learning/LIBERO)

---

## Next

* Start: [Lecture 01 — The RL Problem and the Rollout Engine](Lecture-01.md)
* Back: [Track D — Robotics Guide](../Guide.md)

**Study method:** use the [shared lesson cycle and completion rubric](../../../Learning-Guide.md) with this course's existing labs. Predict first, test one variation, retain a failure case, and revisit the self-check after two days and one week.


## Carry this course into the physical-AI system

**Start from:** A validated simulator task, reward definition, and imitation/reference policy.

**Build and challenge:** Compare training methods at a fixed environment-step and compute budget. Keep evaluation seeds/scenes separate, check reward hacking, and report multiple seeds rather than a best run.

**Pass forward:** A frozen policy and training provenance for independent deployment evaluation. Continue with [the connected course](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).
