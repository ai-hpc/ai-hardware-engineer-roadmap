# Lecture 11: RL for LLMs and VLAs

## Overview

Think of how a student learns to write essays. First they read a whole library (**pre-training**). Then a teacher hands them a stack of model essays to copy (**supervised fine-tuning**). Last, they write their own essays and get them graded (**RL**). The grade can come from a person who picks the better of two essays, from an answer key, or from a teacher who checks each step of the argument. A good student writes four essays on the same prompt and learns most from the ones that beat their own average. A sensible teacher also keeps them on a leash: "improve, but don't turn into a different writer just to please the grader."

That is how chat LLMs are post-trained today. The same three stages describe π0.5 on our kitchen table: web-scale pre-training of the VLM, imitation on robot demonstrations, then practice against a simulator's success check. The recipe transfers almost unchanged with one exception. The LLM recipe needs \(\log \pi_\theta(a \mid o)\) for every action it takes, and a **flow-matching action expert does not give you that cheaply.** Most of this lecture is about that snag and the four families of workarounds people use. The rest is about systems: when every sample is a multi-billion-parameter forward pass, RL post-training is an inference-serving problem with a small optimizer attached.

By the end you should be able to:

* write the RLHF, GRPO, and DPO objectives and say which models each one keeps in GPU memory
* explain why group-relative advantages remove the critic, and when they give no gradient at all
* derive why a deterministic flow-matching sampler has no cheap log-likelihood, and turn it into a stochastic sampler whose per-step log-probs are exact Gaussians
* compare likelihood-based RL, noise-space steering, advantage-weighted / advantage-conditioned regression, and critic-guided selection for a flow VLA
* budget a VLA RL run: memory for policy + reference + optimizer, rollout vs learner GPU-hours, and the cost of each denoising step

---

## 1. Why it matters: one recipe, two domains

| | Chat LLM | π0.5 on the kitchen table |
|---|---|---|
| Pre-training | Next-token prediction on web text | VLM pre-training on web images + text (PaliGemma-class backbone) |
| Supervised fine-tune | Curated instruction / answer pairs | Robot demonstrations, per-suite fine-tunes (e.g., LIBERO) |
| **Score source** | Human preferences → reward model; or a verifier (unit tests, math checker) | Simulator success predicate (`bowl on plate`), free and exact in sim |
| **Compare against** | G answers to the *same prompt* | G episodes from the *same initial table layout* (same sim seed) |
| **Leash** | KL to the SFT model | KL to the original π0.5 checkpoint |
| **Hidden information** | Whole conversation as context | Recent frames only → give the critic privileged sim state ([Lecture 05](Lecture-05.md)) |
| Cost of one sample | One generation: prefill + hundreds of decode steps | One episode: \(T/H\) action chunks, each a VLM prefix + ~10 action-expert steps |
| Log-prob of the output | Free (sum of token log-probs) | **Not free** for a flow-matching head |

The bottom two rows are where the work is. The rest is the policy gradient from [Lecture 04](Lecture-04.md), the clipped surrogate from [Lecture 06](Lecture-06.md), and the KL leash, all reused.

---

## 2. Mental model: how LLMs are post-trained with RL

### 2.1 RLHF: preferences → reward model → PPO

Humans compare two responses \(y_w \succ y_l\) to the same prompt \(x\). A **Bradley-Terry** model turns a scalar reward \(r_\phi\) into a preference probability, and the reward model is fit by logistic regression:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big), \qquad
\mathcal{L}(\phi) = -\mathbb{E}\big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\big]
$$

The policy then maximizes reward minus a KL penalty to the SFT model \(\pi_{\text{ref}}\):

$$
\max_\theta \; \mathbb{E}_{x,\, y \sim \pi_\theta}\big[r_\phi(x, y)\big] - \beta\, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

optimized with PPO (InstructGPT). In memory during training: **policy** (trained), **reference** (frozen), **reward model** (frozen), **value/critic** (trained). Four models of similar size, two with optimizer state. That memory bill is what motivated everything in 2.3-2.4.

### 2.2 Verifiers and process rewards

When correctness can be checked (math answers, code with unit tests), skip the reward model and use the checker directly: **RL with verifiable rewards (RLVR)**. A verifier cannot be "fooled" the way a learned reward model can, though it can still be gamed if it checks the wrong thing (a test suite that only checks the output format).

* **Outcome reward:** one score at the end. Cheap, sparse, exactly our 0/1 success signal.
* **Process reward:** a score for each intermediate step from a learned per-step checker. Denser credit assignment, but it is another learned model in the loop and can be hacked. The robot analog is a learned progress estimator or a dense shaped reward.

### 2.3 The KL-regularized optimum and DPO

The objective in 2.1 has a closed-form maximizer for each prompt:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\big(r(x, y)/\beta\big)
$$

which is the "soft optimality" result from [Lecture 08](Lecture-08.md) again. Solving for \(r\) gives \(r(x,y) = \beta \log \frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)\). Substitute that into the Bradley-Terry loss and \(Z(x)\) cancels in the difference. The result is **DPO**, preference learning with no reward model and no sampling:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

Note what it still needs: \(\log \pi_\theta(y \mid x)\). Every method in this section does.

### 2.4 GRPO and RLOO: the group *is* the baseline

Sample \(G\) responses to the same prompt, score them \(r_1, \dots, r_G\), and use the group statistics as the baseline. **GRPO** (DeepSeekMath) uses a normalized advantage shared by every token of response \(i\):

$$
\hat A_i = \frac{r_i - \operatorname{mean}(r_{1:G})}{\operatorname{std}(r_{1:G}) + \epsilon}
$$

and a PPO-style clipped objective per token \(t\) with ratio \(\rho_{i,t} = \pi_\theta(y_{i,t} \mid x, y_{i,<t}) / \pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})\):

$$
J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|}\sum_{t} \Big(\min\big(\rho_{i,t}\hat A_i,\; \operatorname{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat A_i\big) - \beta\, \hat{D}_{\text{KL}}\Big)\right]
$$

with the per-token KL estimate \(\hat D_{\text{KL}} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1\), which is non-negative and unbiased for the KL under samples from \(\pi_\theta\).

Things to notice, all already in [Lecture 04](Lecture-04.md):

* **No critic.** The group mean is a Monte-Carlo estimate of \(V(x)\). Dropping the value network removes one model and its optimizer state, which is the memory saving the GRPO paper emphasizes.
* **RLOO** uses the *leave-one-out* mean \(b_i = \frac{1}{G-1}\sum_{j \ne i} r_j\). Because \(b_i\) does not depend on sample \(i\), the gradient estimate is exactly unbiased. The plain group-mean advantage works out to exactly \(\frac{G-1}{G}\) times the leave-one-out advantage, so it is unbiased up to that constant scale. The std division is what changes the weighting.
* **Dividing by the std** reweights prompts: a group with success rate near 0 or 1 has small std, so its few informative samples get large advantages. This is a choice you make, not something you get for free. Measure with and without it.
* **Zero-variance groups.** If all \(G\) rewards are equal (all 0 or all 1), then \(\hat A_i = 0\) for every sample and the group contributes nothing except the KL term. With binary rewards this is common. DAPO's "dynamic sampling" keeps drawing prompts until the batch is filled with groups that have mixed outcomes. Section 4 does the arithmetic for robots.

### 2.5 GRPO became a family

Few teams that describe their RL algorithm run the original GRPO unchanged. Paniego's August 2026 survey of public frontier-lab reports ("How frontier models train on outcomes in 2026") found the same pattern repeatedly: GRPO is the starting point, and each lab changes one or two things about it. The variants are small edits to the objective above:

| Variant | What it changes | Problem it targets |
|---|---|---|
| **DAPO** | Asymmetric clip (a higher upper bound, "clip-higher"); dynamic sampling (drop zero-variance groups, keep sampling); token-level loss averaging over the whole batch instead of per-response \(\frac{1}{\lvert y_i \rvert}\); shaping for overlong responses | Entropy collapse, wasted batches, long-sequence weighting |
| **Dr. GRPO** | Removes the per-response length normalization and the std division | A length bias: the per-response \(\frac{1}{\lvert y_i \rvert}\) gives each token of a short correct answer more weight, and each token of a long wrong answer less penalty |
| **GSPO** | Importance ratio and clipping at the **sequence** level (length-normalized sequence likelihood) instead of per token | Noisy per-token ratios; instability, notably for MoE models |
| **CISPO** (MiniMax-M1) | Clips the importance-sampling **weight** (stop-gradient) instead of clipping the token's update | PPO-style clipping zeroes the gradient of rare, high-ratio tokens; CISPO keeps every token contributing |

The tooling now encodes this. In TRL, `GRPOConfig(loss_type=...)` defaults to `"dapo"`, and the docs mark the original `"grpo"` loss as not recommended because of its length bias. `importance_sampling_level="sequence"` gives GSPO. The logged metric `frac_reward_zero_std` is the fraction of groups whose rewards all match: exactly the zero-variance fraction this lecture measures in Section 4. Mistral's Magistral report, quoted in the same survey, makes it a rule: filter out every zero-advantage group when forming a batch.

**Robot translation.** These choices have direct analogs for a chunked VLA, and you should make them deliberately:

* **Token ↔ denoising step / chunk.** "Token-level vs sequence-level ratio" becomes "per-denoising-step ratio (Section 3.3) vs one ratio per chunk or per episode". GSPO's argument, that per-element ratios are noisy when the reward arrives at the end, applies to a 10-step denoising chain too.
* **Length bias exists for robots.** In LIBERO, a success ends the episode early, and a failure runs to the step limit. If you average the loss per episode over its chunks (GRPO's \(\frac{1}{\lvert y_i \rvert}\)), each chunk of a short success gets more weight than each chunk of a long failure. Averaging over all chunks in the batch (DAPO-style) weights every decision equally. Neither choice is wrong, but it is a choice: log which one you used and ablate it.
* **Dynamic sampling = curriculum.** Dropping zero-variance groups and resampling is the per-layout version of the goldilocks sampler in [Lecture 12](Lecture-12.md).

### 2.6 Reward hacking

Optimize a proxy hard enough and the policy finds where the proxy is wrong: a reward model that likes long answers, a test suite with a hole. The KL leash is partly a defense here: it limits how far the policy can move toward the exploit. In simulation the proxy is the **success predicate**, and it can be wrong too. A "bowl on plate" check based on a contact flag could be satisfied by pinning the bowl against the plate's rim. Always watch rollout videos of the highest-reward episodes.

---

## 3. VLAs: where does \(\log \pi\) come from?

### 3.1 VLAs with an explicit action likelihood

If the policy emits discretized action tokens (the OpenVLA family bins each action dimension into tokens), then \(\log \pi_\theta(a \mid o)\) is a sum of token log-probs, the same as for text. GRPO and PPO apply directly: per-initial-state groups replace per-prompt groups, and the sim success check replaces the verifier. **SimpleVLA-RL** is an example of this line: an RL framework built on OpenVLA-OFT with VLA-specific trajectory sampling and parallelized multi-environment rendering. It reports strong LIBERO results and gains on RoboTwin, and it describes a "pushcut" phenomenon in which RL discovers behaviors absent from the demonstrations. (Read the paper for the numbers; they depend on the setup.)

### 3.2 Why a flow-matching head has no cheap likelihood

π0.5's action expert produces a chunk \(a \in \mathbb{R}^{P \times 7}\) (Lecture 03's \(P\) = predicted chunk length) by integrating a learned velocity field from noise. Convention for this lecture: \(\tau = 0\) is noise, \(\tau = 1\) is the action (π0's paper uses the opposite time direction; the math is the same with \(\tau \to 1-\tau\)):

$$
x_0 \sim \mathcal{N}(0, I), \qquad x_{k+1} = x_k + \Delta\, v_\theta(x_k, \tau_k, o), \qquad a = x_K, \quad \Delta = 1/K
$$

That sampler is **deterministic** given \(x_0\). The density of \(a\) follows from the instantaneous change-of-variables formula:

$$
\log p_\theta(a \mid o) = \log \mathcal{N}(x_0; 0, I) - \int_0^1 \nabla_x \cdot v_\theta(x_\tau, \tau, o)\, d\tau
$$

The divergence is the trace of a \(350 \times 350\) Jacobian for a 50×7 chunk, needed at every integration point, along a trajectory that has to be inverted from \(a\) back to \(x_0\), and then differentiated with respect to \(\theta\) for the policy gradient. Exact computation needs one vector-Jacobian product per action dimension (350) at every integration point. Stochastic (Hutchinson) trace estimators cut that to a few products, at the price of noise. For a multi-billion-parameter model in an RL loop, that is not workable. Hence the workarounds.

### 3.3 Workaround 1: make the sampler stochastic

Inject Gaussian noise at each integration step:

$$
x_{k+1} = \underbrace{x_k + \Delta\, v_\theta(x_k, \tau_k, o)}_{\mu_\theta(x_k, \tau_k, o)} + \sigma_k\, \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0, I)
$$

Now every denoising step is a Gaussian with an exact log-density:

$$
\log p_\theta(x_{k+1} \mid x_k, o) = -\frac{\lVert x_{k+1} - \mu_\theta(x_k, \tau_k, o) \rVert^2}{2\sigma_k^2} - d \log \sigma_k - \frac{d}{2}\log 2\pi
$$

Treat the \(K\) denoising steps as an **inner MDP** nested inside each environment step. The probability of a full trajectory (environment steps \(t\), denoising steps \(k\)) is a product of dynamics terms, which do not depend on \(\theta\), and these Gaussians. The log-derivative trick from Lecture 04 then gives, with the environment-step advantage \(A_t\):

$$
\nabla_\theta J = \mathbb{E}\left[\sum_t \sum_{k=0}^{K-1} \nabla_\theta \log p_\theta\big(x^{t}_{k+1} \mid x^{t}_k, o_t\big)\, A_t\right]
$$

This is an exact policy gradient **for the noisy sampler**. The marginal \(p_\theta(a \mid o)\) is never needed, because the action is a deterministic function of the chain \((x_0, \dots, x_K)\). A bonus: with a shared \(\sigma_k\), the KL between the current and reference policy's step distributions is closed form:

$$
\mathrm{KL}\big(\mathcal{N}(\mu_\theta, \sigma_k^2 I) \,\|\, \mathcal{N}(\mu_{\text{ref}}, \sigma_k^2 I)\big) = \frac{\lVert \mu_\theta - \mu_{\text{ref}} \rVert^2}{2\sigma_k^2}
$$

The published variants differ in how they choose the noise:

* **DPPO** (2024) did this for diffusion policies, treating the denoising chain as an MDP and fine-tuning with PPO. It is the precursor for robot control.
* **Flow-GRPO** (2025, text-to-image) converts the flow model's ODE into an SDE so that sampling explores, and it reduces the number of denoising steps used during training rollouts relative to inference.
* **ReinFlow** (2025) injects *learnable* noise into a flow policy's deterministic path, turning it into a discrete-time Markov process with tractable likelihoods. It reports working even at very few denoising steps, on locomotion and manipulation.
* **πRL** (2025) targets flow VLAs (π0 / π0.5) directly with two variants: **Flow-Noise**, where a learnable noise network makes the denoising process a discrete-time MDP with exact log-likelihoods, and **Flow-SDE**, an ODE-to-SDE conversion that forms a two-layer MDP coupling denoising with agent-environment interaction. It reports gains on standard and out-of-distribution evaluations.

The caveat: naive noise injection (the version in this lecture's lab) **changes the policy**. Adding \(\sigma_k \varepsilon\) to each Euler step inflates the variance of the final action. An SDE that preserves the original marginals needs a score-correction term. You train the noisy policy, and you have to check whether the deterministic sampler (\(\sigma = 0\)) inherits the improvement. Measure both.

**Workaround 1b: Flow Policy Optimization (FPO).** Instead of an exact likelihood, FPO builds a PPO-clip-compatible ratio from the **conditional flow-matching loss** (lower loss on an action ≈ higher likelihood), giving an advantage-weighted ratio objective that is agnostic to which sampler you use at train or inference time. It is cheaper per update than unrolling the chain, but it is a surrogate, not an exact ratio.

### 3.4 Workaround 2: steer the noise, freeze the policy (DSRL)

From [Lecture 10](Lecture-10.md): keep the flow policy frozen and train a small RL policy \(\pi_\psi(x_0 \mid o)\) that picks the **input noise**. The flow model maps any noise to a plausible action, so the RL agent cannot leave the data manifold. It needs only black-box sampling access to the base policy, and it is sample-efficient. The cost: the base weights do not change. If the final deliverable must be a single set of weights (a competition that accepts only the checkpoint, a deployment stack with no hook for the noise policy), you have to distill afterwards.

### 3.5 Workaround 3: regression on your own rollouts

Skip likelihoods entirely and use supervised losses, which a flow model already has:

* **Filtered BC / rejection sampling:** roll out, keep successful episodes, fine-tune on them with the ordinary flow-matching loss. This is the simplest baseline, and the one to beat.
* **Advantage-weighted regression:** weight each sample's flow-matching loss by \(\exp(A/\beta)\) (AWR from Lecture 10).
* **Advantage conditioning (π\*0.6 / RECAP).** Physical Intelligence's RECAP ("RL with Experience and Corrections via Advantage-conditioned Policies") trains a distributional value function. The reward is −1 per step and a failure penalty at the end, so the value predicts (negative) steps-to-completion. Each action gets a binarized improvement indicator, \(I_t = [A(o_t, a_t) > \epsilon_\ell]\), with a per-task threshold. The indicator is fed to the policy as text ("Advantage: positive / negative") before the action tokens. Training mixes conditional and unconditional likelihood terms, and for the flow expert the flow-matching loss stands in for the action log-likelihood. At inference the indicator is set to positive, optionally with classifier-free-guidance-style sharpening. Human corrections are always labeled positive. Because the update is supervised, it naturally ingests **heterogeneous data**: demos, autonomous rollouts, and teleoperated interventions.

### 3.6 Workaround 4: critic-guided selection, then distill

Sample \(M\) candidate chunks from the policy, score them with a learned \(Q(o, a)\), and execute the best: the sample-and-rank idea from [Lecture 07](Lecture-07.md) (IDQL-style). It costs \(M\) action-expert passes plus \(M\) critic passes per decision (the VLM prefix is shared). To get a single checkpoint back, distill the selected actions into the policy with the flow-matching loss.

### 3.7 Summary

| Family | Needs | Changes base weights? | Extra cost per decision | Main risk |
|---|---|---|---|---|
| Stochastic sampler + PPO/GRPO | Unrolled chain stored; K-step recompute at update | Yes | Learner: K action-expert passes with grad | Noisy policy ≠ deterministic policy; ratio instability |
| FPO | CFM-loss evaluations | Yes | A few CFM samples per action | Surrogate ratio quality |
| Noise steering (DSRL) | Black-box sampler, small actor | No (needs distill) | Small actor forward | Bounded by base policy's support |
| Filtered / AWR / advantage-conditioned | Flow-matching loss, maybe a critic | Yes | None at rollout | Slower improvement; threshold/temperature tuning |
| Sample-and-rank + distill | Critic, M samples | After distill | M expert + M critic passes | Critic exploitation (Lecture 10) |

---

## 4. Rollout systems for VLA RL

**Groups by seed.** A per-initial-state group is G environments reset with the same seed. The group mean estimates \(V(s_0)\) for that layout (Lecture 04). For binary success with per-layout success probability \(p\), the chance that a group carries no signal is

$$
P(\text{zero variance}) = p^G + (1-p)^G
$$

At \(p = 0.9\) and \(G = 8\) that is about 0.43: almost half your rollouts are wasted. Public π0.5 numbers on the standard LIBERO suites are already high, so this is the *normal* case for nominal layouts. Remedies: filter zero-variance groups and resample (DAPO-style dynamic sampling); spend rollouts on layouts with intermediate success, such as perturbed layouts ([Lecture 12](Lecture-12.md) makes this a curriculum); or increase \(G\) only for layouts with \(p\) near the edges.

**Colocated vs disaggregated.**

| | Colocated | Disaggregated |
|---|---|---|
| Layout | Same GPUs alternate: rollout phase, then train phase | Rollout GPUs (inference) and trainer GPUs are separate |
| Weight sync | In-place (same memory) | Every iteration: \(\approx 2P\) bytes in bf16 (~6 GB for 3 B params); LoRA: adapters only, MBs |
| Utilization | Idle simulator/CPU during training; must offload optimizer state to fit large inference batches | Pipelining possible; rollouts can run one policy version behind (off-policy lag, handled by the ratio) |
| Fits | Single node, LoRA | Multi-node, full-parameter |

Frameworks such as RLinf schedule exactly these choices for embodied and reasoning RL. Its paper reports end-to-end throughput gains from workflow-aware scheduling, which shows that system design here matters about as much as the algorithm.

### 4.1 When the environment is a machine: agentic RL infrastructure

LLM agents (coding, browsing, terminal use) are trained the same way, but their environment is no longer a simulator object in the trainer's memory: it is a **container or microVM per rollout**, booted, used once, and destroyed, or checkpointed and resumed for very long tasks. Paniego's survey of fifteen 2025-2026 lab reports ("One sandbox per rollout, or how labs run RL for agents in 2026") describes the same four layers in every stack that is documented at all. The robot pipeline in this course has exactly the same four layers:

| Layer | LLM agent | This course (LIBERO + π0.5) |
|---|---|---|
| **Task + verifier** | Repository + test suite; search question + checkable answer | BDDL task + suite success predicate |
| **Contract** | `reset` / `step` (OpenEnv speaks this over WebSockets); increasingly the agent *harness* itself, white-box (rebuilt) or black-box (proxied) | Gymnasium `reset` / `step`; openpi's policy server ↔ env client |
| **Sandbox** | One container per rollout; startup latency and failure robustness matter | One simulator worker (CPU process) per env; MuJoCo state is cheap to save, restore, and fork |
| **Trainer** | Consumes rollouts, ideally without waiting for the slowest one | Same |

The hardware lesson that transfers directly is the **long tail**. In a synchronous loop, every update waits for the slowest rollout in the batch. Agent rollouts vary from seconds to hours. LIBERO episodes vary too: successes terminate early, failures run to the 220-520-step limit. So **a synchronous batch is always waiting on its failures**, and the better the policy gets, the more idle time the long tail costs. The fix the lab reports describe (GLM-5's slime, TRL's experimental `AsyncGRPOTrainer`) is to **decouple generation from training**: rollout workers run at their own pace on their own engines, and the trainer consumes whatever has finished. The price is **staleness**: some rollouts come from a policy one or two updates old. The importance ratio in the clipped objective is what corrects for it, and a bound on maximum staleness keeps the correction small. Measure: trainer idle fraction, rollout-duration histogram (successes vs failures), and staleness (policy versions behind) per batch.

**RL specialists, then distill.** The same survey of reports finds a second pattern: labs run RL on *domain specialists*, then transfer the gains into one final model with on-policy distillation (DeepSeek-V4, MiMo-V2-Flash, LFM2.5 are the examples cited). Outcome rewards find the behavior; distillation moves it with a dense per-token signal. The robot version is natural when only one checkpoint can be submitted: RL per suite or per failure mode, then distill the specialists into a single π0.5 with the flow-matching loss on their rollouts (Section 3.6, and Phase 5 of the [capstone](Lecture-14.md)).

**Robot connection.** For the capstone, the "prompt" is a LIBERO task plus a seeded initial layout, the verifier is the suite's success predicate, and the reference is the base π0.5 checkpoint. The perturbed-layout tests are where the policy's success rate sits strictly between 0 and 1, which is exactly where group-relative methods get signal.

---

## 5. The hardware view: regime C, where inference dominates

**Per-chunk cost.** One action chunk costs

$$
t_{\text{chunk}} \approx t_{\text{prefix}}(\text{VLM over images + text}) + K \cdot t_{\text{expert}}
$$

Cutting \(K\) from 10 to 5 saves only the second term. Measure both terms with the Lecture 01 `time_forward` harness before assuming a speedup. Any change of \(K\) or sampler is a policy change: gate it with the [action-parity harness](../VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-02.md).

**Learner cost of likelihood methods.** Recomputing \(\log p_\theta\) of a stored chain needs all \(K\) action-expert steps with gradients, so activation memory scales with \(K\). If the VLM prefix is frozen (or LoRA'd only in the expert), compute it once under `no_grad` and reuse it across the \(K\) steps. Otherwise checkpoint it. A common trick is to compute ratios on a random subset of denoising steps per minibatch.

**Memory budget** (π0-class model; π0's paper reports 3.3 B parameters; use \(P \approx 3 \times 10^9\)):

| Component | Full-parameter (AdamW, mixed precision) | LoRA |
|---|---|---|
| Policy weights (bf16) | ~6 GB | ~6 GB (frozen base) |
| Grads + fp32 master + Adam moments | ~42 GB (≈ 14 B/param) | adapters only, < 1 GB |
| Reference policy | +6 GB frozen copy | **free**: disable adapters |
| Critic | VLM-sized: another ~48 GB; small head on shared features: ≈ 0 | small head |
| Activations (K-step recompute) | batch × K dependent: measure | same |

This is why GRPO-style (no critic) and LoRA (free reference) are the default single-80 GB configuration, and why full-parameter PPO with a VLM-sized critic is a multi-GPU job.

**GPU-hour split.** For \(E\) episodes per iteration:

$$
\text{GPU-h}_{\text{rollout}} \approx \frac{E \cdot (T/H) \cdot t_{\text{chunk}}(N)}{N \cdot 3600}, \qquad
\text{GPU-h}_{\text{learn}} \approx \frac{\text{epochs} \cdot E \cdot (T/H) \cdot t_{\text{update}}}{B \cdot 3600}
$$

Expect the rollout term to dominate unless you run many epochs. Batching across environments (larger \(N\)) is the first lever; chunk-execution horizon \(H\) is the second.

---

## 6. Build it: GRPO on your flow policy (rung 2)

Start from the chunked flow-matching policy for `PickCube-v1` (state observations) from [Lecture 03](Lecture-03.md), with interface `policy(x, tau, obs) -> velocity` and chunks `x` of shape `(B, H, 7)`. In this lab `H` is the full predicted chunk (Lecture 03's \(P\)), and the whole chunk is executed. L03's `Flow.forward(o, x, t)` takes a flat `x` of size `P * 7` and `t` of shape `(B, 1)`, so wrap it to this interface (reshape `x`, unsqueeze `tau`). If your L03 code uses the opposite time convention, flip `tau`.

### Lab 1a — Noisy sampler and exact chain log-probs

```python
# vla_rl/flow_rl.py
import math, torch

def sample_chain(policy, obs, H, A, K, sigma):
    """Noisy Euler sampler. Returns action chunk (B,H,A) and chain (B,K+1,H,A)."""
    x = torch.randn(obs.shape[0], H, A, device=obs.device)
    xs, dt = [x], 1.0 / K
    for k in range(K):
        tau = torch.full((obs.shape[0],), k * dt, device=obs.device)
        x = x + dt * policy(x, tau, obs) + sigma[k] * torch.randn_like(x)
        xs.append(x)
    return x, torch.stack(xs, 1)

def chain_logprob(policy, obs, chain, sigma):
    """Per-step Gaussian log-probs (B,K) and step means (B,K,H,A). One batched pass over all K steps."""
    B, K = chain.shape[0], chain.shape[1] - 1
    x_k, x_next = chain[:, :-1].flatten(0, 1), chain[:, 1:].flatten(0, 1)
    tau = (torch.arange(K, device=obs.device) / K).repeat(B)
    mu = x_k + (1.0 / K) * policy(x_k, tau, obs.repeat_interleave(K, 0))
    s = sigma.repeat(B).view(-1, 1, 1)
    logp = (-0.5 * ((x_next - mu) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)).sum((1, 2))
    return logp.view(B, K), mu.view(B, K, *chain.shape[2:])
```

Sanity check before any RL: `chain_logprob` on a chain freshly produced by `sample_chain` (same weights) must give a ratio of exactly 1, and the per-step log-prob must match `torch.distributions.Normal(mu, s).log_prob(x_next).sum((1,2))`.

### Lab 1b — Per-initial-state groups and the GRPO loss

```python
def group_advantages(success, G, normalize_std=True):
    r = success.view(-1, G)
    adv = r - r.mean(1, keepdim=True)
    std = r.std(1, keepdim=True)
    if normalize_std:
        adv = adv / (std + 1e-6)
    informative = (std.squeeze(1) > 0)                      # groups with mixed outcomes
    return adv.view(-1), informative.repeat_interleave(G)

def grpo_loss(policy, obs, chain, old_logp, mu_ref, adv, sigma, clip=0.2, beta=0.01):
    logp, mu = chain_logprob(policy, obs, chain, sigma)     # (B,K)
    ratio = torch.exp(logp - old_logp)                      # one ratio per denoising step
    A = adv[:, None]
    pg = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean()
    kl = (((mu - mu_ref) ** 2).sum((2, 3)) / (2 * sigma[None] ** 2)).mean()   # closed-form Gaussian KL
    clip_frac = ((ratio - 1).abs() > clip).float().mean()
    return pg + beta * kl, dict(pg=pg.item(), kl=kl.item(), clip_frac=clip_frac.item())
```

### Lab 1c — Rollouts with seeded groups

ManiSkill3's GPU `reset` accepts one seed per parallel environment, so a group is the same seed repeated G times:

```python
# vla_rl/rollout.py
@torch.no_grad()
def collect_groups(env, policy, ref, init_seeds, G, H, A, K, sigma, n_chunks):
    obs, _ = env.reset(seed=[s for s in init_seeds for _ in range(G)])
    assert obs.view(len(init_seeds), G, -1).std(1).max() < 1e-4, "group members must share a layout"
    success = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
    buf = {k: [] for k in ("obs", "chain", "logp", "mu_ref")}
    for _ in range(n_chunks):                                # execute the full chunk: clean credit
        chunk, chain = sample_chain(policy, obs, H, A, K, sigma)
        logp, _ = chain_logprob(policy, obs, chain, sigma)
        _, mu_ref = chain_logprob(ref, obs, chain, sigma)
        for k, v in zip(buf, (obs, chain, logp, mu_ref)):
            buf[k].append(v)
        for h in range(H):
            obs, _, _, _, info = env.step(chunk[:, h].clamp(-1, 1))
            success |= info["success"].bool()
    return {k: torch.stack(v, 1).flatten(0, 1) for k, v in buf.items()}, success.float()
```

(The buffer is flattened env-major, so each environment's `n_chunks` decisions are contiguous; repeat the per-episode advantage with `repeat_interleave(n_chunks)`.) Use `env = gym.make("PickCube-v1", num_envs=num_groups*G, obs_mode="state")` and keep `n_chunks * H` within the task's time limit.

### Lab 1d — Training loop and the filtered-BC baseline

```python
ref = copy.deepcopy(policy).eval().requires_grad_(False)
opt = torch.optim.AdamW(policy.parameters(), lr=1e-5)
sigma = torch.full((K,), 0.05, device="cuda")              # sweep {0.02, 0.05, 0.1}
for it in range(iters):
    seeds = rng_seeds(num_groups)                           # your sampler; disjoint from eval seeds
    data, success = collect_groups(env, policy, ref, seeds, G, H, 7, K, sigma, n_chunks)
    adv, keep = group_advantages(success, G)
    adv, keep = adv.repeat_interleave(n_chunks), keep.repeat_interleave(n_chunks)
    log(zero_var_frac=1 - keep.float().mean(), success=success.mean())
    idx = keep.nonzero().squeeze(1)                         # drop zero-variance groups
    for _ in range(2):                                      # epochs
        for mb in idx[torch.randperm(len(idx))].split(256):
            loss, stats = grpo_loss(policy, data["obs"][mb], data["chain"][mb],
                                    data["logp"][mb], data["mu_ref"][mb], adv[mb], sigma)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
```

Filtered BC on the same rollout budget: collect the same number of episodes per iteration, keep the executed chunks from successful episodes, and take the same number of optimizer steps with the plain flow-matching loss:

```python
def flow_matching_loss(policy, obs, a1):                    # tau=0 noise, tau=1 action
    x0, tau = torch.randn_like(a1), torch.rand(a1.shape[0], device=a1.device)
    xt = (1 - tau)[:, None, None] * x0 + tau[:, None, None] * a1
    return ((policy(xt, tau, obs) - (a1 - x0)) ** 2).mean()
```

Evaluate every few iterations with **deterministic** sampling (\(\sigma = 0\)) and with the training \(\sigma\), on (a) held-out seeds from the training distribution and (b) **perturbed initial poses** outside it (the same perturbation protocol you used in [Lecture 02](Lecture-02.md) and [Lecture 06](Lecture-06.md)).

### Lab 2 (optional, rung 3) — One RL iteration of π0.5 on LIBERO

If you have an 80 GB GPU: use a framework that already wires π0/π0.5 + LIBERO + PPO/GRPO (RLinf lists all three) and run *one* iteration on a single LIBERO task. The goal is not a better policy. It is a measured split of one iteration: rollout seconds, log-prob recompute seconds, optimizer seconds, weight-sync seconds, peak memory. Follow the framework's own LIBERO example rather than inventing flags.

---

## 7. Use it in the real stack

* **LLM RL frameworks**, worth reading for their rollout/learner split: [TRL](https://github.com/huggingface/trl) (has a GRPO trainer), [verl](https://github.com/volcengine/verl) (DAPO was released on it), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
* **VLA RL:** [RLinf](https://github.com/RLinf/RLinf) (embodied + reasoning RL; lists π0/π0.5 via openpi, OpenVLA, LIBERO, ManiSkill, PPO and GRPO), SimpleVLA-RL (OpenVLA-OFT), and the code releases of ReinFlow and πRL.
* **Policy:** [openpi](https://github.com/Physical-Intelligence/openpi) π0.5 LIBERO checkpoint as the reference model; its LoRA fine-tuning configs are the natural place to attach adapters for the "reference = adapters disabled" trick.

---

## 8. Measure it

| Metric | Definition | Why it matters |
|---|---|---|
| Success, nominal / perturbed | Deterministic eval on held-out seeds and on perturbed poses | The real target; perturbed shows whether RL improved understanding or memorized layouts |
| Success, \(\sigma=0\) vs \(\sigma>0\) | Same policy, two samplers | Did the deterministic policy inherit what the noisy one learned? |
| Zero-variance group fraction | Groups with all-equal outcomes / total | Wasted rollouts; compare with \(p^G + (1-p)^G\) |
| KL to reference | Mean closed-form step KL | Leash tightness; spikes precede collapse |
| Clip fraction, ratio stats | From `grpo_loss` | Step size sanity (Lecture 06) |
| Rollout / learner / sync seconds per iteration | Wall-clock split | Which term to optimize next |
| GPU-hours to +X% success | Integrated over the run | The number another team needs to budget |
| Peak GPU memory | `torch.cuda.max_memory_allocated` | Headroom for G, N, K |

Report GRPO vs filtered BC at **equal rollout budget** (episodes) and at **equal wall-clock**, over ≥ 3 seeds.

---

## 9. Ship it

Commit `vla_rl/` containing:

* `flow_rl.py` (sampler, chain log-prob, GRPO loss), `rollout.py`, `train_grpo.py`, `train_filtered_bc.py`
* `results.csv`: one row per (method, seed, iteration) with every metric above
* `COMPARISON.md`: GRPO vs filtered BC table (nominal, perturbed, \(\sigma=0\) vs \(\sigma>0\), zero-variance fraction, GPU-hours), plus one paragraph on which you would use for π0.5 and why
* `gpu_hours.csv`: rollout / learner / eval split, plus the rung-3 single-iteration breakdown if you ran it
* `curves.png`: success vs episodes and vs wall-clock, both methods, ≥ 3 seeds

---

## Exit criteria

You can move on when you can:

* write GRPO's advantage and loss, and state which models RLHF-PPO, GRPO-RLVR, and DPO keep in memory
* derive the inner-MDP policy gradient for a noisy flow sampler and the closed-form step KL
* explain the four workaround families for flow VLAs and pick one for a given constraint (weights-only deliverable, human intervention data, tight memory)
* show your measured zero-variance fraction and relate it to \(p^G + (1-p)^G\)
* produce a GPU-hour split for your run and name the dominant term

---

## Self-check

1. Base π0.5 succeeds 92% of the time on a LIBERO task's nominal layouts. With \(G = 8\), roughly what fraction of groups give zero policy gradient? Name two ways to spend those rollouts better.
2. You must post-train a ~3 B-parameter flow VLA on one 80 GB GPU. Compare full-parameter PPO with a VLM-sized critic, full-parameter GRPO, and LoRA GRPO by what must sit in memory. Which fits, and what does LoRA give you for free besides small optimizer state?
3. You trained with \(\sigma_k = 0.1\) and success under the noisy sampler rose from 60% to 80%, but deterministic (\(\sigma = 0\)) success only went from 62% to 65%. Give a mechanism that explains the gap and two fixes.
4. A teammate proposes cutting rollout denoising steps from 10 to 2 "to make RL 5× faster." Using \(t_{\text{chunk}} \approx t_{\text{prefix}} + K t_{\text{expert}}\), what speedup do you actually expect if the prefix is 60% of the chunk at \(K = 10\)? What must you check before trusting the trained policy at \(K = 10\) again?
5. Your data includes teleoperated corrections recorded during autonomous runs. Why does an advantage-conditioned / weighted-regression method (RECAP-style) handle them naturally, while on-policy GRPO with noise-injected log-probs does not?
6. After 200 GRPO iterations, success on the sim predicate is 97%, but videos show the bowl balanced on the plate's rim. What went wrong, which term in your objective should have limited it, and how do you detect this automatically next time?

---

## References

* Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022 — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
* Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (introduces GRPO), 2024 — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
* Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (RLOO), 2024 — [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)
* Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023 — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
* Zheng et al., "Group Sequence Policy Optimization" (GSPO), 2025 — [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)
* MiniMax, "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention" (introduces CISPO), 2025 — [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)
* Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO), 2025 — [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
* TRL `GRPOTrainer` docs (loss types, sequence-level importance sampling, logged metrics) — [docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
* OpenEnv (Gymnasium-style `reset`/`step`/`state` for agentic RL environments) — [repo](https://github.com/meta-pytorch/OpenEnv)
* S. Paniego, "How frontier models train on outcomes in 2026" (Aug 2026) and "One sandbox per rollout, or how labs run RL for agents in 2026" (Sep 2026), Hugging Face community blog — surveys of public lab reports on GRPO variants, RL-then-distill pipelines, and environment infrastructure
* Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (dynamic sampling), 2025 — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
* Liu et al., "Flow-GRPO: Training Flow Matching Models via Online RL", 2025 — [arXiv:2505.05470](https://arxiv.org/abs/2505.05470)
* Zhang et al., "ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning", 2025 — [arXiv:2505.22094](https://arxiv.org/abs/2505.22094)
* Chen et al., "πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models", 2025 — [arXiv:2510.25889](https://arxiv.org/abs/2510.25889)
* McAllister et al., "Flow Matching Policy Gradients" (FPO), 2025 — [arXiv:2507.21053](https://arxiv.org/abs/2507.21053)
* Li et al., "SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning", 2025 — [arXiv:2509.09674](https://arxiv.org/abs/2509.09674)
* Wagenmaker et al., "Steering Your Diffusion Policy with Latent Space Reinforcement Learning" (DSRL), 2025 — [arXiv:2506.15799](https://arxiv.org/abs/2506.15799)
* Physical Intelligence, "π\*0.6: a VLA That Learns From Experience" (RECAP), 2025 — [arXiv:2511.14759](https://arxiv.org/abs/2511.14759), [blog](https://pi.website/blog/pistar06)

Also useful: Ren et al., "Diffusion Policy Policy Optimization" (DPPO), 2024 — [arXiv:2409.00588](https://arxiv.org/abs/2409.00588); Yu et al., "RLinf", 2025 — [arXiv:2509.15965](https://arxiv.org/abs/2509.15965).

---

## Next in this special course

* Next: [Lecture 12 — Exploration, Curricula, Skills, and Multi-Task RL](Lecture-12.md)
* Previous: [Lecture 10 — Offline RL and Offline-to-Online Fine-Tuning](Lecture-10.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
