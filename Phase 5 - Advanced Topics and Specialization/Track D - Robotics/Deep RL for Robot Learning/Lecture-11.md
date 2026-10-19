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
| **Score source** | Human preferences → reward model; or a verifier (unit tests, math checker) | Simulator success predicate; inexpensive to evaluate, but only as correct as its task definition and implementation |
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

A common RLHF-PPO implementation uses a trainable policy, frozen reference, frozen reward model, and trainable value estimator. They need not be four equally sized or separate networks: sharing, offloading, model size, and the reward source change the memory budget. GRPO removes the learned value estimator; it does not remove all rollout or reward costs.

### 2.2 Verifiers and process rewards

When a checker exists, its result can supply an RL reward. Unit tests, symbolic checks, and simulator success predicates each have a coverage boundary; implementations can contain exploitable bugs and incomplete specifications. Record what is actually verified. A deterministic checker is not a guarantee that the intended real-world task was solved.

* **Outcome reward:** one score at the end. Cheap, sparse, exactly our 0/1 success signal.
* **Process reward:** a score for each intermediate step from a learned per-step checker. Denser credit assignment, but it is another learned model in the loop and can be hacked. The robot analog is a learned progress estimator or a dense shaped reward.

### 2.3 The KL-regularized optimum and DPO

The objective in 2.1 has a closed-form maximizer for each prompt:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\big(r(x, y)/\beta\big)
$$

which is the "soft optimality" result from [Lecture 08](Lecture-08.md) again. Solving for \(r\) gives \(r(x,y) = \beta \log \frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)\). Substitute that into the Bradley-Terry loss and \(Z(x)\) cancels in the difference. The result is **DPO**, preference learning without a separately trained reward model or on-policy generation inside the basic offline update; it still needs a preference dataset, whose construction may involve sampling:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

Note what it still needs: \(\log \pi_\theta(y \mid x)\). Every method in this section does.

### 2.4 GRPO and RLOO: the group *is* the baseline

Sample \(G\) responses to the same prompt, score them \(r_1, \dots, r_G\), and use the group statistics as the baseline. **GRPO** (DeepSeekMath) uses a normalized advantage shared by every token of response \(i\):

$$
\hat A_i = \frac{r_i - \operatorname{mean}(r_{1:G})}{\operatorname{std}(r_{1:G}) + \epsilon}
$$

For each token \(t\), define the current-to-old probability ratio:

\[
\rho_{i,t} = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}
{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}.
\]

Then use a PPO-style clipped objective:

$$
J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|}\sum_{t} \Big(\min\big(\rho_{i,t}\hat A_i,\; \operatorname{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat A_i\big) - \beta\, \hat{D}_{\text{KL}}\Big)\right]
$$

with the per-token KL estimate \(\hat D_{\text{KL}} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1\), which is nonnegative for a finite positive ratio and has the stated expectation under samples from the **current** policy with matching support. Reusing samples from an old policy changes that expectation unless corrected; differentiating a sampled expression also requires the correct objective/weighting. Do not call every implementation of this penalty an unbiased current-policy KL estimator.

Things to notice, all already in [Lecture 04](Lecture-04.md):

* **No critic.** The group mean is a Monte-Carlo estimate of \(V(x)\). Dropping the value network removes one model and its optimizer state, which is the memory saving the GRPO paper emphasizes.
* **RLOO** uses the *leave-one-out* mean \(b_i = \frac{1}{G-1}\sum_{j \ne i} r_j\). Because \(b_i\) does not depend on sample \(i\), the gradient estimate is exactly unbiased. The plain group-mean advantage works out to exactly \(\frac{G-1}{G}\) times the leave-one-out advantage, so the unnormalized group-mean estimator has a known multiplicative bias relative to the return gradient. For G=2 it is half the leave-one-out estimator. This identity assumes conditionally independent samples from the same policy at the same initial condition; std normalization, clipping, and selecting groups introduce additional changes.
* **Dividing by the std** reweights prompts: a group with success rate near 0 or 1 has small std, so its few informative samples get large advantages. This is a choice you make, not something you get for free. Measure with and without it.
* **Zero-variance groups.** If all \(G\) rewards are equal (all 0 or all 1), then \(\hat A_i = 0\) for every sample and the group contributes nothing except the KL term. With binary rewards this is common. DAPO's "dynamic sampling" keeps drawing prompts until the batch is filled with groups that have mixed outcomes. Section 4 does the arithmetic for robots.

### 2.5 GRPO became a family

GRPO is a family of related objectives. Compare the actual loss, masks, sampling distribution, and normalization in a pinned implementation; sharing an algorithm name does not make two training runs equivalent. The primary papers below describe these representative changes:

| Variant | What it changes | Problem it targets |
|---|---|---|
| **DAPO** | Asymmetric clip (a higher upper bound, "clip-higher"); dynamic sampling (drop zero-variance groups, keep sampling); token-level loss averaging over the whole batch instead of per-response \(\frac{1}{\lvert y_i \rvert}\); shaping for overlong responses | Entropy collapse, wasted batches, long-sequence weighting |
| **Dr. GRPO** | Removes the per-response length normalization and the std division | A length bias: the per-response \(\frac{1}{\lvert y_i \rvert}\) gives each token of a short correct answer more weight, and each token of a long wrong answer less penalty |
| **GSPO** | Importance ratio and clipping at the **sequence** level (length-normalized sequence likelihood) instead of per token | Noisy per-token ratios; instability, notably for MoE models |
| **CISPO** (MiniMax-M1) | Clips the importance-sampling **weight** (stop-gradient) instead of clipping the token's update | PPO-style clipping zeroes the gradient of rare, high-ratio tokens; CISPO keeps every token contributing |

The checked [TRL configuration](https://github.com/huggingface/trl/blob/6d6901c5e3963108076131ac4979d344566b13a2/trl/trainer/grpo_config.py) defaults to `loss_type="dapo"`. Its `importance_sampling_level="sequence"` uses a length-normalized sequence ratio; reproducing a full GSPO recipe also requires its other loss, clipping, and sampling settings. Pin the package/commit rather than treating this as a stable default. Log zero-variance groups and any selection/resampling policy; filtering them changes which initial conditions contribute to training.

**Robot translation.** These choices have direct analogs for a chunked VLA, and you should make them deliberately:

* **Token ↔ denoising step / chunk.** "Token-level vs sequence-level ratio" becomes "per-denoising-step ratio (Section 3.3) vs one ratio per chunk or per episode". GSPO's argument, that per-element ratios are noisy when the reward arrives at the end, applies to a 10-step denoising chain too.
* **Length bias exists for robots.** In LIBERO, a success ends the episode early, and a failure runs to the step limit. If you average the loss per episode over its chunks (GRPO's \(\frac{1}{\lvert y_i \rvert}\)), each chunk of a short success gets more weight than each chunk of a long failure. Averaging over all chunks in the batch (DAPO-style) weights every decision equally. Neither choice is wrong, but it is a choice: log which one you used and ablate it.
* **Dynamic sampling = curriculum.** Dropping zero-variance groups and resampling is the per-layout version of the goldilocks sampler in [Lecture 12](Lecture-12.md).

### 2.6 Reward hacking

Optimize a proxy hard enough and the policy finds where the proxy is wrong: a reward model that likes long answers, a test suite with a hole. The KL leash is partly a defense here: it limits how far the policy can move toward the exploit. In simulation the proxy is the **success predicate**, and it can be wrong too. A "bowl on plate" check based on a contact flag could be satisfied by pinning the bowl against the plate's rim. Always watch rollout videos of the highest-reward episodes.

---

## 3. VLAs: where does \(\log \pi\) come from?

### 3.1 VLAs with an explicit action likelihood

The original OpenVLA predicts discretized action tokens. For that specific autoregressive policy, the probability of the sampled token sequence factors into conditional token probabilities. Preserve the exact sampling rule, masks, and old log-probabilities before applying a policy-gradient update.

OpenVLA-OFT is a different action-head contract: its checked quickstart loads an MLP head for **continuous actions trained with L1 regression**. That head does not supply categorical action-token likelihoods merely because the backbone came from OpenVLA. An RL adaptation must define its stochastic policy and scored random variables explicitly. SimpleVLA-RL supports OpenVLA/OpenVLA-OFT variants, but its model modifications and loss implementation must be inspected rather than inferred from the model name. [Pinned OFT inference example](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/README.md), [pinned SimpleVLA-RL code map](https://github.com/PRIME-RL/SimpleVLA-RL/blob/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/README.md).

### 3.2 Why a flow-matching head has no cheap likelihood

For a seven-dimensional robot adapter, a flow action expert produces a chunk \(a \in \mathbb{R}^{P \times 7}\); other embodiments and internal padded action widths differ. Here P denotes predicted chunk length (Lecture 03); generation integrates a learned velocity field from noise. Convention for this lecture: \(\tau = 0\) is noise, \(\tau = 1\) is the action (π0's paper uses the opposite time direction; the math is the same with \(\tau \to 1-\tau\)):

$$
x_0 \sim \mathcal{N}(0, I), \qquad x_{k+1} = x_k + \Delta\, v_\theta(x_k, \tau_k, o), \qquad a = x_K, \quad \Delta = 1/K
$$

The Euler sampler is **deterministic given its initial noise**; fresh initial noise still makes repeated calls stochastic. For an ideal continuous, invertible ODE flow, the density follows from the instantaneous change-of-variables formula:

$$
\log p_\theta(a \mid o) = \log \mathcal{N}(x_0; 0, I) - \int_0^1 \nabla_x \cdot v_\theta(x_\tau, \tau, o)\, d\tau
$$

For a 50×7 mathematical action space, the divergence is the trace of a 350×350 Jacobian along the continuous flow. Exact reverse-mode trace evaluation can require one vector-Jacobian product per dimension, while stochastic trace estimates trade cost for noise. The exact density of a finite Euler map instead involves discrete Jacobian determinants and invertibility conditions; the continuous formula is not automatically its exact likelihood. These costs motivate surrogate or stochastic-chain constructions, but whether a method is affordable requires measurement.

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

This likelihood-ratio identity applies to the **noisy sampler** with parameter-independent initial noise, correct return/discount weights, and a valid action-independent baseline. Replacing the return by estimated advantages, using old-policy data, or clipping ratios makes the practical update an approximation. The marginal \(p_\theta(a \mid o)\) need not be computed for that latent-chain estimator, because the action is a deterministic function of the chain \((x_0, \dots, x_K)\). A bonus: with a shared \(\sigma_k\), the KL between the current and reference policy's step distributions is closed form:

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

From [Lecture 10](Lecture-10.md): keep the flow policy frozen and train a smaller policy over its input noise. This limits actions to the frozen generator's image, but it does **not** guarantee a data-manifold or safety constraint: unusual noise and out-of-distribution observations can still produce poor actions. Evaluate the induced action distribution and closed-loop behavior. Distillation into another policy requires its own parity check.

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

At \(p = 0.9\) and \(G = 8\) that is about 0.43: about 43% of groups have no reward-relative signal under the independent Bernoulli assumption. They can still inform evaluation, a critic, or a regularizer. The actual fraction depends on the checkpoint and layout distribution; measure it. Remedies: filter zero-variance groups and resample (DAPO-style dynamic sampling); spend rollouts on layouts with intermediate success, such as perturbed layouts ([Lecture 12](Lecture-12.md) makes this a curriculum); or increase \(G\) only for layouts with \(p\) near the edges.

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

Synchronous learning waits for its slowest participating rollout, which may be a failure, a long success, or a stalled worker. Separating rollout generation and learning can reduce idle time but introduces policy-version staleness and possibly biased completion-based sampling. Per-action importance ratios do not fully correct a changed state/trajectory distribution, and a fixed number of stale versions does not bound KL. Preserve behavior-policy scores and versions, cap or reject stale data under a measured policy-change rule, and evaluate the complete off-policy correction/selection scheme. Record idle time, duration distributions, and actual divergence as well as version lag.

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
| Reference policy | +6 GB frozen copy | No duplicate base if adapters-off exactly equals the intended frozen reference |
| Critic | VLM-sized: another ~48 GB; small head on shared features: smaller, but nonzero training/activation cost | small head |
| Activations (K-step recompute) | batch × K dependent: measure | same |

This is an illustrative memory ledger, not an 80-GB fit guarantee. Adapter count, optimizer/precision implementation, sequence and image sizes, rollout storage, critic design, and activation checkpointing must be measured. Disabling adapters recovers the intended reference only if the shared base is immutable and the reference adapter state is exactly represented; an already adapted starting policy may need a frozen adapter copy.

**GPU-hour split.** For \(E\) episodes per iteration:

$$
\text{GPU-h}_{\text{rollout}} \approx \frac{E \cdot (T/H) \cdot t_{\text{chunk}}(N)}{N \cdot 3600}, \qquad
\text{GPU-h}_{\text{learn}} \approx \frac{\text{epochs} \cdot E \cdot (T/H) \cdot t_{\text{update}}}{B \cdot 3600}
$$

These expressions estimate wall-hours for a single GPU under the stated batching assumptions. Multiply each phase's wall-hours by its allocated GPU count for GPU-hours; include simulator, synchronization, idle time, and changing episode lengths. Measure which term dominates. Batching across environments (larger \(N\)) is the first lever; chunk-execution horizon \(H\) is the second.

---

## 6. Build it: GRPO on your flow policy (rung 2)

**Execution boundary:** the mathematical functions in Labs 1a–1b are checked on CPU by the optional tests linked from [CPU-Lab.md](CPU-Lab.md). The simulator rollout/training exercise below is a design to implement and validate, not a supplied or executed PickCube trainer.

Start from the chunked flow-matching policy for `PickCube-v1` (state observations) from [Lecture 03](Lecture-03.md), with interface `policy(x, tau, obs) -> velocity` and chunks `x` of shape `(B, H, 7)`. In this lab `H` is the full predicted chunk (Lecture 03's \(P\)), and the whole chunk is executed. L03's `Flow.forward(o, x, t)` takes a flat `x` of size `P * 7` and `t` of shape `(B, 1)`, so wrap it to this interface (reshape `x`, unsqueeze `tau`). If your L03 code uses the opposite time convention, flip `tau`.

### Lab 1a — Noisy sampler and exact chain log-probs

```python
# vla_rl/flow_rl.py
import math, torch

def _sigma_vector(sigma, K, reference, allow_zero):
    sigma = torch.as_tensor(sigma, dtype=reference.dtype, device=reference.device)
    if sigma.shape != (K,) or not torch.isfinite(sigma).all():
        raise ValueError("sigma must be finite with one value per denoising step")
    if (sigma < 0).any() or (not allow_zero and (sigma == 0).any()):
        raise ValueError("likelihood needs strictly positive sigma; sampling permits zero")
    return sigma

@torch.no_grad()
def sample_chain(policy, obs, H, A, K, sigma):
    """Collect fixed latent samples, not a reparameterized training graph."""
    if any(type(v) is not int or v < 1 for v in (H, A, K)):
        raise ValueError("H, A and K must be positive integers")
    if obs.ndim != 2 or obs.shape[0] == 0 or not obs.is_floating_point() or not torch.isfinite(obs).all():
        raise ValueError("obs must be finite floating [batch,features]")
    sigma = _sigma_vector(sigma, K, obs, allow_zero=True)
    x = torch.randn(obs.shape[0], H, A, device=obs.device, dtype=obs.dtype)
    xs, dt = [x], 1.0 / K
    for k in range(K):
        tau = torch.full((obs.shape[0],), k * dt, device=obs.device, dtype=obs.dtype)
        velocity = policy(x, tau, obs)
        if velocity.shape != x.shape or not torch.isfinite(velocity).all():
            raise ValueError("policy must return finite velocity matching the action shape")
        x = x + dt * velocity + sigma[k] * torch.randn_like(x)
        xs.append(x)
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite sampled chain")
    return x, torch.stack(xs, 1)

def chain_logprob(policy, obs, chain, sigma):
    """Score detached latent samples; gradients flow only through current means."""
    if chain.ndim != 4 or min(chain.shape) < 1 or chain.shape[1] < 2:
        raise ValueError("chain must be [B,K+1,H,A] with positive dimensions and K>=1")
    if obs.ndim != 2 or obs.shape[0] != chain.shape[0] or not obs.is_floating_point():
        raise ValueError("obs and chain batch dimensions must match")
    if chain.dtype != obs.dtype or chain.device != obs.device:
        raise ValueError("obs and chain must share floating dtype and device")
    if not torch.isfinite(chain).all() or not torch.isfinite(obs).all():
        raise ValueError("finite observations and chain required")
    chain, obs = chain.detach(), obs.detach()
    B, K = chain.shape[0], chain.shape[1] - 1
    sigma = _sigma_vector(sigma, K, chain, allow_zero=False)
    x_k, x_next = chain[:, :-1].flatten(0, 1), chain[:, 1:].flatten(0, 1)
    tau = (torch.arange(K, device=obs.device, dtype=obs.dtype) / K).repeat(B)
    velocity = policy(x_k, tau, obs.repeat_interleave(K, 0))
    if velocity.shape != x_k.shape or not torch.isfinite(velocity).all():
        raise ValueError("policy must return finite velocity matching the action shape")
    mu = x_k + (1.0 / K) * velocity
    sd = sigma.repeat(B).view(-1, 1, 1)
    logp = (-0.5 * ((x_next - mu) / sd) ** 2 - torch.log(sd) - 0.5 * math.log(2 * math.pi)).sum((1, 2))
    return logp.view(B, K), mu.view(B, K, *chain.shape[2:])
```

Before RL, compare the per-step log-probability with an independent Normal distribution and a scalar hand calculation. Freeze old log-probabilities at collection; unchanged weights then produce a ratio of one up to numeric tolerance. Positive sigma is required for the Gaussian density. With sigma=0, sampling is still possible but this log-density and Gaussian KL are undefined. See the CPU lab's optional snippet tests.

### Lab 1b — Per-initial-state groups and the GRPO loss

```python
def group_advantages(success, G, normalize_std=True, leave_one_out=False):
    if type(G) is not int or G < 2 or success.ndim != 1 or success.numel() == 0 or success.numel() % G:
        raise ValueError("need nonempty complete groups with G>=2")
    if not success.is_floating_point() or not torch.isfinite(success).all():
        raise ValueError("rewards must be finite floating values")
    r = success.detach().reshape(-1, G)
    adv = r - r.mean(1, keepdim=True)
    if leave_one_out:
        adv = adv * (G / (G - 1))
    std = r.std(1, keepdim=True, correction=0)  # stated population-std convention
    if normalize_std:
        adv = adv / (std + 1e-6)
    informative = std.squeeze(1) > 0
    return adv.reshape(-1), informative.repeat_interleave(G)

def grpo_loss(policy, obs, chain, old_logp, mu_ref, adv, sigma, clip=0.2, beta=0.01):
    logp, mu = chain_logprob(policy, obs, chain, sigma)
    if old_logp.shape != logp.shape or mu_ref.shape != mu.shape or adv.shape != (logp.shape[0],):
        raise ValueError("old scores, reference means and advantages must match the batch")
    if not 0 < clip < 1 or beta < 0 or not math.isfinite(beta):
        raise ValueError("invalid clip or KL coefficient")
    if any(not torch.isfinite(v).all() for v in (old_logp, mu_ref, adv)):
        raise ValueError("finite rollout statistics required")
    sigma = _sigma_vector(sigma, logp.shape[1], chain, allow_zero=False)
    ratio = torch.exp(logp - old_logp.detach())
    if not torch.isfinite(ratio).all():
        raise ValueError("nonfinite likelihood ratio")
    A = adv.detach()[:, None]
    pg = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean()
    kl = (((mu - mu_ref.detach()) ** 2).sum((2, 3)) / (2 * sigma[None] ** 2)).mean()
    clip_frac = ((ratio - 1).abs() > clip).float().mean()
    return pg + beta * kl, dict(pg=pg.item(), kl=kl.item(), clip_frac=clip_frac.item())
```

The loss averages inner-step terms and uses per-step ratios, so it is a practical clipped surrogate with a stated K-dependent normalization. The averaged conditional Gaussian KL is evaluated on stored old-policy chain states; it is not automatically the marginal action KL or the full current-policy trajectory KL. Record fixed K and positive shared variances when comparing runs. Zero-variance filtering and std normalization change the training objective/distribution.

### Lab 1c — Simulator collection contract (integration exercise)

Implement a collector against a pinned simulator adapter, then test these conditions before training. A repeated seed is a request for reproducibility, not proof of identical initial states: compare full relevant simulator state, observations, and independent policy RNG streams.

```text
PLANNED SIMULATOR INTEGRATION — no implementation is supplied here
reset each group to the same verified initial condition
for each active episode:
    save observation, sampled latent chain, old log-probs, reference means
    execute admitted actions with documented units, bounds and frame
    accumulate rewards and actual executed duration
    stop this episode on termination/truncation; preserve its final observation
    never let a reset observation or later episode enter this episode's reward
mask inactive/padded decisions before group statistics and loss reduction
retain task/initial-state/episode IDs and all sampler settings
```

For discounted control, aggregate a chunk reward as `sum(gamma**j * reward[j])` and bootstrap with `gamma**executed_steps`, applying true termination and truncation rules from the [existing CPU lab](CPU-Lab.md). A chunk may finish early; its actual duration can differ from H. For an undiscounted binary episode-success objective, say so explicitly and keep every decision attached to the correct episode.

Clipping a sampled action before execution is a deterministic environment-side map. The latent-chain likelihood still scores the original sampled chain; do not pretend it is the Gaussian density of the clipped action. For filtered behavioral cloning, record the actual executed controls and masks, including shortened chunks and normalization. Exclude post-terminal padding from both regression and policy-gradient losses.

### Lab 1d — Training and comparisons (integration exercise)

Freeze the intended reference, collect valid groups without gradients, compute old scores once, and update with a bounded number of epochs. Keep KL/ratio and task-quality diagnostics. An all-uninformative batch needs an explicit skip/resample decision; it must not report a successful policy-gradient update.

Compare with filtered behavioral cloning using the same collection budget and a stated update budget. Sample ordinary flow-matching noise and time for the regression target; do not attach a pathwise gradient through the recorded rollout. This comparison still requires a real dataset and complete collector/trainer implementation.

Evaluate the training sampler and the intended deployment sampler on held-out initial conditions. Setting intermediate sigma to zero removes those injected noises, but **fresh initial Gaussian noise remains stochastic**. Use fixed initial noise to test repeatability, and multiple independent noises to estimate success. Do not label zero-intermediate-noise sampling deterministic without stating the initial-noise policy.

### Lab 2 (optional, rung 3) — One RL iteration of π0.5 on LIBERO

For an optional pretrained-model experiment, select a supported pinned RL framework recipe and calculate its actual hardware/precision/adapter requirements first. A single 80-GB GPU is not guaranteed to fit the model, optimizer, reference, rollout storage, and simulator. Run one documented small iteration only after its environment and dataset prerequisites are present; report measured rollout, recomputation, optimizer, synchronization, and memory costs. No such pretrained VLA RL iteration was executed for this course review.

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
| Success, \(\sigma=0\) vs \(\sigma>0\) | Same policy, two samplers | Did the intended sampler retain the learned behavior? Record both initial and intermediate noise settings. |
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
2. You must post-train a ~3 B-parameter flow VLA on one 80 GB GPU. Compare full-parameter PPO with a VLM-sized critic, full-parameter GRPO, and LoRA GRPO by what must sit in memory. What additional measurements determine fit, and under what conditions can LoRA share the reference base?
3. You trained with \(\sigma_k = 0.1\) and success under the noisy sampler rose from 60% to 80%, but zero-intermediate-noise (\(\sigma = 0\), still sampling initial noise) success only went from 62% to 65%. Give a mechanism that explains the gap and two fixes.
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
