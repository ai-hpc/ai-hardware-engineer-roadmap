# Lecture 06: PPO, Trust Regions, and the KL Leash

## Overview

A golfer who hits one great round does not rebuild their whole swing that evening. A good coach lets them adjust a little after each session, sees what happens, and adjusts again. Big changes based on a handful of shots usually make things worse, because those shots were partly luck, and because a changed swing produces *different* shots that the old evidence says nothing about.

Lectures 04-05 had two wasteful habits. Each batch of rollouts was used for **one** gradient step and then thrown away. And nothing stopped that step from being too big: a step that looks small in parameter space can change the policy's behavior a lot. This lecture fixes both:

* **Reuse old tries** with importance sampling: reweight each old action by how much more or less likely the *new* policy is to take it.
* **Limit each step** so the new policy stays close to the one that collected the data. PPO does this by **clipping** the reweighting ratio, for example to [0.8, 1.2]: there is no extra reward for changing any action's probability by more than 20%. TRPO does it with an explicit KL-divergence constraint, and natural gradient does it by measuring step size in behavior space.

Then there is a second, different leash. When you fine-tune a policy that already works, such as a BC policy or π0.5, you also want it to stay close to **where it started**, so RL improves the task without erasing the general skills it arrived with. That is a KL penalty to a frozen **reference** policy. PPO plus a reference KL is the core recipe of RLHF for chat models, and of most PPO-based VLA post-training.

By the end you should be able to:

* write the importance-sampled surrogate and show that its gradient at \(\theta_{\text{old}}\) is the policy gradient
* state the performance-difference lemma, prove it by telescoping, and explain why it justifies small steps
* sketch the PPO clipped objective for \(A > 0\) and \(A < 0\) and say what clipping does and does not guarantee
* distinguish the **step-size** KL (to \(\pi_{\text{old}}\)) from the **reference** KL (to \(\pi_{\text{ref}}\)), and implement both
* build PPO on PickCube with thousands of GPU envs, read its dashboard (clip fraction, approx-KL, explained variance, rollout vs learn time), and run a KL-leash ablation from a BC initialization

---

## 1. Why it matters

| Problem with REINFORCE / A2C | Consequence | PPO's fix |
|---|---|---|
| One gradient step per batch | Regime C: every expensive VLA rollout is used once | Several epochs of minibatch updates on the same batch |
| Step size in parameters ≠ step size in behavior | One bad step collapses the policy, and the next rollouts come from a broken policy | Clip the probability ratio, or constrain KL |
| Fine-tuning drifts from a good starting point | Gains on the training layouts, losses on everything else | KL penalty to a frozen reference |

> **Robot connection.** π0.5 arrives knowing a lot: how to grasp, how to read instructions, how to handle objects it was never fine-tuned on. RL on a handful of LIBERO layouts can buy success on those layouts by quietly forgetting the rest, and the perturbed-layout tests, where objects are moved, are exactly where that shows up. A **KL leash** to the original π0.5 keeps the improved policy close to its original behavior everywhere it is not being rewarded to change. There is a catch that drives Lecture 11: the PPO ratio and the KL both need \(\log \pi_\theta(a \mid o)\), which a flow-matching action expert does not give cheaply.

---

## 2. Mental model: reuse, with a speed limit and a leash

```text
   collect batch with πθ_old ──► compute advantages Â (critic, GAE) ──► E epochs × M minibatches:
                                                                          ratio = πθ(a|s) / πθ_old(a|s)
                                                                          objective = min(ratio·Â, clip(ratio)·Â)
                                                                          stop early if KL(πθ_old‖πθ) too big
                                                                          (optional) − β·KL(πθ‖πref)
```

| | Step-size leash | Reference leash |
|---|---|---|
| Keeps \(\pi_\theta\) close to | \(\pi_{\theta_{\text{old}}}\), the policy that collected *this batch* | \(\pi_{\text{ref}}\), a frozen copy of the *starting* policy |
| Moves? | Yes: reset every iteration | No: fixed for the whole run |
| Purpose | Keep the surrogate objective valid; stability | Prevent forgetting; stop reward hacking |
| Mechanism | Clip, KL constraint, KL early stop | Penalty \(\beta\, \mathrm{KL}\) in loss or reward |
| Extra memory | Old log-probs (one float per sample) | A frozen copy of the policy (or adapters disabled) |

---

## 3. Importance sampling and the surrogate objective

Data was collected by \(\pi_{\text{old}}\), and we want to evaluate a slightly different \(\pi_\theta\). For any function \(f\):

$$
\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\frac{\pi_\theta(a \mid s)}{\pi_{\text{old}}(a \mid s)} f(a)\right]
$$

Apply this per step with \(f = \hat A\) to get the **surrogate objective**:

$$
L_{\text{old}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}}\big[\, r_t(\theta)\, \hat A_t \,\big], \qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

Since \(\nabla_\theta r_t = r_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\) and \(r_t(\theta_{\text{old}}) = 1\), **the gradient of \(L\) at \(\theta_{\text{old}}\) is exactly Lecture 05's policy gradient.** One step on \(L\) is A2C. Several steps on \(L\) is where PPO starts.

Two warnings are already visible:

* **The ratio is per step, not per trajectory.** A full trajectory importance weight is a product of \(T\) ratios, and its variance grows exponentially with \(T\). The per-step version stays usable, but it ignores the fact that \(\pi_\theta\) would also visit **different states**. The expectation is still over \(s \sim \pi_{\text{old}}\).
* **Ratios far from 1 mean noisy estimates.** If \(r_t = 5\), one sample carries five times its normal weight. Unbounded, the optimizer will happily push a few ratios to extreme values on a minibatch it has memorized.

Section 4 shows when ignoring the state-distribution change is safe.

---

## 4. Why small steps work: the performance-difference lemma

**Lemma (Kakade & Langford, 2002).** For any two policies:

$$
J(\pi') - J(\pi) = \mathbb{E}_{\tau \sim \pi'}\!\left[\sum_{t} \gamma^t A^{\pi}(s_t, a_t)\right]
$$

**Proof by telescoping.** Write \(A^\pi(s_t, a_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)]\) and sum along a trajectory from \(\pi'\):

$$
\mathbb{E}_{\tau \sim \pi'}\!\left[\sum_t \gamma^t \big(r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)\big)\right] = \mathbb{E}_{\tau \sim \pi'}\!\left[\sum_t \gamma^t r_t\right] - \mathbb{E}_{s_0}\big[V^\pi(s_0)\big] = J(\pi') - J(\pi)
$$

The \(V^\pi\) terms cancel pairwise, leaving only \(-V^\pi(s_0)\), and the initial-state distribution does not depend on the policy. \(\square\)

Read it in plain language: the new policy is better by exactly the old policy's advantages, **accumulated over the states the new policy visits**. The surrogate in Section 3 evaluates those advantages on states the **old** policy visits. The two agree when the state distributions agree. How much can they differ? TRPO's main theorem (Schulman et al., 2015) bounds it:

$$
J(\pi') \;\ge\; J(\pi) + L_\pi(\pi') - \frac{4 \epsilon \gamma}{(1-\gamma)^2}\, \max_s \mathrm{KL}\big(\pi(\cdot \mid s)\,\Vert\,\pi'(\cdot \mid s)\big), \qquad \epsilon = \max_{s,a} \lvert A^\pi(s,a) \rvert
$$

where \(L_\pi(\pi')\) is the surrogate advantage term. Three readings of this bound:

* **Small policy change means small change in visited states**, so the surrogate is trustworthy inside a KL ball.
* The penalty scales as \(1/(1-\gamma)^2\), the **square of the effective horizon**. This is the same compounding as behavioral cloning's \(O(\epsilon T^2)\) in Lecture 02: a per-step difference grows into a state-distribution difference over the horizon. Long horizons, or short ones made long by tiny control steps, demand smaller steps. Action chunks help here too (Lecture 13).
* Maximizing the right-hand side gives **guaranteed monotonic improvement** (a minorize-maximize scheme). But the constant is so pessimistic that the resulting steps are tiny, so every practical method replaces the penalty with something looser.

---

## 5. The trust-region family

| Method | Step rule | Cost per update | Notes |
|---|---|---|---|
| **TRPO** | \(\max_\theta L(\theta)\) s.t. \(\mathbb{E}_s[\mathrm{KL}(\pi_{\text{old}} \Vert \pi_\theta)] \le \delta\) | Conjugate gradient (~10 Fisher-vector products) + line search | Principled; awkward with shared actor-critic nets and minibatches |
| **Natural gradient** | \(\Delta\theta \propto F^{-1} g\) | Same machinery as TRPO without the line search | Parameterization-invariant |
| **Adaptive KL penalty** | \(\max L - \beta\, \mathrm{KL}\); double \(\beta\) if KL > 1.5·target, halve if < target/1.5 | First-order | A PPO variant in the original paper |
| **PPO-clip** | \(\max L^{\text{CLIP}}\) (Section 6) | First-order, Adam, minibatches | The default almost everywhere |

**Natural gradient in three lines.** To second order, \(\mathrm{KL}(\pi_\theta \Vert \pi_{\theta + \Delta}) \approx \tfrac12 \Delta^\top F \Delta\), where \(F = \mathbb{E}[\nabla \log \pi\, \nabla \log \pi^\top]\) is the Fisher information matrix. Maximizing the linearized objective \(g^\top \Delta\) subject to \(\tfrac12 \Delta^\top F \Delta \le \delta\) gives:

$$
\Delta^\star = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1} g
$$

The step is measured in **behavior** (KL), not in parameters, so rescaling or reparameterizing the network does not change it. \(F\) is never formed. Conjugate gradient needs only Fisher-vector products, each costing about one extra backward pass. That per-update cost is roughly 10× an Adam step and does not fit minibatch SGD well. In practice this is what made TRPO lose to PPO at scale.

---

## 6. PPO: the clipped surrogate

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\, \hat A_t,\; \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\, \hat A_t\big)\Big]
$$

The shape per sample, as a function of the ratio \(r\):

```text
   A > 0  (good action: raise its probability)        A < 0  (bad action: lower its probability)

   objective                                           objective
      │          ┌──────────── flat: no gain              │  ─────────────┐ flat: no gain
      │         /               past 1+ε                  │   past 1−ε     \
      │        /                                          │                 \  still steep: if r grew,
      │       /  slope A                                  │                  \ the min keeps pushing back
      └──────┼──────┼──────► r                            └──────┼──────┼──────► r
           1−ε   1+ε                                           1−ε   1+ε
```

* **A > 0:** the gain stops at \(r = 1+\varepsilon\). There is no incentive to make a good action more than 20% more likely in one batch. Below \(1-\varepsilon\) the gradient is still active, so if the action became *less* likely, the objective pushes it back.
* **A < 0:** the gain stops at \(r = 1-\varepsilon\). If the ratio overshoots upward, so the bad action became *more* likely, the unclipped term wins the `min` and the gradient pushes it back down.
* The `min` makes \(L^{\text{CLIP}}\) a **pessimistic lower bound** on the unclipped surrogate. It ignores changes that would improve the objective beyond the clip range and keeps those that make it worse.

The full loss, written as a quantity to minimize:

$$
\mathcal{L}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_v\, \mathbb{E}_t\big[(V_\phi(s_t) - \hat R_t)^2\big] - c_e\, \mathbb{E}_t\big[\mathcal{H}(\pi_\theta(\cdot \mid s_t))\big]
$$

with GAE targets \(\hat R_t\) from Lecture 05 and an optional entropy bonus \(c_e\) that keeps \(\sigma\) from collapsing before the task is found.

**What clipping does not guarantee.** Clipping removes the *incentive* to move a sample's ratio past \(1 \pm \varepsilon\). It does not *prevent* it. Gradients from other samples, multiple epochs, and Adam momentum can all carry ratios outside the range. Engstrom et al. (2020) found that much of PPO's advantage over TRPO in common implementations came from code-level choices rather than from the clipping itself. So **monitor the step directly**:

| Diagnostic | Formula (samples from \(\pi_{\text{old}}\), \(r = \pi_\theta/\pi_{\text{old}}\)) | Healthy range (rule of thumb) |
|---|---|---|
| **Approx-KL** \(\mathrm{KL}(\pi_{\text{old}} \Vert \pi_\theta)\) | \(\mathbb{E}[(r - 1) - \log r]\): unbiased, always ≥ 0 (Schulman's "k3") | ~0.01-0.05; spikes mean the step was too big |
| **Clip fraction** | share of samples with \(\lvert r - 1 \rvert > \varepsilon\) | ~0.1-0.3; near 0 means epochs are wasted; near 0.5 means too aggressive |
| **Early stop** | break the epoch loop when approx-KL > `target_kl` | Makes "number of epochs" an upper bound |

---

## 7. The reference leash: KL to the starting policy

Fine-tuning objective with a frozen reference \(\pi_{\text{ref}}\) (the BC policy, or the base VLA):

$$
\max_\theta\; \mathbb{E}_{\pi_\theta}\Big[\sum_t \gamma^t r_t\Big] - \beta\, \mathbb{E}_{s \sim \pi_\theta}\Big[\mathrm{KL}\big(\pi_\theta(\cdot \mid s)\,\Vert\,\pi_{\text{ref}}(\cdot \mid s)\big)\Big]
$$

Two common implementations:

* **In the loss** (analytic KL). For diagonal Gaussians, per action dimension:

$$
\mathrm{KL}\big(\mathcal{N}(\mu, \sigma^2)\,\Vert\,\mathcal{N}(\mu_{\text{ref}}, \sigma_{\text{ref}}^2)\big) = \log\frac{\sigma_{\text{ref}}}{\sigma} + \frac{\sigma^2 + (\mu - \mu_{\text{ref}})^2}{2\sigma_{\text{ref}}^2} - \frac12
$$

* **In the reward** (sampled KL), as in InstructGPT-style RLHF: \(r_t \leftarrow r_t - \beta \big(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{ref}}(a_t \mid s_t)\big)\). The critic then learns a KL-aware value, and drift is penalized as a cost the agent can plan around.

\(\beta\) is a real trade-off, not a free safety knob. Too small and the policy drifts and may exploit simulator bugs or reward-model errors (Lecture 08 shows the reward-model case). Too large and it cannot move far enough from the reference to fix the failures you are training on. The direction \(\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})\) is the standard choice. It penalizes \(\pi_\theta\) for putting probability where the reference would not, which is what "don't do things the original policy never would" means.

> **Robot connection.** For π0.5 the reference is the released checkpoint and \(\beta\) controls how much of its general behavior is protected. Measure the leash's effect where forgetting would show: the **perturbed-layout** suites and the LIBERO suites you are *not* training on. A flow-matching expert has no closed-form \(\pi_\theta(a \mid o)\), so neither form of KL above is directly available. Lecture 11 covers the approximations: per-denoising-step Gaussian KLs after noise injection, or regularizing in noise space.

---

## 8. The implementation details that decide whether PPO works

The ICLR blog post "The 37 Implementation Details of PPO" (Huang et al.) and the large-scale study by Andrychowicz et al. (2020) are the references for this section. The ones that matter most for continuous-control robot tasks:

| Detail | What | ManiSkill `ppo.py` default (at time of writing) |
|---|---|---|
| Advantage normalization | Per minibatch, zero mean and unit std | on |
| Epochs × minibatches | Passes over the batch × splits per pass | 4 × 32 |
| Clip \(\varepsilon\) | Ratio clip range | 0.2 |
| Approx-KL early stop | `target_kl` | 0.1 |
| Global grad-norm clip | `max_grad_norm` | 0.5 |
| Value loss coefficient / clipping | `vf_coef`; clipped value loss | 0.5 / off |
| Entropy coefficient | `ent_coef` | 0.0 |
| Discount / GAE | \(\gamma\), \(\lambda\) | 0.8 / 0.9 (dense reward, 50-step tasks) |
| Rollout shape | `num_envs` × `num_steps` | 512 × 50 |
| Optimizer | Adam, lr, eps | 3e-4, eps 1e-5 |
| Policy std | State-independent learned log-std | init −0.5 |
| Init | Orthogonal, small gain on the policy output layer | yes |

Others to know about: observation normalization with running mean/std (essential when state features have very different scales), reward scaling, learning-rate annealing, and tanh-squashing vs clipping actions. **Value clipping** (clipping \(V\) updates like the ratio) appeared in early PPO implementations. Large-scale studies did not find it reliably helpful, and many codebases default it off. Treat these as hypotheses to test with seeds, not as gospel.

### 8.1 PPO at GPU-simulator scale

GPU simulation changes the shape of the batch. Instead of a few envs running long rollouts, legged-robot work in Isaac Gym / Isaac Lab (Rudin et al., 2021; the `rsl_rl` library) runs **thousands of envs for a few dozen steps each**. Batches have \(10^4\)-\(10^5\) transitions, rollouts are short and truncated, and the critic's bootstrapped value handles the cut (Lecture 05's truncation masks become critical). Large batches cut gradient variance (Lecture 04), and the short rollout keeps the data fresh. The result is policies trained in minutes of wall-clock time instead of days.

---

## 9. The hardware view

### 9.1 Rollout buffer memory

PPO stores, per transition, \(\text{obs} + \text{act} + \{\log\pi_{\text{old}}, V, r, d, \hat A, \hat R\}\) (plus \(\mu_{\text{ref}}\) or \(\log \pi_{\text{ref}}\) with a leash). For \(N\) envs and \(K\) steps:

| Observation | Per transition | \(N = 4096,\ K = 50\) (204,800 transitions) |
|---|---|---|
| State (a few dozen fp32 floats, 8-D action, 6 scalars) | ~0.2-0.3 KB | ~50 MB: irrelevant |
| One 128×128 RGB frame, `uint8` | ~48 KB | ~9.4 GiB |
| Same frame stored as `float32` | ~192 KB | ~37.5 GiB: does not fit next to the model |

So pixel PPO runs with hundreds, not thousands, of envs on a 24-48 GB GPU, stores frames as `uint8`, and converts minibatches to float on the fly. For VLAs the buffer usually stores observations on the **CPU** and moves each minibatch to the GPU.

### 9.2 Learner time vs sample reuse

One PPO iteration costs \(K\cdot(t_{\text{env}} + t_{\text{policy}})\) of rollout plus \(E \cdot M\) optimizer steps of learning. \(E\) is the **sample-reuse factor**:

| Regime | What dominates | How to set \(E\) |
|---|---|---|
| B (GPU sim, MLP) | Often the learner, once \(E \cdot M\) is large | Small \(E\) (a few epochs); spend compute on more envs. Fresh data is cheaper than reuse. |
| C (VLA) | Rollout: VLA forward per chunk | Larger \(E\) amortizes expensive rollouts, until the clip fraction and approx-KL show the batch is used up |

Log `rollout_time` and `update_time` separately every iteration (ManiSkill's script already does). Their ratio tells you which knob to turn.

### 9.3 The memory bill for PPO on a big policy

| Component | Full fine-tune of a \(P\)-param policy | LoRA fine-tune |
|---|---|---|
| Trainable policy + Adam (mixed precision) | ~16 B × \(P\) | ~2 B × \(P\) frozen base + small adapters with their optimizer state |
| Frozen reference for the KL leash | ~2 B × \(P\) (bf16 copy, no optimizer) | **0 extra**: reference = same weights with adapters disabled (PEFT's `disable_adapter()` context) |
| Critic | Lecture 05, Section 7 | Same |
| Old log-probs / ref log-probs | 1 float per sample each | Same |
| Activations | Grows with minibatch × sequence length | Same (LoRA does not shrink activations) |

The LoRA trick removes the reference copy's memory but **not its compute**. Reference log-probs still need a forward pass with adapters off, usually done once per batch before the update epochs. For a 3B policy, the frozen reference alone is ~6 GB in bf16, and the full-fine-tune optimizer state is ~48 GB. This arithmetic is why the capstone offers a LoRA single-GPU path and a full-parameter multi-GPU path.

---

## 10. Build it

Code lives in `ppo/` and reuses `make_env`, `mlp`, `gae`, and `explained_variance` from Lecture 05's `actor_critic/a2c_gae.py`.

### Lab 6a — PPO from scratch on PickCube (1024-4096 envs)

The rollout loop is Lab 5a's, plus storing `logp_old` (and `mu_ref` for Lab 6b). The new part is the update:

```python
# ppo/ppo.py  (update step; rollout as in actor_critic/a2c_gae.py, also storing logp_old and mu_ref)
import torch, torch.nn as nn
from torch.distributions import Normal

def gaussian_kl(mu, std, mu_ref, std_ref):          # KL(N(mu,std) || N(mu_ref,std_ref)), per dimension
    return torch.log(std_ref / std) + (std ** 2 + (mu - mu_ref) ** 2) / (2 * std_ref ** 2) - 0.5

def ppo_update(actor, critic, log_std, opt, B, a, actor_on=True, ref_std=None):
    """B: flat batch with obs, act, logp_old, adv, ret, and optionally mu_ref. Returns diagnostics."""
    n = B["obs"].shape[0]; mb = n // a.minibatches
    params = [p for g in opt.param_groups for p in g["params"]]
    log = {"clipfrac": [], "approx_kl": [], "kl_ref": [], "updates": 0}
    for epoch in range(a.epochs):
        perm = torch.randperm(n, device=B["obs"].device)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            mu, std = actor(B["obs"][idx]), log_std.exp()
            dist = Normal(mu, std)
            logratio = dist.log_prob(B["act"][idx]).sum(-1) - B["logp_old"][idx]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()             # k3 estimator of KL(old || new)
                log["approx_kl"].append(approx_kl)
                log["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
            if approx_kl > a.target_kl:                                         # step leash: stop this iteration
                return log
            adv = B["adv"][idx]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = -torch.min(ratio * adv, ratio.clamp(1 - a.clip, 1 + a.clip) * adv).mean()
            v_loss = 0.5 * (critic(B["obs"][idx]).squeeze(-1) - B["ret"][idx]).pow(2).mean()
            loss = a.vf_coef * v_loss
            if actor_on:                                                        # off during critic warm-up
                loss = loss + pg_loss - a.ent_coef * dist.entropy().sum(-1).mean()
                if a.kl_ref > 0:                                                # reference leash
                    kl = gaussian_kl(mu, std, B["mu_ref"][idx], ref_std).sum(-1).mean()
                    loss = loss + a.kl_ref * kl
                    log["kl_ref"].append(kl.item())
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, a.max_grad_norm); opt.step()
            log["updates"] += 1
    return log
```

Run it with `num_envs ∈ {1024, 2048, 4096}`, `num_steps = 50`, the dense reward, and 3 seeds. Every iteration, log success, clip fraction, approx-KL, explained variance, `kl_ref`, the number of updates actually performed (early stopping makes it vary), and **SPS split into rollout and learn**. Diff your script against ManiSkill's `examples/baselines/ppo/ppo.py`. Every difference is either a bug or a hypothesis.

### Lab 6b — Start from BC, fine-tune with and without the leash

Load your Lecture 02 BC policy as both the **initial actor** and the **frozen reference**. Use the same observation mode and `control_mode` the BC policy was trained with. Pass it to `make_env`. If your BC network is not `mlp(obs_dim, act_dim)`, wrap it so `actor(obs)` returns the action mean.

```python
# ppo/finetune_from_bc.py  (setup and evaluation; the training loop is Lab 6a's)
import copy, torch, gymnasium as gym
bc = torch.load("bc_dagger/bc_policy.pt", weights_only=False)  # whole nn.Module saved in Lecture 02
actor = copy.deepcopy(bc).to(dev)                      # initial policy = BC
ref = copy.deepcopy(bc).to(dev).eval().requires_grad_(False)   # frozen reference
log_std = torch.nn.Parameter(torch.full((act_dim,), -1.0, device=dev))  # start near-deterministic
ref_std = torch.full((act_dim,), 0.37, device=dev)    # fixed reference spread ≈ exp(-1.0)
# during rollout: B["mu_ref"] = ref(obs) under no_grad (cheap for an MLP; a full forward for a VLA)
# first a.critic_warmup iterations: ppo_update(..., actor_on=False) so a random critic can't wreck BC

def tree_index(x, idx):
    return {k: tree_index(v, idx) for k, v in x.items()} if isinstance(x, dict) else x[idx]

@torch.no_grad()
def evaluate(policy, n=512, offset=0.0, seed=12345, control_mode="pd_ee_delta_pose"):  # must match the L02 BC policy
    """Mean-action success rate. offset > 0 moves the cube `offset` metres outside the training spawn square."""
    env = gym.make("PickCube-v1", num_envs=n, obs_mode="state", control_mode=control_mode, sim_backend="physx_cuda")
    obs, _ = env.reset(seed=seed); base = env.unwrapped
    if offset > 0:
        st = base.get_state_dict()                        # actors -> name -> (n, 13) [pos, quat, lin vel, ang vel]
        c = torch.as_tensor(base.cube_spawn_center, device=base.device, dtype=torch.float32)
        d = st["actors"]["cube"][:, :2] - c
        d = d / d.abs().max(-1, keepdim=True).values.clamp_min(1e-6)       # unit square direction
        st["actors"]["cube"][:, :2] = c + d * (base.cube_spawn_half_size + offset)
        obs, _ = env.reset(options={"reset_to_env_states": {"env_states": st}})
    success = torch.zeros(n, dtype=torch.bool, device=base.device)
    for _ in range(50):
        obs, _, _, _, info = env.step(policy(obs).clamp(-1, 1))
        success |= info["success"].bool()
    env.close()
    return success.float().mean().item()
```

`cube_spawn_center` and `cube_spawn_half_size` are attributes of `PickCube-v1` in current ManiSkill3. Check `pick_cube.py` in your installed version. If you wrote a perturbation function in Lecture 02, use that one instead so the numbers are comparable. Then run the ablation, 3 seeds each, with an equal env-step budget:

| Run | Init | `kl_ref` (\(\beta\)) |
|---|---|---|
| PPO scratch | random | 0 |
| BC only | BC | — (no RL) |
| BC → PPO | BC | 0 |
| BC → PPO + leash | BC | 0.01, 0.1, 1.0 |

Evaluate every run at `offset ∈ {0, 0.02, 0.05}` with a **fixed seed list** shared across runs. Do not predict the answer in your notes. Report it. In this small setting the BC policy has no broad pretraining to protect, so the leash may help, hurt, or do nothing on perturbed layouts. Whichever you find, explain it with the KL-to-reference curve and the per-offset success.

---

## 11. Use it in the real stack

* **CleanRL** `ppo_continuous_action.py` and **ManiSkill** `examples/baselines/ppo/ppo.py` (and `ppo_rgb.py` for pixels) are the single-file references. Your Lab 6a should differ from them only where you can say why.
* **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** is the PPO used by Isaac Lab's locomotion and manipulation tasks: thousands of envs, short rollouts, adaptive learning rate driven by KL, and separate critic observations.
* **Stable-Baselines3** PPO is the batteries-included option for CPU environments.
* **LLM and VLA post-training** run the same objective at larger scale: PPO with a reference KL in RLHF (InstructGPT), and PPO/GRPO for VLAs in frameworks such as [RLinf](https://github.com/RLinf/RLinf). Lecture 11 covers what changes when the policy is a flow-matching VLA.

---

## 12. Measure it

The PPO dashboard. Plot all of these against env steps for every run:

| Metric | What healthy looks like | What a problem looks like |
|---|---|---|
| **Success rate** (eval, fixed seeds) | Rising, seeds agreeing | Seeds diverge wildly: step size or reward issue |
| **Approx-KL** \(\mathrm{KL}(\pi_{\text{old}} \Vert \pi_\theta)\) | Small, stable | Spikes then collapse: lower lr, lower epochs, set `target_kl` |
| **Clip fraction** | ~0.1-0.3 | ≈0: wasted epochs; ≫0.3: too aggressive |
| **Updates per iteration** | Near \(E \cdot M\) | Always 1-2: early stop firing, so the batch is stale after one step |
| **Explained variance** | Rising toward 1 | ≤ 0: critic or masks broken (Lecture 05) |
| **KL to reference** | Bounded, set by \(\beta\) | Growing without bound: leash too weak |
| **Policy std** | Shrinking slowly | Collapsed early: raise entropy coefficient or init std |
| **SPS: rollout vs learn** | Ratio known per `num_envs` | — (this is the hardware result) |
| **Peak GPU memory** | vs `num_envs` and obs mode | Pixel buffer arithmetic (Section 9.1) confirmed |

Then report **samples-to-success and wall-clock-to-success** (to 80% success) for PPO vs your Lecture 05 A2C on the same task. This is the number Lecture 07 will compare against SAC.

---

## 13. Ship it

Commit `ppo/` containing:

* `ppo.py`, `finetune_from_bc.py`, `evaluate.py` (fixed seed list, offsets), and `run_all.sh`
* `results/dashboard_<run>.png`: success, approx-KL, clip fraction, explained variance, KL-to-ref, std, one figure per configuration, ≥ 3 seeds
* `results/sps_split.csv`: `num_envs`, rollout s/iter, learn s/iter, peak memory, for 1024/2048/4096 envs (state) and at least one pixel setting
* `results/leash_ablation.md`: the table from Lab 6b with success at each offset (mean ± range over seeds) and final KL to reference
* `NOTES.md`: one paragraph on what the leash did and why, and one paragraph estimating, with Section 9.3's arithmetic, the GPU memory for PPO + reference KL on a 3B-parameter policy with full fine-tuning vs LoRA

---

## Exit criteria

You can move on when you can:

* show that \(\nabla_\theta L_{\text{old}}(\theta)\big|_{\theta_{\text{old}}}\) is the policy gradient, and say what the surrogate ignores
* prove the performance-difference lemma by telescoping and connect the TRPO bound's \(1/(1-\gamma)^2\) to compounding error
* draw \(L^{\text{CLIP}}\) for \(A > 0\) and \(A < 0\), and explain why approx-KL still needs monitoring
* explain the difference between step-size KL and reference KL, including what each costs in memory
* read a PPO dashboard and diagnose at least three failure patterns from it
* show your measured rollout/learn split and say whether more epochs or more envs would help on your hardware

---

## Self-check

1. Your PPO run on PickCube shows approx-KL ≈ 0.001 and clip fraction ≈ 0.01 for the whole run, and learning is slow. What does that say about how the batch is being used, and which two knobs would you change first?
2. After switching from 1024 to 4096 envs at the same `num_minibatches`, your learn time per iteration quadrupled and SPS barely improved. Using Section 9.2, explain why, and propose a configuration that uses the extra envs better.
3. A teammate argues that since PPO clips the ratio, the KL early stop is redundant. Give two concrete mechanisms by which ratios escape the clip range during an update.
4. You fine-tune π0.5 with PPO on 10 LIBERO-Goal layouts. Training success goes from 70% to 95%, but LIBERO-Spatial success, which you did not train on, drops from 90% to 75%. What leash would you add, how would you choose \(\beta\), and what would you measure to show it worked?
5. You have one 80 GB GPU and a 3B-parameter policy. Using Section 9.3, decide whether full-parameter PPO with a reference KL and a separate same-size critic fits. If it doesn't, propose two changes that make it fit and say what each one costs.
6. Why does the TRPO bound's penalty grow as \(1/(1-\gamma)^2\)? Connect it to Lecture 02's BC compounding bound, and explain why executing 10-step action chunks might let you take larger policy steps safely.

---

## References

* CS 285 Lectures 9 "Advanced Policy Gradients" and 10 — [slides and video](https://rail.eecs.berkeley.edu/deeprlcourse/)
* Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms," 2017 — [arXiv](https://arxiv.org/abs/1707.06347)
* Schulman, Levine, Moritz, Jordan, Abbeel, "Trust Region Policy Optimization," 2015 — [arXiv](https://arxiv.org/abs/1502.05477)
* Kakade & Langford, "Approximately Optimal Approximate Reinforcement Learning," ICML 2002 (performance-difference lemma)
* Kakade, "A Natural Policy Gradient," NeurIPS 2001
* Huang et al., "The 37 Implementation Details of Proximal Policy Optimization" — [ICLR blog track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
* Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO," 2020 — [arXiv](https://arxiv.org/abs/2005.12729)
* Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study," 2020 — [arXiv](https://arxiv.org/abs/2006.05990)
* Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," 2021 — [arXiv](https://arxiv.org/abs/2109.11978); [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
* Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022 — [arXiv](https://arxiv.org/abs/2203.02155) (PPO with a per-token KL to the reference)
* Schulman, "Approximating KL Divergence" — [blog](http://joschu.net/blog/kl-approx.html) (the k1/k3 estimators used in every PPO dashboard)
* Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021 — [arXiv](https://arxiv.org/abs/2106.09685)

---

## Next in this special course

* Next: [Lecture 07 — Value-Based and Off-Policy RL: From Q-Learning to SAC](Lecture-07.md)
* Previous: [Lecture 05 — Actor-Critic, GAE, and Privileged Critics](Lecture-05.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
