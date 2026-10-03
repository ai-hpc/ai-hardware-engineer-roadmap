# Lecture 10: Offline RL and Offline-to-Online Fine-Tuning

## Overview

**Offline RL** means learning a policy from a fixed dataset, with no new tries while learning. Think of learning to cook only from someone else's notebook: you can read what they did and how each dish turned out, but you can't taste anything yourself. That is still useful. The notebook has good dishes and bad ones, and a careful reader can do better than the average entry — even better than the best entry, by combining the first half of one recipe with the second half of another.

There is a trap, though. Suppose you ask "what if I put in ten spoons of salt?" The notebook never tried it, so it doesn't say it's bad. A learned value function has to fill that gap with a guess, and an optimizer that picks "the best-looking option" will tend to pick exactly the options whose guesses are too high. Online, the robot would try ten spoons, taste it, and correct the guess. Offline, nothing ever corrects it.

Every offline RL method is a way of getting improvement out of the data without trusting guesses about actions the data never contains. Then, once online interaction is available, the question becomes how to **keep** what was learned offline and improve from there without collapsing. This lecture covers both, following CS 185/285 topics 17-18, and ends with the methods that matter for flow-matching VLAs: IDQL, FQL, and noise-space steering (DSRL).

The hardware angle is simple and important: offline RL **deletes the rollout term** from the cost equation in Lecture 01. On rung 3, where a VLA forward pass dominates everything, that changes what is affordable.

By the end you should be able to:

* explain why Q-learning on a fixed dataset overestimates values of out-of-distribution (OOD) actions, and why online RL doesn't suffer the same way
* derive advantage-weighted regression from a KL-constrained policy improvement problem
* write the CQL regularizer and the IQL expectile loss, and say what each one prevents
* explain stitching, and which methods can and cannot do it
* run an offline → online pipeline (RLPD-style) and measure the "fine-tuning dip"
* describe three ways to do critic-driven improvement on a flow-matching policy, and how to bake the result back into the policy weights

---

## 1. Why it matters: reuse everything you already paid for

By this point in the course you have a pile of robot data: scripted-expert demos from Lecture 02, rollouts from flow policies in Lecture 03, thousands of PPO and SAC episodes from Lectures 06-07 — successes and failures. Throwing that away every time you change the algorithm is wasteful in regime B and ruinous in regime C, where each episode costs dozens of VLA forward passes.

| | Behavioral cloning | Online RL | Offline RL |
|---|---|---|---|
| Data | Expert demos only | Fresh rollouts from current policy | Any fixed dataset with rewards |
| Uses failures? | No (or harmfully) | Yes | Yes |
| Can beat the data? | No | Yes | Yes, within the data's support |
| Needs environment during training? | No | Yes | No |
| Main failure | Compounding error | Sample cost, no signal | OOD overestimation |

**How can a fixed dataset beat itself?** Three ways:

1. **Filter.** Imitate the good parts, ignore the bad.
2. **Generalize.** A value function trained across many episodes can rank actions better than any single episode shows.
3. **Stitch.** If the data contains A→B in one episode and B→C in another, but never A→C, dynamic programming propagates value from C back through B to A, and the policy learns A→C. Imitation — even filtered imitation — cannot do this, because no single successful episode shows the full path.

> **Robot connection.** For the kitchen-table arm: one episode grasps the bowl cleanly but drops it on the way; another starts with a sloppy grasp but places the bowl perfectly once it has it. Neither is a success worth cloning end-to-end. A value-based offline method can learn "grasp like episode 1, place like episode 2." That is stitching, and it is why offline RL is worth the extra machinery over filtered BC.

---

## 2. Mental model: why naive off-policy RL fails offline

### 2.1 The setting

We have a dataset \(\mathcal{D} = \{(s, a, r, s')\}\) collected by some behavior policy \(\pi_\beta\) (often a mixture of several policies, and usually unknown). We want a policy \(\pi\) that maximizes return, without ever running \(\pi\) during training.

### 2.2 Where the errors come from

An off-policy actor-critic (SAC, TD3 from [Lecture 07](Lecture-07.md)) trains \(Q\) with the Bellman backup

$$
Q(s,a) \leftarrow r + \gamma \, \mathbb{E}_{a' \sim \pi(\cdot \mid s')}\big[ Q(s', a') \big], \qquad \pi \leftarrow \arg\max_\pi \mathbb{E}_{a \sim \pi}\big[ Q(s,a) \big]
$$

The left side only ever uses \((s, a)\) pairs from the dataset. The right side queries \(Q(s', a')\) at actions \(a'\) chosen by \(\pi\), and \(\pi\) is trained to maximize \(Q\). If \(a'\) is far from anything in \(\mathcal{D}\), \(Q(s', a')\) is pure extrapolation. The maximization selects actions whose extrapolation errs high — the same argmax-over-noise bias as in Lecture 07 and the model exploitation of [Lecture 09](Lecture-09.md) — and the backup copies those inflated values into \(Q(s,a)\) for in-data actions too. The error propagates and compounds across backups.

Online, this is self-correcting: the policy tries the overvalued action, gets a low return, and the next backup fixes it. Offline, no transition containing \(a'\) will ever arrive. The classic symptom is that **Q-values grow far beyond the maximum possible return** while actual policy performance collapses. With a sparse 0/1 reward and \(\gamma = 0.99\), no true Q can exceed 1. If your critic reports 30, it is extrapolating.

### 2.3 Three families of fixes

| Fix | Idea | Representative methods |
|---|---|---|
| **Stay close to the data** | Constrain \(\pi\) to \(\pi_\beta\) so \(a'\) stays in-distribution | BCQ, TD3+BC, AWR / AWAC |
| **Be pessimistic about unknown actions** | Push Q down on actions outside the data | CQL, Cal-QL, critic ensembles |
| **Never query unknown actions** | Learn values only from dataset actions; extract the policy by weighted imitation | IQL (and IDQL's critic) |

---

## 3. Stay close to the data: AWR, AWAC, TD3+BC

### 3.1 Advantage-weighted regression from a KL constraint

Ask for the best policy that stays near the behavior policy:

$$
\max_\pi \; \mathbb{E}_{s \sim \mathcal{D}} \, \mathbb{E}_{a \sim \pi(\cdot \mid s)} \big[ A(s,a) \big] \quad \text{s.t.} \quad \mathbb{E}_{s \sim \mathcal{D}} \Big[ D_{\mathrm{KL}}\big(\pi(\cdot \mid s) \,\|\, \pi_\beta(\cdot \mid s)\big) \Big] \le \epsilon
$$

Form the Lagrangian per state with multiplier \(\beta\) for the KL and \(\lambda\) for normalization:

$$
\sum_a \pi(a \mid s) A(s,a) \;-\; \beta \sum_a \pi(a \mid s) \log \frac{\pi(a \mid s)}{\pi_\beta(a \mid s)} \;+\; \lambda \Big( \sum_a \pi(a \mid s) - 1 \Big)
$$

Setting the derivative with respect to \(\pi(a \mid s)\) to zero gives \(A - \beta(\log \pi/\pi_\beta + 1) + \lambda = 0\), so

$$
\pi^*(a \mid s) = \frac{1}{Z(s)} \, \pi_\beta(a \mid s) \exp\!\big( A(s,a) / \beta \big)
$$

The optimal constrained policy is the behavior policy, reweighted by exponentiated advantage. To fit a network to it, minimize \(D_{\mathrm{KL}}(\pi^* \,\|\, \pi_\theta)\). Expanding, and using the fact that dataset actions are samples from \(\pi_\beta\):

$$
\theta^* = \arg\max_\theta \; \mathbb{E}_{(s,a) \sim \mathcal{D}} \Big[ \exp\!\big( A(s,a)/\beta \big) \, \log \pi_\theta(a \mid s) \Big]
$$

(dropping \(Z(s)\), as practical implementations do). This is **weighted behavioral cloning**: copy every dataset action, but copy good ones much more strongly. \(\beta \to \infty\) recovers plain BC; small \(\beta\) approaches "copy only the best action." The policy never sees an action outside the data, so there is no OOD query in the *policy* update.

* **AWR** (Peng et al., 2019) estimates \(A\) with Monte-Carlo returns minus a learned \(V\).
* **AWAC** (Nair et al., 2020) estimates \(A\) with an off-policy \(Q\) critic, which makes it usable for offline pre-training followed by online fine-tuning.
* **Filtered BC** is the hard-threshold limit: weight 1 for successful episodes, 0 otherwise. Always run it as a baseline — on sparse-success robot data it is often surprisingly hard to beat.

In practice, clip the weights (for example at 100) or normalize them per batch; a few huge weights otherwise dominate the gradient.

### 3.2 TD3+BC: the one-line baseline

Fujimoto & Gu (2021) add a BC term to the TD3 actor loss:

$$
\pi = \arg\max_\pi \; \mathbb{E}_{(s,a) \sim \mathcal{D}} \Big[ \lambda \, Q\big(s, \pi(s)\big) - \big( \pi(s) - a \big)^2 \Big], \qquad \lambda = \frac{\alpha}{\frac{1}{B}\sum_{i} \lvert Q(s_i, a_i) \rvert}
$$

The normalization by the mean \(\lvert Q \rvert\) keeps the trade-off scale-free. The paper's message is worth absorbing: a minimal change to a standard algorithm is a strong baseline, and many more complex methods do not beat it by much. Start there before reaching for anything heavier.

---

## 4. Pessimism: Conservative Q-Learning

CQL (Kumar et al., 2020) adds a regularizer to the Bellman loss that pushes Q **down** on actions the current policy (or a broad proposal) likes, and **up** on actions in the data:

$$
\min_Q \; \alpha \, \mathbb{E}_{s \sim \mathcal{D}} \Big[ \log \sum_a \exp Q(s,a) \;-\; \mathbb{E}_{a \sim \pi_\beta(\cdot \mid s)} \big[ Q(s,a) \big] \Big] \;+\; \tfrac{1}{2} \, \mathbb{E}_{(s,a,s') \sim \mathcal{D}} \Big[ \big( Q(s,a) - \hat{\mathcal{B}}^\pi \bar Q(s,a) \big)^2 \Big]
$$

The log-sum-exp is a soft maximum over all actions, so the first term only lets Q be large on actions that also appear in the data. The paper shows the resulting Q lower-bounds the policy's true value (in expectation, under its assumptions). For continuous actions the log-sum-exp is estimated by sampling a handful of actions from a uniform distribution and from the current policy, and importance-weighting them.

Costs to keep in mind: \(\alpha\) is sensitive and dataset-dependent, and every critic update evaluates Q on several sampled actions per state — a multiplier on learner cost (Section 8).

---

## 5. Never query unknown actions: Implicit Q-Learning

IQL (Kostrikov, Nair, Levine, 2021) avoids the problem at its root: it never evaluates Q at an action that isn't in the dataset. The trick is to make \(V(s)\) estimate the value of the *best in-data actions* without computing a max:

$$
L_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \Big[ L_2^\tau\big( \bar Q(s,a) - V_\psi(s) \big) \Big], \qquad L_2^\tau(u) = \big\lvert \tau - \mathbb{1}(u < 0) \big\rvert \, u^2
$$

$$
L_Q(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}} \Big[ \big( r + \gamma V_\psi(s') - Q_\theta(s,a) \big)^2 \Big]
$$

\(L_2^\tau\) is **expectile regression**. With \(\tau = 0.5\) it is ordinary MSE, so \(V\) learns the *average* Q of the behavior policy (a SARSA-style value, no improvement). As \(\tau \to 1\), errors where Q exceeds V are penalized much more, so \(V\) rises toward the *upper end* of the in-data action values: an implicit max restricted to the data's support. Values around 0.7-0.9 are typical. The Q backup uses \(V(s')\), not \(Q(s', \pi(s'))\), so no OOD action is ever queried.

The policy is then extracted with advantage-weighted regression (Section 3.1), using \(A = \bar Q(s,a) - V(s)\):

$$
L_\pi(\phi) = - \, \mathbb{E}_{(s,a) \sim \mathcal{D}} \Big[ \exp\!\big( \beta_{\text{inv}} (\bar Q(s,a) - V(s)) \big) \log \pi_\phi(a \mid s) \Big]
$$

(here \(\beta_{\text{inv}}\) is an *inverse* temperature, as in the IQL paper). Value learning and policy extraction are fully decoupled: you can train the critic once and try several policy-extraction methods on top of it. IDQL (Section 7) exploits exactly that.

IQL can stitch, because the Q/V backup is still dynamic programming. And because the expectile is an in-sample max, it is less prone to the runaway values of Section 2.2.

---

## 6. Offline-to-online: keeping what you learned

Offline pre-training gives a policy that is good but limited by the data's coverage. The natural next step is to fine-tune it online. Three things go wrong:

* **The fine-tuning dip.** A conservative critic (CQL) systematically underestimates actions it hasn't seen. When online data arrives, values of newly tried actions jump up, the policy chases them, and performance often drops sharply before recovering — sometimes below where you started. Cal-QL (Nakamoto et al., 2023) addresses this by learning a value that is conservative but *calibrated*: a lower bound on the learned policy's value, but not pushed below the value of a reference policy such as the behavior policy. The paper implements it as a one-line change to CQL.
* **Over-constraint.** Behavior regularization that was essential offline now prevents the policy from moving to better actions it can test. Anneal it, or switch to a less constrained update.
* **Buffer imbalance.** A large offline buffer plus a trickle of online data means almost every batch is stale offline data.

### 6.1 RLPD: the simple, strong default

RLPD (Ball, Smith, Kostrikov, Levine, 2023) asks whether a standard off-policy method can just use offline data online, and finds that it can with a few changes:

```text
SAC, from scratch (no offline pre-training), plus:
  1. symmetric sampling — every batch is 50% offline data, 50% online data
  2. LayerNorm in the critic — bounds how far Q can extrapolate to unseen actions
  3. an ensemble of critics with a pessimistic min over a random subset (REDQ-style)
  4. high update-to-data ratio — many gradient steps per environment step
```

Symmetric sampling fixes the buffer imbalance by construction. LayerNorm addresses Section 2.2's extrapolation without an explicit conservatism term. High UTD extracts more learning per expensive online sample. Because it skips offline pre-training entirely, there is no pre-trained conservative critic to "un-learn," so there is no dip of the CQL kind — though the early online phase still depends on how good the offline data is.

> **Robot connection.** For the arm, RLPD is the natural "reuse all old runs" algorithm: drop every demo, every failed rollout, and every previous policy's episodes into the offline half of the batch, and train online in sim with the other half. It needs an actor whose action log-probs or reparameterized samples are available (it is SAC underneath). For a flow-matching π0.5, that is the obstacle Section 7 deals with.

---

## 7. RL for diffusion and flow policies: IDQL, FQL, steering

A flow-matching policy ([Lecture 03](Lecture-03.md)) generates an action chunk by integrating a learned velocity field from noise \(w \sim \mathcal{N}(0, I)\) for ~10 steps. That makes it expressive (multimodal, high-dimensional chunks), but awkward for the actor updates above: there is no cheap exact \(\log \pi(a \mid s)\) for AWR or SAC's entropy term, and backpropagating \(Q\) through 10 integration steps is expensive and unstable. Three practical patterns:

| Method | What is trained | Base flow policy | Inference cost | Bake into weights? |
|---|---|---|---|---|
| **IDQL** (Hansen-Estruch et al., 2023) | IQL-style critic; diffusion behavior policy by BC | Trained by BC only, never by RL | \(N\) samples × NFE, then critic picks | Needs distillation |
| **FQL** (Park, Li, Levine, 2025) | Flow BC policy + a separate **one-step** policy trained to maximize Q while staying close to the flow policy's output | BC only; the one-step policy does the RL | One forward pass | The one-step policy *is* the deployable network |
| **DSRL** (Wagenmaker et al., 2025) | A small policy over the **input noise** \(w\) of a frozen diffusion/flow policy | Frozen; black-box access only | Noise policy + base policy | Needs distillation |

**IDQL — sample, then let the critic choose.** Draw \(N\) candidate actions from the behavior-cloned diffusion policy and select among them using weights from the critic (the paper reinterprets IQL's critic as defining an implicit actor and importance-resamples toward it). Every candidate is in-distribution by construction, because it comes from the BC model. This is critic-guided sample-and-rank, the continuous-action version of [Lecture 07](Lecture-07.md)'s "sample many moves, keep the best."

**FQL — a fast helper that stays close to the flow.** Train the flow policy \(\mu_\phi(s, w)\) with ordinary flow matching on the data. Separately train a one-step network \(\mu_\omega(s, w)\) with

$$
L(\omega) = \mathbb{E}_{s \sim \mathcal{D}, \, w \sim \mathcal{N}(0,I)} \Big[ - Q\big(s, \mu_\omega(s, w)\big) + \alpha \, \big\lVert \mu_\omega(s, w) - \mu_\phi(s, w) \big\rVert^2 \Big]
$$

where \(\mu_\phi(s, w)\) means "integrate the flow from noise \(w\)." The distillation term keeps the one-step policy's outputs near what the expressive flow would have produced *for the same noise*; the Q term pushes it toward higher value. No gradient flows through the integration loop, and at test time there is no iterative generation at all.

**DSRL — steer the noise, not the action.** Treat the frozen flow policy as a deterministic map \(w \mapsto a = \mu_\phi(s, w)\) and run RL over \(w\): a small actor \(\pi_\psi(w \mid s)\) chooses the noise, the base policy turns it into an action chunk. The output is always something the base policy can produce, so it stays a sensible movement. The base weights are never modified, training needs only forward passes of the base model, and any off-policy algorithm (SAC in noise space) works. Keep \(w\) inside a bounded range (for example, tanh-scaled to a few standard deviations), otherwise the actor will find noise vectors the base policy never saw in training — the OOD problem again, moved into noise space.

### 7.1 Bake it back into the weights

IDQL and DSRL both improve behavior *around* a policy without changing its weights. That is a problem when the deliverable is the policy itself — the capstone ([Lecture 14](Lecture-14.md)) is scored on a checkpoint, and a deployment stack (the [VLA course](../VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-01.md)) runs one network under a deadline. The fix is **distillation**: run the improved system (critic-selected samples, or the noise-steered policy) in sim, keep the resulting \((o, \text{chunk})\) pairs (optionally weighted by advantage, or only from successful episodes), and fine-tune the flow policy on them with the ordinary flow-matching loss. Then evaluate the distilled checkpoint *alone*, without critic or noise policy, on nominal and perturbed layouts. If the distilled number is much lower than the steered number, the improvement didn't transfer into the weights.

---

## 8. The hardware view: offline is learner-bound

With no environment in the loop, Lecture 01's iteration time collapses to

$$
t_{\text{iter}} \;\approx\; t_{\text{load}}(B) + t_{\text{learn}}(B)
$$

and the bottleneck moves to the **data pipeline and the learner**.

**Dataset memory.** A state-based PickCube dataset of \(10^6\) transitions with a ~50-dim observation is \(10^6 \times 50 \times 4\) bytes ≈ 200 MB per observation array: it fits on the GPU, and sampling is a single indexing op. A pixel dataset is a different machine: two 224×224×3 uint8 cameras are 301,056 bytes per step, so \(10^6\) steps is ≈ 300 GB. That lives on host RAM or disk; you need a multi-worker loader, JPEG/video decode (on CPU workers or GPU decoders), and pinned-memory transfers, and you should measure loader throughput separately from learner throughput. Pre-computing frozen vision features is not automatically smaller: a few hundred visual tokens at a hidden width of 1-2k in bf16 is roughly 0.5-1 MB per image, *more* than the raw frame. Store pixels compressed, or pool features aggressively.

**Learner cost per method:**

| Method | Networks updated | Extra forward passes per batch | Notes |
|---|---|---|---|
| BC / filtered BC | \(\pi\) | — | Cheapest; regime "supervised" |
| AWR | \(\pi\), \(V\) | — | MC returns precomputed once |
| TD3+BC | \(\pi\), 2×\(Q\) | target nets | ~TD3 cost |
| CQL | \(\pi\), 2×\(Q\) | Q on \(M\) sampled actions per state (uniform + policy) | Learner cost multiplied by sampled actions |
| IQL | \(\pi\), 2×\(Q\), \(V\) | — | Never samples actions in critic updates |
| RLPD | \(\pi\), ensemble of \(Q\) | ensemble × UTD | Deliberately learner-heavy |

**Where this matters for VLAs (regime C).** Online RL on π0.5 spends most of its GPU-hours on rollouts. Offline methods remove that term entirely and pay only learner cost. That cost is substantial for a multi-billion-parameter policy, but it is the same well-understood cost as supervised fine-tuning. The flow-policy methods differ sharply in what they put on the GPU:

* **IDQL** keeps the VLA frozen during training, but multiplies inference by \(N\) candidates × NFE. Batch the \(N\) candidates (they share one VLM prefix), or distill.
* **FQL** needs gradients through the one-step policy, which for a VLA means backprop through at least the action expert. At inference it removes the ~10-step integration entirely: a direct latency win for the [VLA course](../VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-01.md).
* **DSRL** never backpropagates through the VLA. Training memory is the small noise actor and critic; each environment step still costs one VLA forward pass, so it is regime C for rollouts but cheap for learning.
* **Distillation** is one supervised fine-tuning run: predictable, and it converts any of the above into weights.

---

## 9. Build it: from a mixed dataset to online fine-tuning

You will build `offline/` on rung 2 (ManiSkill3 `PickCube-v1`, state observations, sparse reward).

### Lab 1 — Build a mixed dataset

Collect episodes from four sources, tagged so you can ablate them later: (a) your Lecture 02 scripted expert, (b) the expert with Gaussian action noise \(\sigma \in \{0.1, 0.3\}\), (c) early and mid PPO checkpoints from Lecture 06 (mostly failures and partial grasps), (d) uniform random actions. Store episodes as fixed-length arrays with a length per episode, ending each at its first success:

```python
# offline/collect.py
import torch, gymnasium as gym, mani_skill.envs

@torch.no_grad()
def rollout(policy, n_envs=512, T=50, seed=0, tag="expert"):
    env = gym.make("PickCube-v1", num_envs=n_envs, obs_mode="state",
                   control_mode="pd_ee_delta_pose", reward_mode="sparse")
    obs, _ = env.reset(seed=seed)
    O, A, R = [], [], []
    done = torch.zeros(n_envs, dtype=torch.bool, device=obs.device)
    ep_len = torch.full((n_envs,), T, device=obs.device)
    for t in range(T):
        act = policy(obs)
        nobs, rew, term, trunc, info = env.step(act)
        O.append(obs); A.append(act); R.append(rew.float())
        newly = info["success"] & ~done
        ep_len[newly] = t + 1; done |= newly
        obs = nobs
    O.append(obs); env.close()
    return dict(obs=torch.stack(O, 1).cpu(), act=torch.stack(A, 1).cpu(), rew=torch.stack(R, 1).cpu(),
                ep_len=ep_len.cpu(), success=done.cpu(), tag=[tag] * n_envs)
```

Flatten to transitions with the per-episode mask, treat the success step as terminal (no bootstrap past it), and treat the time limit as truncation (bootstrap — the Lecture 05 bug, again). Report the dataset composition: transitions and success rate per source.

**Optional stitching set.** Build a second dataset where no single episode succeeds end-to-end: "grasp-and-hold" episodes that lift the cube and then wander, plus episodes that *start* with the cube already grasped (restore those states with `set_state`, as in Lecture 09's bonus) and carry it to the goal. Filtered BC has nothing to copy here; a value-based method should still succeed.

### Lab 2 — BC, filtered BC, AWR, IQL

All four share one deterministic MLP policy, trained with a weighted MSE loss (a Gaussian with fixed variance, so a weighted MSE *is* weighted log-likelihood):

```python
# offline/iql.py  (core update; BC = weights 1, filtered BC = weights from success tag)
import torch, torch.nn as nn, torch.nn.functional as F

def mlp(i, o, h=256, ln=False):
    norm = (lambda: nn.LayerNorm(h)) if ln else (lambda: nn.Identity())
    return nn.Sequential(nn.Linear(i, h), norm(), nn.ReLU(), nn.Linear(h, h), norm(), nn.ReLU(), nn.Linear(h, o))

class TwinQ(nn.Module):
    def __init__(self, ds, da):
        super().__init__(); self.q1, self.q2 = mlp(ds + da, 1), mlp(ds + da, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], -1); return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

def expectile(diff, tau):
    return (torch.where(diff > 0, tau, 1 - tau) * diff ** 2).mean()

def iql_step(batch, Q, Q_targ, V, pi, opts, tau=0.7, beta_inv=3.0, gamma=0.99, polyak=0.005):
    s, a, r, s2, term = batch
    with torch.no_grad():
        q_t = torch.min(*Q_targ(s, a))
    v = V(s).squeeze(-1)
    loss_v = expectile(q_t - v, tau)                              # in-sample "max"
    with torch.no_grad():
        target = r + gamma * (1 - term) * V(s2).squeeze(-1)       # never queries pi(s2)
    q1, q2 = Q(s, a)
    loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    with torch.no_grad():
        w = torch.exp(beta_inv * (q_t - v)).clamp(max=100.0)     # AWR weights
    loss_pi = (w * ((pi(s) - a) ** 2).sum(-1)).mean()
    for loss, opt in zip((loss_v, loss_q, loss_pi), opts):
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        for p, pt in zip(Q.parameters(), Q_targ.parameters()):
            pt.lerp_(p, polyak)
    return dict(v=v.mean().item(), q=q1.mean().item(), w_max=w.max().item())
```

For AWR, replace \(\bar Q - V\) with \(G_t - V(s_t)\) using precomputed discounted Monte-Carlo returns and a \(V\) fitted to them by regression. Keep the whole dataset on the GPU and sample batches by random indexing.

Run all four methods on the mixed dataset (≥ 3 seeds) and evaluate every 10k updates on 256 parallel environments, in-distribution and with the cube spawned outside the dataset's initial range. Log mean predicted Q on dataset actions next to the actual discounted return: with a 0/1 reward, a Q above 1 is a bug or extrapolation. Then run a **naive offline SAC/TD3** (no regularization) on the same data and watch its Q diverge. Keep that plot; it is the clearest picture of Section 2.2.

### Lab 3 — Online fine-tuning with RLPD

Switch to online training with your Lecture 07 SAC, plus the RLPD changes. The only structural code is the batch sampler and the critic:

```python
# offline/rlpd.py  (changes to your Lecture 07 SAC)
critics = nn.ModuleList([mlp(ds + da, 1, ln=True) for _ in range(n_critics)])   # LayerNorm critics

def sample_batch(B):
    off = offline_buf.sample(B // 2)          # tensors on GPU, from Lab 1
    on = online_buf.sample(B - B // 2)        # filled by the GPU-parallel env
    return [torch.cat([x, y]) for x, y in zip(off, on)]

def target_q(s2, a2):                         # pessimistic min over a random subset of the ensemble
    idx = torch.randperm(n_critics)[:2]
    x = torch.cat([s2, a2], -1)
    return torch.min(torch.stack([critics_targ[i](x).squeeze(-1) for i in idx]), 0).values
```

Compare four runs on the same online budget: (a) SAC from scratch, no data; (b) RLPD with the mixed dataset; (c) IQL pre-trained offline, then continued online with IQL updates; (d) RLPD with only the expert subset. Sweep UTD \(\in \{1, 4, 16\}\) for (b). Plot success vs online env steps *and* vs wall-clock. For (c), measure the **dip**: the drop from the offline score to the minimum during the first 10% of online training.

### Lab 4 (optional) — Noise-space steering on your flow policy

Take the Lecture 03 flow-matching chunked policy trained on PickCube demos. Freeze it. Train a SAC actor whose action is the noise \(w\) (shape: chunk length × 7, tanh-scaled to \([-2.5, 2.5]\)) and whose critic is \(Q(s, w)\). The environment wrapper calls the frozen flow policy on \((s, w)\) and executes the first \(H\) actions of the chunk. Compare against the unsteered flow policy. Then **distill**: collect 50k \((s, \text{chunk})\) pairs from successful steered episodes, fine-tune the flow policy with flow matching, and evaluate the distilled weights alone. Report all three numbers side by side.

---

## 10. Use it in the real stack

* **Reference implementations:** [IQL](https://github.com/ikostrikov/implicit_q_learning) (JAX, original), [RLPD](https://github.com/ikostrikov/rlpd), [FQL](https://github.com/seohongpark/fql), [DSRL](https://github.com/ajwagen/dsrl), and [CORL](https://github.com/tinkoff-ai/CORL) (single-file PyTorch offline RL baselines in CleanRL style — diff your Lab 2 against its IQL and TD3+BC).
* **Robot datasets** — Open X-Embodiment style mixtures, LIBERO's demonstration sets, and your own sim rollouts — are almost all demos without dense rewards. Offline RL on them needs a reward: sim success labels (free), a learned success detector ([Lecture 08](Lecture-08.md)), or Monte-Carlo success-to-go computed from episode outcomes.
* **For π0.5 (rung 3):** the cheapest first experiment is filtered BC / AWR-weighted fine-tuning on the policy's own LIBERO rollouts (successes up-weighted, failures down-weighted or dropped), using openpi's fine-tuning path. It needs no critic in the VLA, no log-probs, and no online loop. It is the baseline any fancier method in [Lecture 11](Lecture-11.md) and the capstone has to beat.

---

## 11. Measure it

| Metric | Definition | Why it matters |
|---|---|---|
| **success rate** | In-distribution and perturbed initial poses, ≥ 3 seeds | The goal |
| **Q overestimation** | Mean predicted Q on dataset (and policy) actions minus MC return | Diagnoses Section 2.2; must stay ≤ 1 for 0/1 rewards |
| **AWR weight stats** | Max and effective sample size of the weights per batch | Too peaked = effectively imitating a handful of samples |
| **learner-updates/s** | Per method, same batch size | Offline wall-clock is all learner |
| **loader throughput** | Samples/s from the data pipeline alone | Must exceed what the learner consumes (pixel datasets!) |
| **fine-tuning dip** | Offline score minus minimum score early in online training | Offline-to-online stability |
| **online samples-to-success** | Env steps to X% with vs without offline data | The value of the dataset |
| **wall-clock-to-success** | Including offline pre-training time | What you actually pay |
| **steered vs distilled gap** | Success of steered system minus distilled weights alone | Did the improvement make it into the policy? |

---

## 12. Ship it

Commit `offline/` containing:

* `collect.py`, `iql.py` (with BC / filtered BC / AWR modes), `rlpd.py`, optional `dsrl.py`, and a `run_all.sh`
* `dataset_card.md` — sources, transitions, success rate per source, observation/action spec, terminal vs truncation handling
* `offline_table.csv` + `offline_table.md` — BC vs filtered BC vs AWR vs IQL (and naive offline SAC), mean ± std over ≥ 3 seeds, in-distribution and perturbed, plus learner-updates/s per method
* `q_divergence.png` — predicted Q vs true return for naive offline SAC vs IQL
* `offline_to_online.png` — success vs online env steps and vs wall-clock for the four Lab 3 runs, with the dip annotated
* `OFFLINE_NOTE.md` — which method you would use first on π0.5 rollouts and why, with your learner-cost and dip measurements as evidence

---

## Exit criteria

You can move on when you can:

* explain, with the Bellman backup on the board, why offline Q-learning overestimates OOD actions and why online RL self-corrects
* derive \(\pi^* \propto \pi_\beta \exp(A/\beta)\) from the KL-constrained problem and turn it into the weighted-BC loss
* write the CQL regularizer and the IQL expectile loss, and say which one queries OOD actions and which never does
* show a stitching case from your own data where IQL beats filtered BC, or explain why your data didn't allow one
* run RLPD and report its samples-to-success and wall-clock against SAC from scratch
* name one way to apply critic-based improvement to a flow-matching policy, and describe how you would bake the result back into the weights and verify that it transferred

---

## Self-check

1. Your offline TD3 run on PickCube reports mean Q-values of 25 after 200k updates, with a 0/1 sparse reward and \(\gamma = 0.99\), and evaluation success is 3%. Explain exactly what happened in terms of the Bellman backup, and give two different one-change fixes from different families.
2. On your stitching dataset (no end-to-end successes), filtered BC gets 0% and IQL gets a nonzero success rate. Explain both numbers. What happens to IQL as \(\tau \to 0.5\)?
3. A CQL-pretrained policy at 60% offline success drops to 20% within the first 20k online steps, then climbs to 85%. Why does the dip happen, and what would you try to avoid it? Would RLPD show the same dip, and why or why not?
4. You have 200 GPU-hours to improve π0.5 on one LIBERO suite. An online PPO run spends most of its time on VLA rollouts. You also have 5,000 stored rollouts from earlier evaluations, with success labels. Using the Section 8 cost model, argue which method you would run first and what result would make you switch to online RL.
5. IDQL with \(N = 32\) candidates per decision gives the best success in your sim, but triples inference latency and misses the 20 Hz deadline. Give two ways to keep most of the gain within the deadline, and the measurement that tells you whether each one worked.
6. A competition only accepts model weights. Your DSRL noise policy raised success from 70% to 82%, but after distilling into the flow policy the checkpoint scores 73%. List three plausible causes and the experiment that distinguishes them.

---

## References

* CS 285 lectures on offline RL and offline-to-online fine-tuning — [slides and video](https://rail.eecs.berkeley.edu/deeprlcourse/)
* Levine, Kumar, Tucker, Fu, "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems," 2020 — [paper](https://arxiv.org/abs/2005.01643)
* Peng, Kumar, Zhang, Levine, "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning," 2019 — [paper](https://arxiv.org/abs/1910.00177)
* Nair, Gupta, Dalal, Levine, "AWAC: Accelerating Online Reinforcement Learning with Offline Datasets," 2020 — [paper](https://arxiv.org/abs/2006.09359)
* Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning" (TD3+BC), 2021 — [paper](https://arxiv.org/abs/2106.06860)
* Kumar, Zhou, Tucker, Levine, "Conservative Q-Learning for Offline Reinforcement Learning," 2020 — [paper](https://arxiv.org/abs/2006.04779)
* Kostrikov, Nair, Levine, "Offline Reinforcement Learning with Implicit Q-Learning," 2021 — [paper](https://arxiv.org/abs/2110.06169)
* Ball, Smith, Kostrikov, Levine, "Efficient Online Reinforcement Learning with Offline Data" (RLPD), 2023 — [paper](https://arxiv.org/abs/2302.02948)
* Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning," 2023 — [paper](https://arxiv.org/abs/2303.05479)
* Hansen-Estruch, Kostrikov, Janner, Kuba, Levine, "IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies," 2023 — [paper](https://arxiv.org/abs/2304.10573)
* Park, Li, Levine, "Flow Q-Learning," 2025 — [paper](https://arxiv.org/abs/2502.02538)
* Wagenmaker et al., "Steering Your Diffusion Policy with Latent Space Reinforcement Learning" (DSRL), 2025 — [paper](https://arxiv.org/abs/2506.15799)

---

## Next in this special course

* Next: [Lecture 11 — RL for LLMs and VLAs](Lecture-11.md)
* Previous: [Lecture 09 — Model-Based RL and World Models](Lecture-09.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
