# Lecture 09: Model-Based RL and World Models

## Overview

Every method so far has been **model-free**: the policy or the critic learns directly from trial and error and never asks *why* the world responded the way it did. Model-based RL asks that question explicitly. It learns a **model** of the world — "if I do this here, what happens next?" — and then uses the model to plan, or to practice in imagination, instead of paying for every try in the real environment.

An everyday picture: a new driver who has learned how the car responds to the wheel and the pedals can rehearse a tricky parking manoeuvre in their head before trying it. That rehearsal is cheap and safe. The catch is that it is only as good as their mental model of the car. If that model has a bug — say they think the car turns much tighter than it does — then rehearsing harder makes things worse, because the rehearsal keeps selecting manoeuvres that only work inside the buggy model. A video game with a collision bug, where you can drive through a wall, is the same failure: a player optimizing for score will find the wall.

That is the central tension of this lecture. A learned model can make RL far more **sample-efficient**, and an optimizer pointed at a learned model will **exploit its errors**. Every practical method is a way of using the model while staying where it is trustworthy: short imagined rollouts, uncertainty estimates from ensembles, re-planning at every step.

This lecture follows CS 185/285 topics 15-16. The hardware thread is new: in model-based control, **planning is batched inference inside the control loop**. Candidates × horizon × ensemble members all have to fit before the next control tick.

By the end you should be able to:

* train a probabilistic ensemble dynamics model and separate aleatoric from epistemic uncertainty
* explain model exploitation formally, and name three ways to limit it
* implement random shooting and CEM-based model-predictive control (MPC), and estimate their latency from the model's forward-pass time
* derive why model error compounds with rollout length, and explain why Dyna/MBPO use short branched rollouts from real states
* describe at a high level what latent world models (Dreamer, TD-MPC2) learn and how they plan or train policies
* say when model-based RL is worth it in wall-clock terms on each rung of the course's environment ladder

---

## 1. Why it matters: learn the rules, then practice in your head

| | Model-free (Lectures 04-08) | Model-based (this lecture) |
|---|---|---|
| What is learned | \(\pi(a \mid s)\), \(V(s)\), \(Q(s,a)\) | \(\hat p(s_{t+1} \mid s_t, a_t)\), \(\hat r(s_t, a_t)\) (plus often \(\pi\), \(V\)) |
| Supervision | A scalar reward, delayed | Every transition is a fully labelled regression example |
| Sample efficiency | Low to moderate | Often much higher |
| Main failure | Variance, no signal | Model bias and exploitation |
| Compute spent at | Training time | Training **and** decision time (planning) |

The second row is the key to sample efficiency. A transition \((s, a, s')\) gives the model-free learner one noisy scalar (the reward). It gives a dynamics model a dense target: the whole next-state vector. The model can learn from *every* transition, including ones where nothing rewarding happened. That is why model-based methods usually need far fewer environment steps — the exact factor depends heavily on the task, so treat any number you read as task-specific and measure your own.

> **Robot connection.** A real robot is regime C for data: every episode costs minutes of wall-clock and a human to reset the table. A simulator is a *perfect* world model you already have during training — you can save the exact state where the policy nearly dropped the bowl and replay from there. A learned model's ensemble members disagreeing is a signal for where the robot should collect more practice. Both ideas show up in the lab.

---

## 2. Mental model: dynamics models and how they lie

### 2.1 Training a dynamics model

The simplest recipe (Nagabandi et al., 2017) is supervised regression on transitions collected by any policy. Two practical choices matter far more than architecture:

* **Predict the change, not the next state:** \(\hat s_{t+1} = s_t + f_\theta(s_t, a_t)\). Most of the state barely moves in one control step, so predicting \(\Delta s\) is a much easier target.
* **Normalize** inputs and \(\Delta s\) per dimension. Joint angles, velocities, and positions differ by orders of magnitude.

A **deterministic** model minimizes \(\lVert s_{t+1} - s_t - f_\theta(s_t, a_t) \rVert^2\). A **probabilistic** model outputs a Gaussian over \(\Delta s\), with mean \(\mu_\theta\) and diagonal variance \(\sigma^2_\theta\), and minimizes the negative log-likelihood:

$$
\mathcal{L}(\theta) = \sum_{(s,a,s')} \sum_{j} \left[ \frac{\big(\Delta s_j - \mu_{\theta,j}(s,a)\big)^2}{\sigma^2_{\theta,j}(s,a)} + \log \sigma^2_{\theta,j}(s,a) \right], \qquad \Delta s = s' - s
$$

The variance term lets the model say "this transition is inherently noisy here" (contact events, slipping objects) instead of chasing the noise with its mean.

### 2.2 Model exploitation, formally

Let \(J(\mathbf{a})\) be the true return of an action sequence and \(\hat J(\mathbf{a}) = J(\mathbf{a}) + e(\mathbf{a})\) its estimate under the model, with error \(e\). The planner picks \(\hat{\mathbf{a}} = \arg\max_{\mathbf{a}} \hat J(\mathbf{a})\). Then

$$
\hat J(\hat{\mathbf{a}}) \;\ge\; \hat J(\mathbf{a}^*) \quad\Rightarrow\quad \mathbb{E}\big[\hat J(\hat{\mathbf{a}})\big] \;\ge\; J(\mathbf{a}^*) + \mathbb{E}\big[e(\mathbf{a}^*)\big]
$$

So even with an unbiased model (\(\mathbb{E}[e] = 0\) everywhere), the *selected* plan's predicted return is biased upward. The harder you optimize, the more the argmax is drawn to wherever \(e\) happens to be large and positive. This is the same "optimizer's curse" behind Q-learning overestimation in [Lecture 07](Lecture-07.md) and OOD overestimation in offline RL ([Lecture 10](Lecture-10.md)).

Errors are largest where the model has seen no data, and a policy that improves visits exactly those states. The fixes all bound how far you trust the model:

1. **Re-plan every step (MPC).** Execute only the first action, observe the real result, plan again. A wrong prediction gets corrected after one step instead of after a whole episode.
2. **Keep imagined rollouts short** and start them from real states (Section 4).
3. **Stay where the model is confident** — penalize or truncate imagined rollouts where uncertainty is high.
4. **Iterate data collection:** run the planner in the real environment, add those transitions, refit. The model gets corrected exactly where the planner was tempted.

### 2.3 Two kinds of uncertainty, and ensembles

| | Aleatoric ("dice") | Epistemic ("knowledge") |
|---|---|---|
| Source | The world is genuinely stochastic or partially observed | The model hasn't seen enough data here |
| More data helps? | No | Yes |
| Estimated by | Predicted variance \(\sigma^2_\theta\) | Disagreement between independently trained models |
| What to do with it | Plan for the spread of outcomes | Avoid, or go collect data there |

Train an **ensemble** of \(E\) models (5 is a common choice) on bootstrap resamples of the data with different random inits (Lakshminarayanan et al., 2017). With member means \(\mu_e\) and variances \(\sigma_e^2\), the law of total variance splits the predictive variance cleanly:

$$
\underbrace{\operatorname{Var}[\Delta s]}_{\text{total}} \;=\; \underbrace{\frac{1}{E}\sum_{e} \sigma_e^2}_{\text{aleatoric}} \;+\; \underbrace{\frac{1}{E}\sum_{e} \big(\mu_e - \bar\mu\big)^2}_{\text{epistemic}}, \qquad \bar\mu = \frac{1}{E}\sum_e \mu_e
$$

On data the members have all seen, they agree. Far from the data, each extrapolates differently, so disagreement grows. That is a cheap epistemic signal. It is not a calibrated probability — you will plot disagreement against actual error in the lab to see how well it tracks.

**PETS** (Chua et al., 2018) combines a probabilistic ensemble with *trajectory sampling*: each imagined trajectory ("particle") is assigned to an ensemble member and samples from that member's Gaussian at every step. The spread of particle returns then reflects both kinds of uncertainty, and the planner optimizes their mean.

---

## 3. Planning with a model: shooting, CEM, MPPI, MPC

Given a model, the simplest controller needs no policy at all. At each step, solve

$$
\mathbf{a}^*_{t:t+H-1} = \arg\max_{a_t, \dots, a_{t+H-1}} \; \mathbb{E}_{\hat p}\Big[ \textstyle\sum_{k=0}^{H-1} \hat r(s_{t+k}, a_{t+k}) \Big]
$$

execute \(a^*_t\), observe \(s_{t+1}\), and solve again. That is **model-predictive control** with a receding horizon \(H\). The optimizers, in increasing sophistication:

| Optimizer | How | Notes |
|---|---|---|
| **Random shooting** | Sample \(N\) action sequences, simulate each in the model, keep the best | Embarrassingly parallel. Works surprisingly well for small action spaces and short \(H\); degrades as \(H \times d_a\) grows. |
| **CEM** (cross-entropy method) | Sample \(N\) sequences from a Gaussian; keep the top \(K\) "elites"; refit mean and std to the elites; repeat \(I\) times | The workhorse of PETS-style MPC. Same parallelism as shooting, much better in higher dimensions. |
| **MPPI** | Like CEM, but weight *all* samples by \(\exp(\hat J / \lambda)\) instead of a hard top-\(K\) | Smoother updates. Standard in robotics MPC (Williams et al., 2017), and the planner inside TD-MPC. |
| **Gradient-based** | Backpropagate \(\hat J\) through a differentiable model into the actions | Cheap per iteration, but long-horizon gradients through learned dynamics are poorly conditioned and easily exploit the model. |
| **Tree search** | Expand discrete actions with a search tree (MCTS) | Dominant for discrete games; awkward for continuous 7-D arm actions. |

Two tricks make CEM-MPC practical:

* **Warm start:** shift last step's optimized mean by one step and use it as the next initial mean. The previous plan is usually almost right.
* **Execute more than one step** if the planner is slow — the same receding-horizon trade-off as executing \(H\) steps of a VLA action chunk ([Lecture 03](Lecture-03.md)).

The connection to chunking is direct: a VLA that emits a 50-step chunk is a policy that has *amortized* the planning step — it outputs in one forward pass what a CEM planner would have searched for.

---

## 4. Learning policies from models: Dyna, MBPO, and compounding error

### 4.1 Why long imagined rollouts fail

Suppose the true dynamics are \(s' = f(s,a)\) and the model \(\hat f\) has one-step error at most \(\epsilon\), and \(\hat f\) is \(L\)-Lipschitz in \(s\). Let \(e_k = \lVert \hat s_k - s_k \rVert\) be the error after \(k\) imagined steps from the same start and the same actions:

$$
e_{k+1} = \lVert \hat f(\hat s_k, a_k) - f(s_k, a_k) \rVert \;\le\; \lVert \hat f(\hat s_k, a_k) - \hat f(s_k, a_k) \rVert + \lVert \hat f(s_k, a_k) - f(s_k, a_k) \rVert \;\le\; L e_k + \epsilon
$$

Unrolling from \(e_0 = 0\):

$$
e_k \;\le\; \epsilon \,\frac{L^k - 1}{L - 1}
$$

which is linear in \(k\) when \(L \approx 1\) and exponential when \(L > 1\). Contact-rich manipulation has steep local dynamics (a tiny gripper offset decides whether the bowl is grasped), so the effective \(L\) is large exactly where it matters. And this is the error for a *fixed* action sequence. A policy optimized against the model also drifts into states where \(\epsilon\) itself is larger. This is the model-based version of the compounding error in behavioral cloning ([Lecture 02](Lecture-02.md)).

### 4.2 Dyna and MBPO: short branches from real states

**Dyna** (Sutton, 1991) mixes real experience with simulated experience generated by the model, and trains an ordinary model-free learner on both. **MBPO** (Janner et al., 2019) makes the recipe robust for deep RL:

```text
repeat:
  1. run policy in real env  → add to D_env
  2. fit probabilistic ensemble on D_env
  3. sample M real states from D_env
     from each, roll out the CURRENT policy in a random ensemble member for k steps  → D_model
  4. many SAC updates on batches drawn mostly from D_model
```

The important design choice is in step 3. Rollouts **branch from real states** and are **short** (\(k\) is often 1 to a handful of steps). Starting from real states keeps the imagined data on the real state distribution, and keeping \(k\) small keeps \(e_k\) small. The MBPO paper backs this with a bound on the gap between model return and true return that grows with \(k\), model error, and policy shift, and uses it to justify short rollouts with a gradually increasing \(k\).

Because step 4 trains on mostly synthetic data, MBPO runs at a very high **update-to-data ratio** — many gradient steps per real environment step. That is the same lever as the high-UTD off-policy methods in Lecture 07, with model rollouts supplying extra data.

---

## 5. Latent world models: Dreamer and TD-MPC2

Predicting raw pixels a few steps ahead is expensive and spends capacity on irrelevant detail (the wood grain on the table). **Latent world models** learn a compact state \(z_t\) — a "dream space" — and do all imagination there.

* **World Models** (Ha & Schmidhuber, 2018) — a VAE compresses frames to \(z\), a recurrent model predicts the next \(z\), and a small controller is trained inside the learned model. This is where the term "practicing in a dream" comes from.
* **Dreamer** (Hafner et al., 2019; DreamerV3, 2023) — a recurrent state-space model with deterministic and stochastic latent parts, trained with reconstruction, reward, and KL terms (the ELBO from [Lecture 08](Lecture-08.md)). An actor and a critic are trained purely on imagined latent trajectories started from real encoded states: the Dyna idea, moved into latent space and made differentiable. DreamerV3's contribution is a set of normalization and robustness choices that let one hyperparameter setting work across very different domains.
* **TD-MPC / TD-MPC2** (Hansen et al., 2022; 2023) — no decoder at all. The latent model is trained to predict future latents, rewards, and values (it is "task-oriented": it only has to model what matters for return). At decision time it runs MPPI in latent space over a short horizon, bootstraps the tail with a learned \(Q\), and uses a learned policy to seed the samples. This is a clean hybrid: planning handles the near future, the critic summarizes the far future.

| | Plans at decision time? | Decoder? | Policy learned? | Decision-time cost |
|---|---|---|---|---|
| PETS | yes (CEM) | n/a (state model) | no | high: \(I \times H\) model calls |
| MBPO | no | n/a | yes (SAC) | one policy forward |
| Dreamer | no | yes | yes (actor in imagination) | encoder + actor forward |
| TD-MPC2 | yes (MPPI) | no | yes (as sampling prior + \(Q\)) | encoder + \(I \times H\) latent steps |

A related research direction trains large action-conditioned **video world models** and uses them to evaluate or train robot policies without a physics simulator. The appeal for VLAs is obvious; the same exploitation warning applies, at much larger scale.

---

## 6. The hardware view: planning is batched inference in the control loop

For CEM-MPC with \(N\) candidates, \(P\) particles per candidate, horizon \(H\), and \(I\) CEM iterations, the work per control step is \(I \cdot H\) **sequential** model calls, each on a batch of \(N \cdot P\) states:

$$
t_{\text{plan}} \;\approx\; I \cdot H \cdot t_{\text{model}}(N \cdot P), \qquad \text{FLOPs} \;\approx\; 2 \cdot I \cdot H \cdot N \cdot P \cdot n_{\text{params}}
$$

The ensemble does not add sequential depth if you implement it as a batched matmul (`torch.baddbmm` over a leading ensemble dimension), so \(E\) members cost roughly one wider call. Two consequences:

* **Depth, not FLOPs, is usually the floor.** A 3-layer 256-wide MLP (~1.5 × 10⁵ parameters) on a batch of 10⁴ states is about 3 GFLOP — tens of microseconds of math on a modern GPU — but each call still launches several kernels. With \(I = 5\), \(H = 15\), that is 75 dependent calls, so kernel-launch overhead and Python dispatch set the latency (regime A behaviour inside a regime B problem). CUDA Graphs or `torch.compile` capture the whole unrolled plan and remove most of it.
* **The GPU makes CEM practical.** On a CPU, \(N \cdot P = 10^4\) model evaluations per step is slow; on a GPU it is one batch. That is why sampling-based MPC came back in deep RL once GPUs were standard.

The control deadline is fixed: at 20 Hz you have 50 ms for perception, planning, and communication. Every knob trades latency for plan quality:

| Knob | Latency effect | Quality effect |
|---|---|---|
| \(N\) (candidates) | ~flat until the GPU saturates, then linear | Better coverage of action space |
| \(P\) (particles) | multiplies batch like \(N\) | Better estimate of expected return under uncertainty |
| \(E\) (ensemble size) | small (batched) | Better epistemic estimate; more memory for weights |
| \(H\) (horizon) | **linear, sequential** | Sees further, but compounds model error |
| \(I\) (CEM iterations) | **linear, sequential** | Better optimum; more exploitation of model errors |

**Where model-based pays off on the ladder.** In regime B (GPU sim, 10⁵-10⁶ env steps/s), samples are nearly free, and a model-free learner like PPO usually wins on wall-clock — fitting and querying a model is extra learner work spent to save something that was already cheap. In regime C, or on a real robot, each sample is expensive, and the model's sample efficiency becomes worth its compute. Always report both samples-to-success and wall-clock-to-success.

**The simulator as a perfect model.** A GPU simulator with state save/restore *is* a world model with zero error. You can set \(N\) parallel environments to the same saved state and roll out \(N\) candidate plans in one batched sim step — CEM with the true dynamics. It is too slow and too privileged to deploy, but it is an excellent **teacher** (a scripted expert for DAgger in [Lecture 02](Lecture-02.md)) and an excellent tool for **resetting to hard states** during RL.

---

## 7. Build it: ensemble dynamics + CEM-MPC on PushCube

You will build `mbrl/` on rung 2: ManiSkill3 `PushCube-v1`, state observations, end-effector delta-pose control (7-D actions, matching the running example).

### Lab 1 — Collect transitions

```python
# mbrl/collect.py
import torch, gymnasium as gym
import mani_skill.envs

def make_env(n):
    return gym.make("PushCube-v1", num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pose", reward_mode="normalized_dense")

@torch.no_grad()
def collect(env, policy, episodes_per_env=4, T=50):
    S, A, R, S2 = [], [], [], []
    for ep in range(episodes_per_env):
        obs, _ = env.reset(seed=ep)
        for t in range(T):
            act = policy(obs)
            nobs, rew, term, trunc, info = env.step(act)
            S.append(obs); A.append(act); R.append(rew); S2.append(nobs)
            obs = nobs
    return [torch.cat(x) for x in (S, A, R, S2)]

env = make_env(256)
space = env.unwrapped.single_action_space                 # per-env bounds, shape [7]
lo, hi = [torch.as_tensor(b, device="cuda") for b in (space.low, space.high)]
rand = lambda o: lo + (hi - lo) * torch.rand(o.shape[0], lo.shape[-1], device="cuda")
torch.save(collect(env, rand), "mbrl/data_random.pt")
```

Random actions rarely touch the cube, so the model will know almost nothing about pushing. Seed the dataset with noisy rollouts of a scripted expert (adapt your PickCube expert from Lecture 02: move behind the cube, push toward the goal) and later add MPC rollouts — the PETS loop of plan → collect → refit. Use `single_action_space` for bounds: the batched `env.action_space` carries a leading env dimension.

### Lab 2 — Probabilistic ensemble

```python
# mbrl/model.py
import torch, torch.nn as nn, torch.nn.functional as F

class EnsembleLinear(nn.Module):
    def __init__(self, E, din, dout):
        super().__init__()
        self.W = nn.Parameter(torch.randn(E, din, dout) / din**0.5)
        self.b = nn.Parameter(torch.zeros(E, 1, dout))
    def forward(self, x):                      # x: [E, B, din]
        return torch.baddbmm(self.b, x, self.W)

class EnsembleDynamics(nn.Module):
    def __init__(self, E, ds, da, h=256):
        super().__init__()
        self.E, self.ds = E, ds
        self.body = nn.ModuleList([EnsembleLinear(E, ds + da, h), EnsembleLinear(E, h, h), EnsembleLinear(E, h, h)])
        self.head = EnsembleLinear(E, h, 2 * ds + 1)           # mean Δs, log-var Δs, reward
        self.max_lv = nn.Parameter(torch.full((1, 1, ds), 0.5))
        self.min_lv = nn.Parameter(torch.full((1, 1, ds), -10.0))
        for n in ("s_mu", "s_sd", "a_mu", "a_sd", "d_mu", "d_sd"):
            self.register_buffer(n, torch.zeros(0))

    def set_norm(self, S, A, S2):
        D = S2 - S
        self.s_mu, self.s_sd = S.mean(0), S.std(0) + 1e-6
        self.a_mu, self.a_sd = A.mean(0), A.std(0) + 1e-6
        self.d_mu, self.d_sd = D.mean(0), D.std(0) + 1e-6

    def forward(self, s_n, a_n):               # normalized inputs, [E, B, *]
        x = torch.cat([s_n, a_n], -1)
        for layer in self.body:
            x = F.silu(layer(x))
        out = self.head(x)
        mu, lv, r = out[..., :self.ds], out[..., self.ds:-1], out[..., -1]
        lv = self.max_lv - F.softplus(self.max_lv - lv)        # soft-bound log-variance (as in PETS code)
        lv = self.min_lv + F.softplus(lv - self.min_lv)
        return mu, lv, r

    def step(self, s, a):                      # raw units in and out
        mu_d, lv_d, r = self((s - self.s_mu) / self.s_sd, (a - self.a_mu) / self.a_sd)
        return s + self.d_mu + mu_d * self.d_sd, lv_d + 2 * self.d_sd.log(), r

def train(model, S, A, R, S2, steps=20_000, B=256, lr=1e-3):
    model.set_norm(S, A, S2)
    D_n = ((S2 - S) - model.d_mu) / model.d_sd
    S_n, A_n = (S - model.s_mu) / model.s_sd, (A - model.a_mu) / model.a_sd
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for it in range(steps):
        idx = torch.randint(0, len(S), (model.E, B), device=S.device)   # bootstrap per member
        mu, lv, r = model(S_n[idx], A_n[idx])
        nll = (((mu - D_n[idx]) ** 2) * (-lv).exp() + lv).mean()
        loss = nll + F.mse_loss(r, R[idx]) + 0.01 * (model.max_lv.sum() - model.min_lv.sum())
        opt.zero_grad(); loss.backward(); opt.step()
```

Hold out 10% of transitions. Report held-out one-step NLL and MSE per ensemble member, and the **k-step open-loop error** for \(k = 1, 5, 15, 30\) by rolling the model forward on held-out action sequences. Compare the curve with the \(\epsilon (L^k - 1)/(L - 1)\) shape from Section 4.1.

### Lab 3 — CEM-MPC with trajectory sampling

```python
# mbrl/cem.py
import torch

@torch.no_grad()
def imagine_return(model, s0, acts, P):
    """s0: [M, ds]; acts: [M, N, H, A]; P particles per candidate, split across E members (TS-inf)."""
    M, N, H, A = acts.shape; E = model.E; p = P // E
    s = s0[:, None, None].expand(M, N, p, -1).reshape(1, M * N * p, -1).expand(E, -1, -1)
    a = acts[:, :, None].expand(M, N, p, H, A).reshape(1, M * N * p, H, A).expand(E, -1, -1, -1)
    ret = 0.0
    for t in range(H):                                   # sequential: the latency floor
        mu, lv, r = model.step(s, a[:, :, t])
        s = mu + torch.randn_like(mu) * (0.5 * lv).exp() # sample aleatoric noise
        ret = ret + r
    return ret.view(E, M, N, p).mean(dim=(0, 3))         # [M, N] mean over particles

@torch.no_grad()
def cem_plan(model, s0, mean, lo, hi, N=500, K=50, iters=5, P=20):
    M, H, A = mean.shape
    std = (hi - lo).expand(M, H, A) / 4
    for _ in range(iters):
        acts = (mean[:, None] + std[:, None] * torch.randn(M, N, H, A, device=s0.device)).clamp(lo, hi)
        ret = imagine_return(model, s0, acts, P)
        idx = ret.topk(K, dim=1).indices
        elite = torch.gather(acts, 1, idx[..., None, None].expand(-1, -1, H, A))
        mean, std = elite.mean(1), elite.std(1).clamp_min(0.02)
    return mean                                          # [M, H, A]

def run_mpc(env, model, lo, hi, H=15, T=50, seed=0, **kw):
    obs, _ = env.reset(seed=seed)
    M, A = obs.shape[0], lo.shape[-1]
    mean = ((lo + hi) / 2).expand(M, H, A).clone()
    for t in range(T):
        plan = cem_plan(model, obs, mean, lo, hi, **kw)
        obs, rew, term, trunc, info = env.step(plan[:, 0])
        mean = torch.cat([plan[:, 1:], plan[:, -1:]], 1)  # warm start: shift by one step
    return info["success"].float().mean().item()
```

Run MPC with 64-256 parallel environments planning simultaneously (the planner is batched over \(M\)). Then add a **conservative variant**: subtract \(\lambda \cdot\) (ensemble disagreement summed over the horizon) from each candidate's return. Compare success, and inspect whether the unpenalized planner picks plans the true environment doesn't reproduce.

### Lab 4 — Latency sweep and the Pareto plot

```python
# mbrl/latency.py
import itertools, time, torch
def time_plan(model, s0, mean, lo, hi, iters=20, **kw):
    for _ in range(3): cem_plan(model, s0, mean, lo, hi, **kw)        # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): cem_plan(model, s0, mean, lo, hi, **kw)
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters
```

Sweep \(N \in \{100, 500, 2000\}\), \(H \in \{5, 15, 30\}\), \(E \in \{1, 5, 10\}\), \(I \in \{1, 3, 5\}\) at \(M = 1\) (one robot, the deployment case). Plot success against planning latency, one point per configuration, and draw the Pareto front. Mark the 50 ms (20 Hz) line. Then wrap `cem_plan` in `torch.compile(mode="reduce-overhead")` (which uses CUDA Graphs) and re-measure the front.

### Lab 5 — Does disagreement predict error?

On held-out transitions *and* on states visited by your MPC controller, compute per-transition epistemic disagreement (Section 2.3) and the true one-step error of the ensemble mean. Scatter-plot them on log axes and report the Spearman rank correlation. Repeat with states from a perturbed initial distribution (cube spawned outside the training range). Disagreement should rise there; check whether error rises with it.

### Bonus — Reset-to-near-failure curriculum

ManiSkill3 exposes `env.unwrapped.get_state()` (a flat `[num_envs, D]` tensor) and `env.unwrapped.set_state(state, env_idx)`. During PPO training (Lecture 06), keep a buffer of states from steps shortly before a failed episode ended, or where your critic's \(V\) dropped sharply. At each reset, restore a fraction (say 25%) of environments from that buffer instead of the default initial distribution:

```python
# sketch — integrate into your PPO loop from Lecture 06
hard = state_buffer.sample(n_hard)                       # [n_hard, D] saved sim states
obs, _ = env.reset()
env.unwrapped.set_state(hard, env_idx=torch.arange(n_hard, device="cuda"))
obs = env.unwrapped.get_obs()                            # re-read observations after restoring
```

Check that task information (for PushCube, the goal position) is part of what `get_state` saves on your version; the base class docs note that task information may need to be handled by the task. Compare success on the *default* initial distribution with and without the curriculum, so you can see whether hard-state practice transfers or overfits.

---

## 8. Use it in the real stack

* **PETS / MBPO reference code:** both papers released code; MBPO's ensemble and branched-rollout logic is the clearest to read alongside your Lab 2.
* **TD-MPC2:** [code](https://github.com/nicklashansen/tdmpc2) with released multi-task checkpoints — the most practical modern model-based baseline for continuous control.
* **DreamerV3:** [code](https://github.com/danijar/dreamerv3) (JAX).
* **Classical MPC in robotics:** MPPI-style samplers run on GPUs in many legged-locomotion and manipulation stacks, usually with *analytic* models rather than learned ones. The learned-model version replaces the physics model with your ensemble; the sampling machinery is the same.
* **For VLAs:** world models are entering policy evaluation and data generation. If you use one to score a π0.5 checkpoint, treat its numbers with the same suspicion as a learned success detector ([Lecture 08](Lecture-08.md)): validate against the real simulator on a held-out set before trusting rankings.

---

## 9. Measure it

| Metric | Definition | Why it matters |
|---|---|---|
| **one-step NLL / MSE** | Held-out, per ensemble member | Basic model quality |
| **k-step open-loop error** | Error after \(k\) imagined steps, \(k \in \{1,5,15,30\}\) | Measures compounding; sets a safe \(H\) or \(k\) |
| **disagreement–error correlation** | Spearman between epistemic estimate and true error | Is the uncertainty usable? |
| **planning latency** | ms per control step at \(M=1\), p50 and p99 | Must fit the control deadline |
| **plans/s** | Planned control steps per second at \(M = 64\)-\(256\) | Throughput when collecting data |
| **success rate** | In-distribution and perturbed initial cube poses | The actual goal |
| **samples-to-success** | Real env steps (model training data) to reach X% | Model-based's selling point |
| **wall-clock-to-success** | Including model fitting and planning | What it really costs; compare with your PPO from Lecture 06 |
| **peak GPU memory** | During planning at max \(N \cdot P\) | Sets max candidates per batch |

---

## 10. Ship it

Commit `mbrl/` containing:

* `collect.py`, `model.py`, `cem.py`, `latency.py`, and a `run_all.sh`
* `kstep_error.png` — open-loop error vs \(k\) per ensemble member, log scale
* `disagreement_vs_error.png` — scatter for held-out, MPC-visited, and perturbed states, with Spearman values in the legend
* `pareto.png` + `latency.csv` — success vs planning latency over the (\(N, H, E, I\)) sweep, eager vs compiled, with the 20 Hz deadline marked
* `MBRL_NOTE.md` — one paragraph each: (1) which knob dominated latency and why (sequential depth vs batch), (2) samples-to-success and wall-clock-to-success vs your Lecture 06 PPO, (3) one concrete case where the planner exploited the model, and what fixed it

---

## Exit criteria

You can move on when you can:

* write the Gaussian NLL for a probabilistic dynamics model and the law-of-total-variance split into aleatoric and epistemic parts
* show, with the argmax argument, why planning against an unbiased model still overestimates the chosen plan's return
* derive \(e_k \le \epsilon (L^k - 1)/(L - 1)\) and use it to explain MBPO's short branched rollouts
* implement CEM-MPC and predict its latency from \(I\), \(H\), and the model's per-call time — then confirm with a measurement
* explain, with your own numbers, whether model-based RL beat PPO on PushCube in samples and in wall-clock, and why the answer flips on a real robot

---

## Self-check

1. Your CEM planner reports a predicted return 40% higher than the return the real environment delivers when you execute the plan, and the gap grows when you raise CEM iterations from 3 to 10. What is happening, and what are two changes that should shrink the gap without collecting new data?
2. Your ensemble's members agree closely on a state where the gripper is about to close on the cube edge, yet the prediction is badly wrong. Is that an aleatoric or an epistemic failure, and why can't ensemble disagreement catch it? What in the data collection would you change?
3. Planning takes 80 ms at \(N = 500\), \(H = 15\), \(I = 5\) and 82 ms at \(N = 2000\). You need 50 ms. Which knobs do you turn, and what do you do to the implementation before touching any knob at all?
4. On rung 2, MBPO reaches 70% success on PushCube with 10× fewer env steps than PPO but takes 3× longer in wall-clock. A colleague concludes MBPO is "worse." Using the regime vocabulary from Lecture 01, explain when they are right and when they are wrong, and what would change for a real-robot version of the task.
5. You save sim states from just before failures and reset 50% of environments to them. Training success on those states climbs quickly, but success from the default initial distribution falls. Give two explanations and an experiment that separates them.
6. A VLA team proposes training π0.5 with RL entirely inside a learned video world model to avoid simulator engineering. Using Sections 2.2 and 4.1, list the risks and the minimum validation you would require before trusting a reported improvement.

---

## References

* CS 285 lectures on optimal control and planning, model-based RL, and model-based policy learning — [slides and video](https://rail.eecs.berkeley.edu/deeprlcourse/)
* Nagabandi, Kahn, Fearing, Levine, "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning," 2017 — [paper](https://arxiv.org/abs/1708.02596)
* Chua, Calandra, McAllister, Levine, "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS), 2018 — [paper](https://arxiv.org/abs/1805.12114)
* Lakshminarayanan, Pritzel, Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," 2017 — [paper](https://arxiv.org/abs/1612.01474)
* Sutton, "Dyna, an Integrated Architecture for Learning, Planning, and Reacting," SIGART Bulletin, 1991
* Janner, Fu, Zhang, Levine, "When to Trust Your Model: Model-Based Policy Optimization" (MBPO), 2019 — [paper](https://arxiv.org/abs/1906.08253)
* Williams et al., "Information Theoretic MPC for Model-Based Reinforcement Learning" (MPPI), ICRA 2017
* Ha & Schmidhuber, "World Models," 2018 — [paper](https://arxiv.org/abs/1803.10122)
* Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer), 2019 — [paper](https://arxiv.org/abs/1912.01603); "Mastering Diverse Domains through World Models" (DreamerV3), 2023 — [paper](https://arxiv.org/abs/2301.04104)
* Hansen, Wang, Su, "Temporal Difference Learning for Model Predictive Control" (TD-MPC), 2022 — [paper](https://arxiv.org/abs/2203.04955); Hansen, Su, Wang, "TD-MPC2: Scalable, Robust World Models for Continuous Control," 2023 — [paper](https://arxiv.org/abs/2310.16828)
* ManiSkill3 docs — [maniskill.readthedocs.io](https://maniskill.readthedocs.io/) (state save/restore, controllers, reward modes)

---

## Next in this special course

* Next: [Lecture 10 — Offline RL and Offline-to-Online Fine-Tuning](Lecture-10.md)
* Previous: [Lecture 08 — The Probabilistic Toolkit: ELBO, VAEs, Control as Inference, Inverse RL](Lecture-08.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
