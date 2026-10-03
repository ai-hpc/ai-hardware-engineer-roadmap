# Lecture 04: Policy Gradients: Do More of What Worked

## Overview

Teaching a puppy a trick works without explaining physics to it. The puppy tries things. When a try earns a treat, it gets a little more likely to do that again. When a try earns nothing, that try gets a little less likely. After enough tries the trick appears, and nobody ever wrote down the right movement.

A **policy gradient** method does exactly this to a neural network. The policy gives every action a probability. You run it many times, look at the scores, and nudge the parameters so that actions from high-scoring tries become more probable and actions from low-scoring tries become less probable. Lectures 02-03 needed an expert to say what the right action was. Here, nobody does. The method needs only two things:

* a **score** for each try (for our arm: did the bowl end up on the plate?)
* the policy's own **log-probability** of the actions it took, and its gradient

It needs no model of the dynamics, no differentiable simulator, and no expert labels. That is surprising at first, and Section 3 shows exactly why it works. The price is noise. A lucky initial layout can make bad moves look good. Most of this lecture is about reducing that noise: baselines, reward-to-go, and big batches. The last of these sets the hardware budget.

By the end you should be able to:

* derive \(\nabla_\theta J(\theta)\) with the log-derivative trick and explain why the dynamics \(p(s_{t+1} \mid s_t, a_t)\) drop out
* write the REINFORCE estimator for a discrete (softmax) and a continuous (Gaussian) policy
* prove that subtracting a state-dependent baseline leaves the gradient unbiased, and explain reward-to-go as a consequence of causality
* build a **per-initial-state group baseline** (the GRPO / RLOO idea) on a GPU simulator and explain why groups where every try scored the same give no signal
* measure gradient variance directly and explain why batch size is set by variance, not by GPU memory

---

## 1. Why it matters: learning without an expert

| | Behavioral cloning (L02-03) | Policy gradient (this lecture) |
|---|---|---|
| Needs | Expert action for every visited state | A score per episode |
| Gradient | \(\nabla \log \pi_\theta(a^\star \mid s)\) for expert action \(a^\star\) | \(\nabla \log \pi_\theta(a \mid s) \times\) how well \(a\) turned out, for the policy's **own** actions |
| Data | Fixed dataset, reusable | Fresh rollouts from the current policy, used once |
| Ceiling | The expert | Whatever the score rewards |
| Main failure | Distribution shift | High variance; zero signal when every try scores the same |

Look at the gradient row. A policy gradient is **behavioral cloning on your own tries, weighted by how well each try went**. Good tries get positive weight (copy them), bad tries get negative weight (push away from them). Every RL post-training method for VLAs in Lecture 11 is a variant of this one idea, including GRPO, which drives most LLM reasoning post-training.

> **Robot connection.** π0.5 already succeeds on most nominal LIBERO layouts. When it fails, typically on a layout where objects were moved, there is no expert on hand to say what it should have done. There is a success check. Policy gradients turn "succeeded / failed on this layout" into a parameter update. In simulation you can go further: restart the **same** table setup several times and compare each try to *that setup's* average. This is the group baseline of Section 5, and it is the same trick chat models use under the name GRPO.

---

## 2. Mental model: nudge, weighted by surprise

In one sentence, the update for every action the policy took is:

```text
  change in that action's chance  =  (direction that makes it more likely)  ×  (its score − the usual score)
                                       ∇θ log πθ(a | s)                           R − b
```

* **Direction.** \(\nabla_\theta \log \pi_\theta(a \mid s)\) is a vector in parameter space. A small step along it makes action \(a\) more likely in state \(s\). You get it from autograd, exactly as in BC.
* **Weight.** The score minus a baseline. Positive means the action was part of a better-than-usual try, so make it more likely. Negative means worse than usual, so make it less likely. Zero means the try was normal and nothing changes.
* **Average over many tries.** One try is a terrible estimate of whether an action was good, because the outcome also depends on luck: the layout, sampling noise, and contact physics. Averaging over many tries cancels the luck.

Three refinements turn this sentence into a working algorithm. Each is a variance-reduction step, and each is derived below:

| Refinement | Plain language | Effect |
|---|---|---|
| **Baseline** \(b\) | Compare to the usual score, not to zero | Removes the "everything scored positive, so push everything up" noise |
| **Reward-to-go** | An action only gets credit for what happens *after* it | Removes noise from rewards the action could not have caused |
| **Big batches** | Average over many tries before stepping | Variance falls as 1/(batch size) |

---

## 3. The derivation

### 3.1 The log-derivative trick

The objective from Lecture 01 is an expectation over trajectories:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\big[R(\tau)\big] = \int p_\theta(\tau)\, R(\tau)\, d\tau, \qquad R(\tau) = \sum_{t=0}^{T-1} r(s_t, a_t)
$$

\(R(\tau)\) is not differentiable in \(\theta\). For our arm it is a 0/1 step. The trajectory *distribution* is differentiable, so move the gradient onto it and use \(\nabla p = p \, \nabla \log p\):

$$
\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau = \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau = \mathbb{E}_{\tau \sim p_\theta}\big[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\big]
$$

The right-hand side is an expectation under the current policy, so it can be estimated by **running the policy and averaging**. The reward is only evaluated, never differentiated. This is why a sparse binary success signal is enough.

### 3.2 Why the dynamics drop out

Write out the probability of a trajectory:

$$
p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, p(s_{t+1} \mid s_t, a_t)
$$

Take the log, which turns the product into a sum, and then the gradient:

$$
\nabla_\theta \log p_\theta(\tau) = \underbrace{\nabla_\theta \log p(s_0)}_{0} + \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) + \sum_t \underbrace{\nabla_\theta \log p(s_{t+1} \mid s_t, a_t)}_{0}
$$

The initial-state distribution and the physics do not depend on \(\theta\), so their gradients are zero. **The dynamics never appear in the gradient.** They still matter: they decide which states and rewards you see. But you only need to *sample* from them, by running the simulator or the real robot. You never need to write them down or differentiate through them. Contacts, friction, and the non-differentiable success check are all fine.

The general form of this estimator, \(\mathbb{E}[\nabla \log p \cdot f]\), is known as the **score-function** or **likelihood-ratio** estimator. Lecture 08 contrasts it with the reparameterization gradient.

### 3.3 REINFORCE

Substituting gives the estimator from Williams (1992), averaged over \(N\) sampled trajectories:

$$
\nabla_\theta J(\theta) \approx \hat g = \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i) \right) R(\tau^i)
$$

The algorithm:

```text
repeat:
  1. run πθ for N episodes            → (s, a, r) for every step          (TRY)
  2. compute a weight per step         → R(τ), reward-to-go, minus baseline (JUDGE)
  3. loss = −(1/N) Σ_i Σ_t log πθ(a_t|s_t) · weight_t                     (IMPROVE)
     one gradient step; throw the data away
```

In code, the "gradient" is just the gradient of a **pseudo-loss**: a weighted negative log-likelihood whose weights are treated as constants. This is weighted BC on your own samples. Note the \(1/N\) normalization: the estimator averages over *trajectories*. Dividing by the total number of steps instead silently changes the effective learning rate whenever episode lengths change.

### 3.4 Log-probs for the two policy types

**Discrete (CartPole).** The network outputs logits \(z\), and \(\pi(a \mid s) = \mathrm{softmax}(z)_a\):

$$
\nabla_z \log \pi(a \mid s) = \mathbf{e}_a - \pi(\cdot \mid s)
$$

The update raises the logit of the taken action and lowers the others in proportion to their probability.

**Continuous (our arm).** A diagonal Gaussian with a network for the mean \(\mu_\theta(s)\) and a learned, state-independent log-std \(\log\sigma\). This is the standard choice in CleanRL, ManiSkill, and rsl_rl. For a \(d\)-dimensional action:

$$
\log \pi_\theta(a \mid s) = -\sum_{j=1}^{d} \left[ \frac{(a_j - \mu_{\theta,j}(s))^2}{2\sigma_j^2} + \log \sigma_j \right] - \frac{d}{2}\log 2\pi, \qquad \frac{\partial \log \pi}{\partial \mu_j} = \frac{a_j - \mu_j}{\sigma_j^2}
$$

The mean moves **toward** sampled actions with positive weight and **away** from those with negative weight. The step is larger when \(\sigma\) is small. Exploration is the sampling noise \(\sigma\) itself: if \(\sigma\) collapses, every try is the same and there is nothing left to compare. In practice, sample \(a\) from the Gaussian, **clip only the copy sent to the environment** to the action bounds, and compute the log-prob of the unclipped sample.

---

## 4. Taming the variance

The estimator is unbiased but extremely noisy. Suppose every episode scores between 0 and 1. With no baseline, *every* sampled action gets a non-negative weight, so every action is pushed up and the signal survives only through small differences in the pushes. Three fixes follow.

### 4.1 Reward-to-go (causality)

An action at time \(t\) cannot affect rewards that arrived before \(t\). Formally, for \(t' < t\):

$$
\mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, r_{t'}\big] = \mathbb{E}_{s_{0:t}, a_{0:t-1}}\Big[ r_{t'} \; \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\big] \Big] = 0
$$

The inner expectation is zero because \(\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a \mid s)] = \int \nabla \pi(a \mid s)\, da = \nabla \int \pi(a \mid s)\, da = \nabla 1 = 0\). Past rewards add zero-mean noise, so drop them:

$$
\hat g = \frac{1}{N}\sum_i \sum_t \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i)\, \hat Q_t^i, \qquad \hat Q_t^i = \sum_{t'=t}^{T-1} r_{t'}^i
$$

With a single terminal success reward, every step in an episode has the same reward-to-go, so this fix does nothing for the sparse case. It matters for dense or shaped rewards, and the time-indexed baseline below still helps.

### 4.2 Baselines are free

Subtract any \(b(s_t)\) that does not depend on \(a_t\). The same identity shows the expected change is zero:

$$
\mathbb{E}_{a_t \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, b(s_t)\big] = b(s_t)\, \mathbb{E}_{a_t \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\big] = 0
$$

So \(\mathbb{E}[\hat g]\) is unchanged while the variance can drop a lot. The natural choice is \(b(s_t) \approx V^\pi(s_t)\), the expected reward-to-go from \(s_t\). The weight then becomes an advantage estimate, \(\hat Q_t - V(s_t) \approx A^\pi(s_t, a_t)\): did this try do better or worse than usual *from here*? Lecture 05 learns \(V\) with a network. This lecture uses cheaper Monte Carlo baselines.

**The variance-optimal constant baseline** is not the mean return. Minimizing the variance of \(g_k = \partial_k \log p_\theta(\tau)\,(R - b)\) for parameter \(k\) gives

$$
b_k^\star = \frac{\mathbb{E}\big[(\partial_k \log p_\theta(\tau))^2 R(\tau)\big]}{\mathbb{E}\big[(\partial_k \log p_\theta(\tau))^2\big]}
$$

This is the return weighted by gradient magnitude. Nobody computes it per parameter in practice. The mean return is the usual cheap approximation, and it captures most of the gain.

**Caveat on "free".** A baseline computed from the same batch, including the sample it is subtracted from, is very slightly correlated with that sample's action. This bias is \(O(1/N)\). The leave-one-out form in Section 5 removes it exactly.

### 4.3 Variance vs batch size

\(\hat g\) is a mean of \(N\) i.i.d. per-trajectory terms, so \(\mathrm{Var}[\hat g] = \sigma_g^2 / N\). A useful diagnostic is the **relative variance**:

$$
\rho(N) = \frac{\mathbb{E}\,\lVert \hat g - \bar g \rVert^2}{\lVert \bar g \rVert^2} = \frac{\rho(1)}{N}
$$

When \(\rho \gg 1\), a step is mostly noise. To make \(\rho \approx 1\) you need \(N \approx \rho(1)\) trajectories per step. A good baseline divides \(\rho(1)\), and therefore the required batch, by a large factor. That is a direct saving in rollout compute. **The batch size of a policy-gradient method is set by the variance of the estimator, not by the GPU memory of the learner.** Lab 4c measures \(\rho\) for each estimator.

---

## 5. Group baselines: compare each try to its own setup

Here is a cheap Monte Carlo estimate of \(V^\pi(s_0)\). Start \(G\) episodes from the **same initial state** \(s_0\) (same layout, same goal) and average their returns. The tries differ only in policy sampling noise (and in any physics noise), so the group mean measures how good the current policy is *on this setup*. Each try is then judged against that.

For try \(i\) in a group with returns \(R_1, \dots, R_G\):

| Variant | Advantage \(A_i\) | Notes |
|---|---|---|
| Group mean | \(R_i - \tfrac{1}{G}\sum_j R_j\) | Includes \(R_i\) in its own baseline. Biased low by a factor \((1 - 1/G)\), which is only a rescaling. |
| Leave-one-out (RLOO) | \(R_i - \tfrac{1}{G-1}\sum_{j \ne i} R_j = \tfrac{G}{G-1}\big(R_i - \bar R\big)\) | Baseline independent of try \(i\): exactly unbiased |
| GRPO | \(\big(R_i - \bar R\big) / (\mathrm{std}(R) + \epsilon)\) | Also rescales each group by its spread. Used in DeepSeekMath. Later work questions the std division because it re-weights groups by difficulty. |

Why this beats a batch-wide mean on a robot task: different layouts have very different difficulty. A batch-wide baseline treats "succeeded on an easy layout" as good news and "failed on a hard layout" as bad news, but most of that variance comes from the **layout, not the action**. The group baseline removes it.

### 5.1 The zero-signal problem

With binary success, a group in which every try succeeded, or every try failed, has \(R_i = \bar R\) for all \(i\). **Every advantage is exactly zero and the group contributes no gradient.** Those rollouts were paid for and taught nothing. If the policy's success rate on a layout is \(p\), the probability that a group of \(G\) is informative is \(1 - p^G - (1-p)^G\):

| Success rate \(p\) on this layout | \(G=2\) | \(G=4\) | \(G=8\) | \(G=16\) |
|---|---|---|---|---|
| 0.01 or 0.99 | 0.02 | 0.04 | 0.08 | 0.15 |
| 0.05 or 0.95 | 0.10 | 0.19 | 0.34 | 0.56 |
| 0.2 or 0.8 | 0.32 | 0.59 | 0.83 | 0.97 |
| 0.5 | 0.50 | 0.88 | 0.99 | 1.00 |

Two consequences matter for the capstone. First, a strong base policy such as π0.5 on nominal layouts sits near \(p \approx 1\), so **most groups are wasted**. The signal lives on layouts it sometimes fails. Second, a policy that never succeeds learns nothing at all from sparse reward. Lecture 12 turns this into a compute-allocation policy: sample initial states whose success rate is strictly between 0 and 1. For now, **log the fraction of zero-variance groups**. It is a direct measure of wasted rollout compute.

> **Robot connection: the flow-matching puzzle.** Everything above needs \(\log \pi_\theta(a \mid s)\). For a Gaussian or softmax head it is one line. π0.5's action expert is a **flow-matching** model: it turns noise into a 50-step action chunk by integrating a learned velocity field about ten times. Its exact likelihood requires integrating the divergence of that field along the whole path, which is expensive and noisy to estimate. So "just run REINFORCE on π0.5" is not straightforward. Lecture 11 covers the workarounds (noise injection that makes each denoising step Gaussian, noise-space steering, advantage-weighted regression). The group-baseline machinery you build here carries over unchanged.

---

## 6. The hardware view: on-policy means every sample is used once

Plug REINFORCE into the Lecture 01 iteration model with one pass over the data (\(E = 1\)) and one gradient step per batch:

$$
t_{\text{iter}} \approx T \cdot \big(t_{\text{env}}(N) + t_{\text{policy}}(N)\big) + t_{\text{learn}}(N T)
$$

Every one of the \(N T\) transitions costs an environment step and a policy forward pass, contributes to a **single** gradient step, and is then discarded. Whether that is affordable depends entirely on the regime:

| Regime | Cost of one batch of \(N = 1024\) episodes of 50 steps | Verdict for REINFORCE |
|---|---|---|
| **A** CPU toy (CartPole) | Python env loop dominates; GPU idle | Fine for learning the math; vectorize envs |
| **B** GPU sim (ManiSkill3, MLP) | ~51k transitions in one batched loop; seconds or less | **Data is nearly free.** Spend it on large batches and large groups. |
| **C** VLA in sim (π0.5) | 1024 × (50/H) action-chunk inferences of a multi-billion-parameter model | **Every sample is expensive.** Throwing data away after one step hurts. This pushes toward reuse (L06 PPO epochs, L07 off-policy, L10 offline). |

Memory and compute details that matter in practice:

* **Do not keep the autograd graph from the rollout.** Run the rollout under `torch.no_grad()`, store `(obs, action)`, and recompute log-probs in one batched learner forward pass. Keeping graphs across a 50-step rollout of 1024 envs is how a tiny MLP runs out of memory. For a VLA this is mandatory.
* **Rollout buffer size** is \(N \cdot T \cdot (\text{obs} + \text{act} + 2)\) floats. For state observations it is small. For pixels, store `uint8` (Lecture 06 does the arithmetic).
* **Group size multiplies rollouts per initial state.** With \(G = 8\) and 1024 envs you see 128 distinct layouts per batch. More groups covers more layouts; bigger groups give a lower-noise baseline and more informative groups (table above). It is a fixed budget split, and the right split depends on how varied the layouts are.
* **Samples-to-success vs wall-clock-to-success.** In regime B a better baseline mostly shows up in wall-clock (fewer iterations). In regime C it shows up as GPU-hours of VLA inference saved. Report both.

---

## 7. Build it

All code lives in `pg/`. Lab 4a is CPU only. Labs 4b-4c need ManiSkill3 and a GPU.

### Lab 4a — REINFORCE on CartPole (discrete, regime A)

```python
# pg/reinforce_cartpole.py
import argparse, numpy as np, torch, torch.nn as nn, gymnasium as gym

p = argparse.ArgumentParser()
p.add_argument("--weight", choices=["return", "return_mean", "rtg", "rtg_mean"], default="rtg_mean")
p.add_argument("--episodes", type=int, default=16)   # trajectories per gradient step
p.add_argument("--iters", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

env = gym.make("CartPole-v1")
policy = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

def episode():
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    O, A, R, done = [], [], [], False
    while not done:
        with torch.no_grad():                          # no graph kept during the rollout
            a = torch.distributions.Categorical(logits=policy(torch.as_tensor(obs))).sample().item()
        O.append(obs); A.append(a)
        obs, r, term, trunc, _ = env.step(a)
        R.append(r); done = term or trunc
    return np.array(O, np.float32), np.array(A), np.array(R, np.float32)

for it in range(args.iters):
    eps = [episode() for _ in range(args.episodes)]
    rtg = [np.cumsum(R[::-1])[::-1].copy() for _, _, R in eps]       # reward-to-go per step
    rets = np.array([g[0] for g in rtg])
    T = max(len(g) for g in rtg)
    pad = np.full((len(eps), T), np.nan)
    for i, g in enumerate(rtg):
        pad[i, : len(g)] = g
    b_t = np.nanmean(pad, axis=0)                                     # time-indexed mean reward-to-go
    W = []
    for i, g in enumerate(rtg):
        w = {"return": np.full_like(g, rets[i]), "return_mean": np.full_like(g, rets[i] - rets.mean()),
             "rtg": g, "rtg_mean": g - b_t[: len(g)]}[args.weight]
        W.append(w)
    O = torch.as_tensor(np.concatenate([e[0] for e in eps])); A = torch.as_tensor(np.concatenate([e[1] for e in eps]))
    w = torch.as_tensor(np.concatenate(W), dtype=torch.float32)
    logp = torch.distributions.Categorical(logits=policy(O)).log_prob(A)
    loss = -(logp * w).sum() / args.episodes                          # average over trajectories
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 10 == 0: print(f"it {it:4d}  mean return {rets.mean():6.1f}")
```

Run all four `--weight` modes for 3 seeds each. Expected shape: `return` learns slowly and erratically, and `rtg_mean` reaches the 500 cap in noticeably fewer iterations. Keep the plots. Your numbers, not this sentence, are the result.

### Lab 4b — Gaussian REINFORCE on PushCube with per-initial-state groups (regime B)

The key systems trick is making \(G\) parallel environments start from **the same layout**. ManiSkill3's GPU tasks draw their randomness from one seeded torch generator per reset, so passing the same seed to several envs does not by itself guarantee identical layouts. Instead, reset normally, read the full simulator state, copy each group leader's state to its group, and reset to those states. An assertion checks the result.

```python
# pg/reinforce_pushcube.py
import argparse, time, torch, torch.nn as nn, gymnasium as gym
import mani_skill.envs  # registers PushCube-v1, PickCube-v1, ...

p = argparse.ArgumentParser()
p.add_argument("--baseline", choices=["none", "rtg", "batch", "group"], default="group")
p.add_argument("--num-envs", type=int, default=1024)
p.add_argument("--group", type=int, default=8)
p.add_argument("--reward", choices=["normalized_dense", "sparse"], default="normalized_dense")
p.add_argument("--iters", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed)
N, G = args.num_envs, args.group

env = gym.make("PushCube-v1", num_envs=N, obs_mode="state", reward_mode=args.reward, sim_backend="physx_cuda")
dev = env.unwrapped.device
obs_dim, act_dim = env.observation_space.shape[-1], env.action_space.shape[-1]
T = 50  # PushCube-v1 registers max_episode_steps=50

actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                      nn.Linear(256, act_dim)).to(dev)
log_std = nn.Parameter(torch.full((act_dim,), -0.5, device=dev))
opt = torch.optim.Adam(list(actor.parameters()) + [log_std], lr=3e-4)

def tree_index(x, idx):
    return {k: tree_index(v, idx) for k, v in x.items()} if isinstance(x, dict) else x[idx]

def reset_groups(seed):
    env.reset(seed=seed)                                         # N independent layouts
    leader = torch.arange(N, device=dev) // G * G                # env 0..G-1 copy env 0, etc.
    states = tree_index(env.unwrapped.get_state_dict(), leader)
    obs, _ = env.reset(options={"reset_to_env_states": {"env_states": states}})
    o = obs.view(N // G, G, -1)
    assert torch.allclose(o, o[:, :1].expand_as(o), atol=1e-5), "group members start from different layouts"
    return obs

def rollout(seed):
    obs = reset_groups(seed)
    O, A, R, M = [], [], [], []
    alive = torch.ones(N, device=dev); success = torch.zeros(N, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for t in range(T):
            mu = actor(obs); a = mu + log_std.exp() * torch.randn_like(mu)
            O.append(obs); A.append(a); M.append(alive.clone())
            obs, r, term, trunc, info = env.step(a.clamp(-1, 1))  # clip only what the env sees
            R.append(r.float() * alive)
            success |= info["success"].bool() & alive.bool()
            alive = alive * (~term.bool()).float()               # episode ends at first success
    return torch.stack(O), torch.stack(A), torch.stack(R), torch.stack(M), success

def weights(R):                                                  # R: (T, N)
    rtg = R.flip(0).cumsum(0).flip(0)                            # reward-to-go, gamma = 1
    if args.baseline == "none":  return rtg[:1].expand_as(rtg)   # whole-episode return on every step
    if args.baseline == "rtg":   return rtg
    if args.baseline == "batch": return rtg - rtg.mean(1, keepdim=True)
    g = rtg.view(T, N // G, G)                                   # leave-one-out group baseline
    return (g - (g.sum(-1, keepdim=True) - g) / (G - 1)).view(T, N)

for it in range(args.iters):
    t0 = time.perf_counter()
    O, A, R, M, success = rollout(seed=args.seed * 100_000 + it)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    W = weights(R)
    logp = torch.distributions.Normal(actor(O.flatten(0, 1)), log_std.exp()).log_prob(A.flatten(0, 1)).sum(-1)
    loss = -(logp * W.flatten() * M.flatten()).sum() / N
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    ret = R.sum(0).view(N // G, G)
    zero_groups = (ret.max(-1).values - ret.min(-1).values < 1e-6).float().mean().item()
    print(f"it {it:4d} succ {success.float().mean():.3f} zero-var groups {zero_groups:.2f} "
          f"rollout {t1 - t0:.2f}s learn {t2 - t1:.3f}s  env-steps/s {N * T / (t1 - t0):,.0f}")
```

Things to verify yourself rather than trust:

* The assertion passes. If it fires, the task keeps some layout state outside the simulator state. Inspect `get_state_dict().keys()`.
* `normalized_dense` is ManiSkill's shaped reward. With `--reward sparse` from a random init, success is near zero, almost every group is zero-variance, and REINFORCE stalls. **Run it anyway and record the zero-variance fraction**: it is the Section 5.1 table showing up in your logs. Lectures 06 (start from BC) and 12 (curricula) fix it.
* Success at the first step where `info["success"]` is true ends the episode for credit purposes (the `alive` mask). PushCube keeps simulating, but later rewards are masked.

### Lab 4c — Measure gradient variance directly

Freeze a policy: take a checkpoint from midway through Lab 4b, where success is between roughly 0.2 and 0.8, because at init with sparse reward every gradient is zero. Collect one large rollout with \(N = 4096\). Split it into disjoint sub-batches of \(n\) whole groups and compute the flattened gradient of each sub-batch with each estimator:

```python
def flat_grad(O, A, W, M, n_traj):
    logp = torch.distributions.Normal(actor(O.flatten(0, 1)), log_std.exp()).log_prob(A.flatten(0, 1)).sum(-1)
    loss = -(logp * W.flatten() * M.flatten()).sum() / n_traj
    grads = torch.autograd.grad(loss, list(actor.parameters()) + [log_std])
    return torch.cat([g.flatten() for g in grads])

# for each estimator, for n in (64, 128, 256, 512): g_m over disjoint sub-batches m = 1..M
# rel_var = mean_m ||g_m - g_bar||^2 / ||g_bar||^2, where g_bar = gradient of the full 4096-episode batch
```

Check that `rel_var` falls as \(1/n\) for each estimator. The ratio between estimators at fixed \(n\) is how many times more rollouts the weaker one needs for the same step quality. Write that ratio in the results table. It is the cost of a missing baseline, measured in environment steps.

---

## 8. Use it in the real stack

* **Spinning Up VPG** ([docs](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)) is the canonical clean derivation-plus-code for this lecture. It uses a learned value baseline, which is a preview of Lecture 05.
* **LLM post-training libraries** implement exactly Section 5. Hugging Face TRL ships GRPO and RLOO trainers, and the DeepSeekMath paper defines GRPO. The robot version replaces "G answers to the same prompt" with "G rollouts from the same initial simulator state".
* **VLA RL frameworks** such as [RLinf](https://github.com/RLinf/RLinf) run GRPO- and PPO-style training of VLAs on simulated benchmarks including LIBERO and ManiSkill. Read how their config defines a "group" for embodied tasks and compare it with `reset_groups` above. Whether group members truly share an initial state is the detail that decides whether the baseline is valid.
* **Production robot RL** rarely uses plain REINFORCE. It uses PPO (Lecture 06) with a learned critic (Lecture 05). Everything in those lectures is REINFORCE plus a better weight and a safer step.

---

## 9. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Success rate vs env steps** | Mean over ≥ 3 seeds, shaded min-max or CI | Sample efficiency of each estimator |
| **Success rate vs wall-clock** | Same runs, x-axis in seconds | What a better baseline buys on *this* hardware |
| **Relative gradient variance \(\rho(n)\)** | Lab 4c, per estimator, per sub-batch size | The direct mechanism behind the curves |
| **Rollouts-for-equal-\(\rho\)** | Ratio of \(n\) needed to reach the same \(\rho\) | Variance reduction converted into compute |
| **Zero-variance group fraction** | Share of groups whose returns are all equal | Wasted rollout compute; predicts stalls under sparse reward |
| **Rollout vs learn time** | `t1 - t0` vs `t2 - t1` per iteration | Confirms regime B: rollout ≫ learn for REINFORCE |
| **Policy entropy / \(\sigma\)** | `log_std.exp().mean()` per iteration | Collapse means no exploration and no comparisons left |

---

## 10. Ship it

Commit `pg/` containing:

* `reinforce_cartpole.py`, `reinforce_pushcube.py`, `grad_variance.py`, and `run_ablation.sh` (all estimators × 3 seeds)
* `results/variance_table.csv`: estimator, \(n\), relative variance, rollouts-for-equal-\(\rho\)
* `results/curves_steps.png` and `results/curves_wallclock.png`: success vs env steps and vs seconds, one line per estimator, ≥ 3 seeds each
* `results/zero_var_groups.png`: zero-variance group fraction over training, dense vs sparse reward
* `NOTES.md`: one paragraph explaining which estimator you would use for a regime C problem (π0.5 on LIBERO) and how many rollouts per gradient step you would budget, citing your variance numbers. Add the per-iteration rollout/learn split to your cost ledger from Lecture 01.

---

## Exit criteria

You can move on when you can:

* derive \(\nabla_\theta J = \mathbb{E}[\sum_t \nabla \log \pi_\theta(a_t \mid s_t)\, R]\) on a whiteboard and point to the line where the dynamics vanish
* prove in two lines that a state-dependent baseline is unbiased, and say why reward-to-go is a special case of the same identity
* explain why leave-one-out is unbiased and the plain group mean is a rescaling
* show your measured relative variance for each estimator and convert the gap into "x times more rollouts"
* predict, from a policy's per-layout success rate and group size, what fraction of rollouts will produce zero gradient

---

## Self-check

1. Your simulator has a contact bug that makes the success predicate non-differentiable and occasionally discontinuous. Does REINFORCE care? Would a method that backpropagates through the simulator care? Point to the equation that decides it.
2. You run GRPO-style training on PickCube with \(G = 8\) and sparse success reward. After 50 iterations the loss is exactly zero on 95% of groups and the policy is not changing. Give two possible states the policy could be in, how to tell them apart from logs you already have, and one fix for each.
3. A teammate normalizes the REINFORCE loss by the total number of time steps in the batch instead of the number of episodes. Episodes end early on success. What happens to the effective learning rate as the policy improves, and why might training appear to slow down right when it starts working?
4. On a CPU-simulated task with a VLA policy (regime C), switching from a batch-mean baseline to a group baseline cut relative gradient variance by 4×. Your manager asks what that is worth. Express the answer in VLA forward passes per gradient step, and say what you would do with the savings.
5. Why must the Gaussian log-prob be computed on the *unclipped* sampled action, even though the environment only saw the clipped one? What goes wrong in the gradient if you compute it on the clipped action and the mean drifts outside the action bounds?
6. Explain to someone who knows BC why "a policy gradient is weighted BC on your own samples" is accurate, and where the analogy breaks: what does BC have that REINFORCE lacks, and vice versa?

---

## References

* CS 285 Lecture 5 "Policy Gradients" — [slides and video](https://rail.eecs.berkeley.edu/deeprlcourse/)
* Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine Learning*, 1992 — [paper](https://link.springer.com/article/10.1007/BF00992696)
* Sutton, McAllester, Singh, Mansour, "Policy Gradient Methods for Reinforcement Learning with Function Approximation," NeurIPS 1999 — [paper](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
* Greensmith, Bartlett, Baxter, "Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning," JMLR 2004 — [paper](https://www.jmlr.org/papers/v5/greensmith04a.html) (baselines and their optimal form)
* Schulman et al., "Gradient Estimation Using Stochastic Computation Graphs," 2015 — [arXiv](https://arxiv.org/abs/1506.05254) (score-function vs pathwise estimators in one framework)
* OpenAI Spinning Up, "Intro to Policy Optimization" — [docs](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
* Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," 2024 — [arXiv](https://arxiv.org/abs/2402.03300) (introduces GRPO)
* Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs," 2024 — [arXiv](https://arxiv.org/abs/2402.14740) (RLOO)
* Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective," 2025 — [arXiv](https://arxiv.org/abs/2503.20783) (critique of GRPO's std and length normalization)
* ManiSkill3 — [paper](https://arxiv.org/abs/2410.00425), [docs](https://maniskill.readthedocs.io/)

---

## Next in this special course

* Next: [Lecture 05 — Actor-Critic, GAE, and Privileged Critics](Lecture-05.md)
* Previous: [Lecture 03 — Modern Imitation: Multimodality, Action Chunks, Flow Matching](Lecture-03.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
