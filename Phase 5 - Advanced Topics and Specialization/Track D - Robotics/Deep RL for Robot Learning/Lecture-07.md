# Lecture 07: Value-Based and Off-Policy RL: From Q-Learning to SAC

## Overview

Lectures 04-06 threw data away. A policy-gradient batch is used once (or for a few PPO epochs) and then discarded, because the gradient is only correct for the policy that collected it. This lecture keeps every try.

Here is the intuition. Picture a chess player who keeps a notebook. Next to every position they write how good each move turned out to be, and they keep revising those notes as they learn. To pick a move, they look up the position and play the move with the best note. The notes can come from any game they have ever played, including games from when they were much worse, because "how good is this move here" does not depend on who made the move back then. That notebook is the **Q-function**. Learning it from old experience is **off-policy RL**.

The price is stability. A Q-function is trained to match targets built from *itself*, and with neural networks nothing guarantees that this converges. Most of this lecture is the list of tricks that make it work in practice: target networks, replay buffers, double Q, clipped double Q, entropy bonuses. Together they produce **SAC**, the default off-policy method for continuous robot control.

There is also a hardware story, the reverse of PPO's: off-policy methods turn RL from **rollout-bound** into **learner-bound**. Whether that is a good trade depends on the regime from [Lecture 01](Lecture-01.md), and this lecture's main artifact is a table that settles it with your own numbers.

By the end you should be able to:

* run value iteration by hand on a gridworld and explain the γ-ripple pattern
* state the Bellman optimality operator and prove it is a γ-contraction in two lines
* explain why \(\max\) over noisy Q-estimates overestimates, and how double Q and clipped double Q fix it
* derive the SAC critic target, the reparameterized actor loss, and the temperature loss
* explain what the update-to-data (UTD) ratio costs in wall-clock and memory, and when SAC beats PPO and when it loses
* explain why a 50×7 action chunk rules out enumerating actions, and the two ways around that

---

## 1. Why it matters: reuse every try

| | On-policy (PG, PPO) | Off-policy (DQN, SAC) |
|---|---|---|
| Data used | Only from the current policy | Anything in the replay buffer: old policies, demos, scripted experts |
| Each sample used | Once, or for \(E\) epochs | Many times (UTD ratio × residence time in the buffer) |
| What is learned | Policy (+ a V baseline) | \(Q(s,a)\), with the policy derived from it |
| Stability | Good; the gradient is unbiased | Fragile; bootstrapped targets can diverge |
| Bottleneck | Rollouts | Learner |

For a robot, data is the expensive part whenever you leave fast GPU simulation: a real arm, a CPU-simulated LIBERO scene, or a VLA whose forward pass costs tens of milliseconds. In those settings, a method that reuses each transition many times is attractive. Off-policy learning is also the base of offline RL ([Lecture 10](Lecture-10.md)), which takes reuse to the limit with no new data at all.

---

## 2. Mental model: Q-values and the ripple

\(Q^*(s,a)\) answers "if I take move \(a\) here and play perfectly afterwards, how much reward do I get?" Once you have it, the policy is simple: \(\pi(s) = \arg\max_a Q^*(s,a)\). No policy network is needed, at least while actions can be enumerated.

### 2.1 Value iteration on a gridworld

Take a 4×5 grid. Reward is `+1` for stepping into the goal `G`, walls `#` block movement, and \(\gamma = 0.9\). Start every value at zero and repeatedly apply "value of a square = best over moves of (reward + γ × value of where you land)":

```text
 after 1 sweep                after 2 sweeps               converged (7 sweeps)
 0.00 0.00 0.00 1.00  G       0.00 0.00 0.90 1.00  G       0.73 0.81 0.90 1.00  G
 0.00  ##  0.00  ##  1.00     0.00  ##  0.00  ##  1.00     0.66  ##  0.81  ##  1.00
 0.00  ##  0.00 0.00 0.00     0.00  ##  0.00 0.00 0.90     0.59  ##  0.73 0.81 0.90
 0.00 0.00 0.00 0.00 0.00     0.00 0.00 0.00 0.00 0.00     0.53 0.59 0.66 0.73 0.81
```

The goal's value ripples outward one square per sweep. A square \(k\) steps from the goal ends at \(\gamma^{k-1}\): 1, 0.9, 0.81, 0.73, … Two things to notice. First, the largest change per sweep in this run is exactly 1, 0.9, 0.81, …, which is the contraction in the next subsection made visible. Second, **the ripple has to travel the whole path length**. A sparse reward at the end of a 300-step episode needs on the order of 300 sweeps' worth of backups to reach the start. This is why sparse long-horizon tasks are slow for TD learning, and why acting in chunks (fewer, bigger decisions) helps ([Lecture 13](Lecture-13.md)).

### 2.2 The Bellman optimality operator

Formally, value iteration repeatedly applies the operator \(\mathcal{T}\):

$$
(\mathcal{T}Q)(s,a) = r(s,a) + \gamma \, \mathbb{E}_{s' \sim p(\cdot \mid s,a)} \Big[ \max_{a'} Q(s',a') \Big]
$$

\(Q^*\) is its fixed point: \(\mathcal{T}Q^* = Q^*\). It is a **γ-contraction** in the max-norm. For any two Q-functions,

$$
\big| (\mathcal{T}Q_1)(s,a) - (\mathcal{T}Q_2)(s,a) \big| \le \gamma \, \mathbb{E}_{s'} \Big| \max_{a'} Q_1(s',a') - \max_{a'} Q_2(s',a') \Big| \le \gamma \, \| Q_1 - Q_2 \|_\infty
$$

The second step uses \(|\max_x f(x) - \max_x g(x)| \le \max_x |f(x) - g(x)|\). So every sweep shrinks the distance to \(Q^*\) by at least a factor of γ, and value iteration converges from any start. The guarantee holds for **tables**. Once a neural network represents Q, each update is "apply \(\mathcal{T}\), then project onto what the network can represent," and that combination is not a contraction in general.

---

## 3. From tables to networks: fitted Q and DQN

Pixels and continuous joint angles give far too many states for a table. Replace the table with a network \(Q_\phi\) and turn the backup into regression:

$$
y = r + \gamma \max_{a'} Q_{\bar\phi}(s', a'), \qquad
\mathcal{L}(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \big[ \ell\big(Q_\phi(s,a) - y\big) \big]
$$

Run this as an outer loop ("compute targets for the whole dataset, then fit") and you have **fitted Q iteration**. Run it online with one gradient step per environment step and you have **Q-learning**. Note what is absent: the policy that collected \((s,a,r,s')\) appears nowhere in the loss. The target uses the max over \(a'\), not the action the old policy actually took next. That is why any data works.

Exploration comes from acting **ε-greedily**: take a random action with probability ε (annealed from 1.0 toward ~0.05), and otherwise take the greedy one. The tricks that turned this into DQN, and why each one exists:

| Trick | Problem it fixes | How |
|---|---|---|
| **Replay buffer** | Consecutive transitions are highly correlated, and SGD assumes i.i.d. data | Store \(10^5\)-\(10^6\) transitions, sample uniform minibatches |
| **Target network** \(Q_{\bar\phi}\) | The target moves with every gradient step ("chasing your own tail") | Copy \(\phi \to \bar\phi\) every \(C\) steps, or Polyak-average \(\bar\phi \leftarrow \tau\phi + (1-\tau)\bar\phi\) with \(\tau \approx 0.005\) |
| **Huber loss** \(\ell\) | Rare huge TD errors produce exploding gradients | Quadratic near 0, linear in the tails (`F.smooth_l1_loss`) |
| **Double Q** | \(\max\) over noisy estimates is biased upward | Select \(a'\) with the online net, evaluate it with the target net |
| **Many seeds** | Off-policy runs vary a lot from seed to seed | Report ≥ 3-5 seeds, never a single curve |

### 3.1 Why max overestimates

Suppose every action's true value is 0 and the network's estimates are the truth plus independent unit noise. Then \(\max_a \hat Q(s,a)\) is the maximum of the noise, which is positive in expectation. For 10 actions it is about +1.5. In general,

$$
\mathbb{E}\big[\max_a \hat Q(s,a)\big] \;\ge\; \max_a \mathbb{E}\big[\hat Q(s,a)\big]
$$

because \(\max\) is convex (Jensen). The bias then **propagates**: the overestimate at \(s'\) becomes part of the target at \(s\), and bootstrapping compounds it along the trajectory. **Double Q-learning** separates the choice of action from its evaluation:

$$
y^{\text{double}} = r + \gamma \, Q_{\bar\phi}\big(s', \arg\max_{a'} Q_\phi(s',a')\big)
$$

The online net's noise picks the action, and the target net's mostly independent noise scores it, so the upward bias largely cancels.

**The deadly triad.** Sutton & Barto name three ingredients that together can make value learning diverge: **function approximation + bootstrapping + off-policy data**. Deep Q-learning has all three, and the tricks only make divergence rare enough to live with. Log the mean predicted Q next to the actual discounted return, and treat a Q that climbs past the largest possible return as a bug report.

---

## 4. Continuous actions: sample-and-rank, DDPG, TD3

\(\arg\max_a Q(s,a)\) is trivial with 2 actions and impossible to enumerate over \(\mathbb{R}^7\). There are two ways out.

**Option A: sample-and-rank.** Sample many candidate actions, score each with Q, and keep the best. QT-Opt (Kalashnikov et al. 2018) used the **cross-entropy method** for this: sample from a Gaussian, keep the top few, refit the Gaussian to them, repeat for a couple of iterations. It used no actor network and learned vision-based grasping on real robots. The cost is a batch of Q evaluations for every action, both when acting and inside every target computation.

**Option B: learn an actor that outputs the argmax.** **DDPG** (Lillicrap et al. 2015) trains a deterministic actor \(\mu_\theta(s)\) by gradient ascent on the critic:

$$
\nabla_\theta J \approx \mathbb{E}_{s \sim \mathcal{D}} \Big[ \nabla_a Q_\phi(s,a) \big|_{a = \mu_\theta(s)} \, \nabla_\theta \mu_\theta(s) \Big]
$$

This uses the critic's **action gradient**: "which way should I nudge the action to raise Q?" That is very different from the score-function gradient of Lecture 04, and it needs a differentiable path from the action to Q. DDPG is brittle, and **TD3** (Fujimoto et al. 2018) adds three fixes:

1. **Clipped double Q.** Train two critics and use \(\min(Q_{\bar\phi_1}, Q_{\bar\phi_2})\) in the target. The min is a deliberately pessimistic estimate that counters the overestimation, which the actor would otherwise actively exploit.
2. **Target policy smoothing.** Add clipped noise to \(a'\) in the target, so the critic cannot fit a narrow, spurious peak.
3. **Delayed actor updates.** Update the actor (and the targets) once every \(d = 2\) critic updates, so the actor climbs a critic that has had time to settle.

The general lesson: **the actor is an adversary of the critic's errors.** Any spot where Q is wrongly high, gradient ascent will find. Pessimism (min over critics) is the standard defense. It comes back, much stronger, in offline RL.

---

## 5. Max-entropy RL and SAC

**Intuition.** Reward the robot for doing the task *and* for staying a little unpredictable. A policy that keeps several good options alive explores better, is harder for critic errors to trap, and recovers more easily when the world surprises it. The objective adds an entropy bonus with temperature \(\alpha\):

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \Big[ \sum_t \gamma^t \big( r(s_t,a_t) + \alpha \, \mathcal{H}(\pi(\cdot \mid s_t)) \big) \Big]
$$

Lecture 08 derives this objective from first principles ("control as inference"). Here we use it.

### 5.1 Soft Bellman backup

The soft value of a state includes the entropy the policy will collect there:

$$
V(s') = \mathbb{E}_{a' \sim \pi(\cdot \mid s')} \big[ Q(s',a') - \alpha \log \pi(a' \mid s') \big]
$$

SAC (Haarnoja et al. 2018) trains two critics against a clipped-double-Q soft target, with \(a'\) sampled fresh from the *current* policy:

$$
y = r + \gamma (1 - d) \Big( \min_{i=1,2} Q_{\bar\phi_i}(s', a') - \alpha \log \pi_\theta(a' \mid s') \Big), \quad a' \sim \pi_\theta(\cdot \mid s')
$$

Here \(d\) is 1 only for true **termination**. On a time-limit truncation you still bootstrap (the same bug as in Lecture 05, with the same fix).

### 5.2 Reparameterized actor loss

The policy is a tanh-squashed Gaussian: \(u = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\), and \(a = \tanh(u)\). Because \(a\) is a differentiable function of \(\theta\) for fixed noise \(\epsilon\), the actor minimizes

$$
\mathcal{L}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, \, \epsilon} \Big[ \alpha \log \pi_\theta(a_\theta(s,\epsilon) \mid s) - \min_i Q_{\phi_i}(s, a_\theta(s,\epsilon)) \Big]
$$

and backpropagates through Q into the action. This is the **reparameterization gradient**. Lecture 08 shows it is the same trick that makes VAEs trainable, and compares its variance with REINFORCE. The tanh changes the density, so the log-prob needs a Jacobian correction:

$$
\log \pi(a \mid s) = \log \mathcal{N}(u; \mu, \sigma^2) - \sum_{j} \log\big(1 - \tanh^2(u_j)\big)
$$

### 5.3 Automatic temperature

A fixed \(\alpha\) is fragile because the reward scale changes from task to task. SAC instead targets an entropy level \(\bar{\mathcal{H}}\) (commonly \(-\dim(\mathcal{A})\), i.e. −7 for our arm) and adapts \(\alpha\):

$$
\mathcal{L}(\alpha) = \mathbb{E}_{a \sim \pi} \big[ -\alpha \big( \log \pi(a \mid s) + \bar{\mathcal{H}} \big) \big]
$$

If the policy is less random than the target, \(\alpha\) rises, and the reverse. Optimize \(\log\alpha\) so that \(\alpha\) stays positive.

### 5.4 Update-to-data ratio and high-UTD methods

The **UTD ratio** is the number of gradient steps per environment transition. A higher UTD means more reuse and better sample efficiency, up to the point where the critic overfits early data and stops adapting. Several lines of work push UTD up safely:

**REDQ** (Chen et al. 2021) uses an ensemble of critics and takes the target as the min over a random subset of two. **DroQ** (Hiraoka et al. 2021) gets a similar effect from dropout + layer norm in the critics, at lower compute. **Resets** (Nikishin et al. 2022, "primacy bias") periodically re-initialize part or all of the networks while keeping the buffer, so a high-UTD agent does not lock in what it learned from its first few transitions.

Every one of these spends **more learner compute per env step**, which is the subject of section 7.

---

## 6. Robot connection: a 50×7 chunk cannot be enumerated

π0.5 emits a chunk of ~50 seven-dimensional actions, which is 350 continuous numbers. DQN's \(\max_{a'}\) is impossible, and even CEM over \(\mathbb{R}^{350}\) is hopeless. Two routes stay open:

* **Keep an actor.** Train the policy against a critic, as DDPG/SAC do. For a flow-matching VLA, "backprop Q through the action" means backprop through ~10 integration steps of the action expert. That is possible, but expensive and memory-heavy (Lectures 10-11 cover cheaper variants).
* **Sample-and-rank with a generative proposal.** The VLA is already a very good proposal distribution. Sample \(K\) candidate chunks from π0.5, score them with a critic \(Q(s, a_{t:t+H})\), and execute the best. This is V-GPS (Nakamoto et al. 2024), which re-ranks a generalist policy's actions using a value function learned offline, without touching the policy's weights. Q-chunking (Li et al. 2025) shows the critic side can be learned directly over chunks, with a bonus: backups over a chunk behave like unbiased \(n\)-step backups.

Sample-and-rank costs \(K\) policy samples plus \(K\) critic evaluations per decision, which moves the cost back into the regime-C inference bottleneck. And if the improved behavior has to live *in the weights* (a competition that accepts only a checkpoint, for example), the ranked choices must be distilled back into the policy (Lecture 10).

---

## 7. Hardware view: off-policy flips the bottleneck

Write the off-policy iteration the same way as Lecture 01's \(t_{\text{iter}}\). Each vector step collects \(N\) transitions and then performs \(G = \text{UTD} \cdot N\) gradient steps on minibatches of size \(B\):

$$
t_{\text{iter}} \approx t_{\text{env}}(N) + t_{\text{policy}}(N) + \text{UTD} \cdot N \cdot t_{\text{update}}(B)
$$

which caps the effective data rate at

$$
\text{env-steps/s} \;\lesssim\; \frac{\text{learner-updates/s}}{\text{UTD}}
$$

One SAC update (two critics + targets + actor + α, each forward and backward) with small MLPs is **launch-bound**. Expect on the order of hundreds to a few thousand updates/s, nearly independent of \(B\) until \(B\) gets large. Measure your own. At UTD = 1, that caps SAC at a few thousand env-steps/s, while the same GPU simulates 10⁵-10⁶ steps/s. **In regime B, off-policy RL leaves the simulator idle almost all the time.** This is why GPU-sim SAC baselines run few environments and UTD below 1, and why PPO with thousands of envs usually wins **wall-clock** in regime B even though it needs many more samples.

In regime C the arithmetic flips. One π0.5 chunk costs tens of milliseconds, so the policy term dominates, and a learner update on a small critic is cheap by comparison. Now every transition is expensive and reusing it 10-20 times is nearly free. **Off-policy wins in regime C** (and on real robots), as long as the critic and actor updates themselves do not involve the VLA's backward pass.

### 7.1 Replay buffer memory

| Stored per transition | Bytes (approx.) | 1 M transitions |
|---|---|---|
| PickCube state obs (~40-50 floats) ×2 (obs, next_obs) + action + reward + done, fp32 | ~400 B | ~0.4 GB, fits on GPU |
| One 128×128 RGB frame, uint8, obs + next_obs | ~98 KB | ~98 GB, host RAM only |
| Same frame stored as fp32 | ~393 KB | ~393 GB, a classic mistake |
| Two 224×224 RGB cameras (LIBERO-style), uint8, obs only | ~301 KB | ~301 GB |

Rules that follow:

* **Store pixels as uint8** and convert to float on the GPU, inside the sampled minibatch.
* **Don't store next_obs twice.** Keep frames in a ring buffer and store indices. That halves memory, at the cost of care at episode boundaries.
* **State buffers live on the GPU** next to the simulator (no PCIe round trip; sampling is one `torch.randint` + gather). **Pixel buffers live in pinned host memory** with `non_blocking=True` minibatch copies. Profile the copy; at high UTD it can become the bottleneck.
* For VLAs, store **embeddings** rather than raw images only if the encoder is frozen; otherwise the stored features go stale.

---

## 8. Build it

All code goes in `offpolicy/`.

### Lab 7a — Value iteration and the γ-ripple

```python
# offpolicy/value_iteration.py
import numpy as np

GRID = ["....G",
        ".#.#.",
        ".#...",
        "....."]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def backup(grid, V, r, c, gamma):
    H, W, best = len(grid), len(grid[0]), -np.inf
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W) or grid[nr][nc] == "#":
            nr, nc = r, c                                   # bump into wall: stay put
        best = max(best, 1.0 if grid[nr][nc] == "G" else gamma * V[nr, nc])
    return best

def value_iteration(grid, gamma=0.9, max_iters=100):
    V = np.zeros((len(grid), len(grid[0])))
    for it in range(1, max_iters + 1):
        V_new = np.array([[0.0 if ch in "#G" else backup(grid, V, r, c, gamma)   # goal is terminal
                           for c, ch in enumerate(row)] for r, row in enumerate(grid)])
        delta, V = np.abs(V_new - V).max(), V_new
        print(f"sweep {it}: max change {delta:.4f}")
        if delta == 0:
            break
    return V

print(np.round(value_iteration(GRID), 2))
```

Print \(V\) after sweeps 1, 2, 3 and compare against section 2.1. Then make the reward stochastic (the goal pays `+1` with probability 0.8) and add a 10% "slip" to a random neighbor. Confirm that the max change still shrinks at least as fast as \(\gamma^k\).

### Lab 7b — DQN on CartPole: target net and double Q

```python
# offpolicy/dqn_cartpole.py
import argparse, copy, random, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, gymnasium as gym

p = argparse.ArgumentParser()
for flag, default in [("--target", 1), ("--double", 1), ("--seed", 0), ("--steps", 100_000)]:
    p.add_argument(flag, type=int, default=default)
args = p.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

env = gym.make("CartPole-v1")
q = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
q_targ = copy.deepcopy(q)
opt = torch.optim.Adam(q.parameters(), lr=2.5e-4)
buf = collections.deque(maxlen=50_000)
gamma, batch, sync_every = 0.99, 128, 500
T = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32)

obs, _ = env.reset(seed=args.seed); ep_ret, returns = 0.0, []
for step in range(args.steps):
    eps = max(0.05, 1.0 - step / (0.5 * args.steps))
    if random.random() < eps:
        a = env.action_space.sample()
    else:
        with torch.no_grad():
            a = q(T(obs)).argmax().item()
    nobs, r, term, trunc, _ = env.step(a)
    buf.append((obs, a, r, nobs, float(term)))     # store termination only: bootstrap through time limits
    obs, ep_ret = nobs, ep_ret + r
    if term or trunc:
        returns.append(ep_ret); ep_ret = 0.0
        obs, _ = env.reset()

    if step < 1_000:
        continue
    o, a_, r_, no, d = map(T, zip(*random.sample(buf, batch)))
    with torch.no_grad():
        net = q_targ if args.target else q
        if args.double:                              # online net selects, target net evaluates
            na = q(no).argmax(1, keepdim=True)
            nq = net(no).gather(1, na).squeeze(1)
        else:
            nq = net(no).max(1).values
        y = r_ + gamma * (1 - d) * nq
    qa = q(o).gather(1, a_.long().unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(qa, y)                   # Huber
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
    if args.target and step % sync_every == 0:
        q_targ.load_state_dict(q.state_dict())
    if step % 5_000 == 0:
        print(f"step {step:6d}  ret(last20) {np.mean(returns[-20:]):6.1f}  mean Q {qa.mean():7.2f}")
```

Run the 2×2 grid `--target {0,1} --double {0,1}` over 5 seeds. (With `--target 0 --double 1`, selection and evaluation use the same net, so it collapses to plain DQN. Note that in the writeup.) CartPole pays +1 per step, so with γ = 0.99 no true Q-value can exceed \(1/(1-\gamma) = 100\). **Plot mean predicted Q against that ceiling.** Overestimation is now something you can see on a chart, not just a claim.

### Lab 7c — SAC on PickCube with a UTD sweep

Compact SAC for GPU-parallel ManiSkill3. Diff it against ManiSkill's own `examples/baselines/sac/sac.py`, which uses the same wrapper and the same `final_observation` handling.

```python
# offpolicy/sac_pickcube.py
import argparse, copy, math, time
import torch, torch.nn as nn, torch.nn.functional as F, gymnasium as gym
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

p = argparse.ArgumentParser()
for flag, typ, default in [("--num-envs", int, 32), ("--utd", float, 1.0),   # utd = grad steps / transition
                           ("--total-steps", int, 1_000_000), ("--seed", int, 0)]:
    p.add_argument(flag, type=typ, default=default)
args = p.parse_args()
torch.manual_seed(args.seed); dev = torch.device("cuda")

env = gym.make("PickCube-v1", num_envs=args.num_envs, obs_mode="state",
               control_mode="pd_ee_delta_pose", reward_mode="normalized_dense")
env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)
od, ad = env.single_observation_space.shape[0], env.single_action_space.shape[0]

def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)).to(dev)

class Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.net = mlp(od, 2 * ad)
    def forward(self, o):
        mu, log_std = self.net(o).chunk(2, -1)
        log_std = log_std.clamp(-5, 2); std = log_std.exp()
        u = mu + std * torch.randn_like(mu)                     # reparameterization
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp -= (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)   # tanh Jacobian, stable form
        return torch.tanh(u), logp

actor = Actor()
q1, q2 = mlp(od + ad, 1), mlp(od + ad, 1)
q1t, q2t = copy.deepcopy(q1), copy.deepcopy(q2)
log_alpha = torch.zeros(1, device=dev, requires_grad=True)
opt_q = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=3e-4)
opt_pi = torch.optim.Adam(actor.parameters(), lr=3e-4)
opt_a = torch.optim.Adam([log_alpha], lr=3e-4)
gamma, tau, B, target_ent = 0.8, 0.005, 1024, -float(ad)    # short 50-step episodes: small gamma

cap = 1_000_000                                                # GPU-resident replay buffer
buf = {k: torch.zeros(cap, n, device=dev) for k, n in
       [("o", od), ("a", ad), ("r", 1), ("no", od), ("d", 1)]}
ptr, size = 0, 0

def Q(net, o, a): return net(torch.cat([o, a], -1)).squeeze(-1)

def update():
    i = torch.randint(0, size, (B,), device=dev)
    o, a, r, no, d = (buf[k][i] for k in ("o", "a", "r", "no", "d"))
    r, d = r.squeeze(-1), d.squeeze(-1)
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        na, nlogp = actor(no)
        y = r + gamma * (1 - d) * (torch.min(Q(q1t, no, na), Q(q2t, no, na)) - alpha * nlogp)
    lq = F.mse_loss(Q(q1, o, a), y) + F.mse_loss(Q(q2, o, a), y)
    opt_q.zero_grad(); lq.backward(); opt_q.step()
    pa, logp = actor(o)
    lpi = (alpha * logp - torch.min(Q(q1, o, pa), Q(q2, o, pa))).mean()
    opt_pi.zero_grad(); lpi.backward(); opt_pi.step()
    la = -(log_alpha.exp() * (logp.detach() + target_ent)).mean()
    opt_a.zero_grad(); la.backward(); opt_a.step()
    with torch.no_grad():
        for net, tgt in ((q1, q1t), (q2, q2t)):
            for p_, pt in zip(net.parameters(), tgt.parameters()):
                pt.lerp_(p_, tau)                              # Polyak average

obs, _ = env.reset(seed=args.seed)
step, debt, t_roll, t_learn, t0 = 0, 0.0, 0.0, 0.0, time.perf_counter()
while step < args.total_steps:
    ts = time.perf_counter()
    with torch.no_grad():
        act = actor(obs)[0] if step > 5_000 else 2 * torch.rand(args.num_envs, ad, device=dev) - 1
    nobs, rew, term, trunc, info = env.step(act)
    real_next = nobs.clone()
    if "final_info" in info:                                   # auto-reset happened for these envs
        done = info["_final_info"]
        real_next[done] = info["final_observation"][done]
        succ = info["final_info"]["episode"]["success_once"][done].float().mean().item()
        print(f"step {step:8d}  success {succ:.2f}  wall {time.perf_counter() - t0:7.1f}s  "
              f"roll {t_roll:6.1f}s  learn {t_learn:6.1f}s")
    idx = (torch.arange(args.num_envs, device=dev) + ptr) % cap
    for k, v in (("o", obs), ("a", act), ("r", rew[:, None]), ("no", real_next), ("d", term[:, None])):
        buf[k][idx] = v.float()
    ptr, size = (ptr + args.num_envs) % cap, min(size + args.num_envs, cap)
    obs, step = nobs, step + args.num_envs
    torch.cuda.synchronize(); t_roll += time.perf_counter() - ts

    if step > 5_000:
        ts = time.perf_counter()
        debt += args.utd * args.num_envs                       # gradient steps owed this vector step
        while debt >= 1:
            update(); debt -= 1
        torch.cuda.synchronize(); t_learn += time.perf_counter() - ts
```

Sweep `--utd {1, 4, 16}` × 3 seeds, keeping `--num-envs 32`. For every run, record samples-to-80%-success, wall-clock-to-80%-success, and the rollout/learn split. Then rerun your Lecture 06 PPO on the same task, control mode, and reward mode, and put everything in one table (section 11). If you have time, add `--num-envs 1024 --utd 1` and watch the learn term swamp everything. That run is the regime-B argument of section 7 in a single number.

---

## 9. Use it in the real stack

* **Reference code:** [CleanRL](https://github.com/vwxyzjn/cleanrl) `dqn.py`, `sac_continuous_action.py`, `td3_continuous_action.py`, `ddpg_continuous_action.py`. ManiSkill3's `examples/baselines/sac/` (state and RGB-D, with `utd` and `training_freq` arguments) and `examples/baselines/rlpd/` (Lecture 10). [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) as a known-good baseline when your numbers disagree.
* **Real robots:** QT-Opt (grasping from pixels with a CEM argmax) is the canonical large-scale example. Real-world RL systems since then are mostly SAC/RLPD-family methods with a demonstration-seeded buffer, for exactly the regime-C reason in section 7.
* **VLA scale:** value-guided re-ranking (V-GPS) and chunked critics (Q-chunking) keep the expensive policy frozen and put the learning into a small critic.

---

## 10. Measure it

| Metric | How | What it tells you |
|---|---|---|
| **Mean predicted Q vs realized discounted return** | Log both on the same axis | Overestimation and divergence. Q above the max possible return is a bug. |
| **learner-updates/s** | Updates / `t_learn` | With UTD, caps the data rate (section 7) |
| **Rollout vs learn split** | `t_roll`, `t_learn` with `cuda.synchronize()` | Which side is the bottleneck. For SAC in regime B, expect learn ≫ roll. |
| **samples-to-80%** | Env steps at first eval ≥ 80% success | Sample efficiency; should improve with UTD up to a point |
| **wall-clock-to-80%** | Seconds to the same threshold | What you pay. Often gets *worse* with UTD in regime B. |
| **Replay memory** | Bytes per transition × capacity, plus `max_memory_allocated` | Whether a pixel buffer fits on the GPU at all |
| **α and policy entropy** | Log both | Temperature tuning working? An entropy collapse often comes before a performance collapse. |
| **Seed spread** | Min/median/max over ≥ 3 seeds | Off-policy variance. Never report a single run. |

---

## 11. Ship it

Commit `offpolicy/` containing:

* `value_iteration.py` + `ripple.txt` (sweeps 1-3 and converged values)
* `dqn_cartpole.py` + `dqn_overestimation.png` (mean Q vs the 100 ceiling, 4 variants × 5 seeds)
* `sac_pickcube.py` + `sac_utd_curves.png` (success vs env steps *and* vs wall-clock, UTD ∈ {1, 4, 16})
* `PPO_VS_SAC.md`, the key hardware lesson, built around this table:

| Method | num_envs | UTD / epochs | Env steps to 80% | Wall-clock to 80% | learner-updates/s | Rollout : learn time | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| PPO (L06) | 1024-4096 | E epochs | | | | | |
| SAC | 32 | 1 | | | | | |
| SAC | 32 | 4 | | | | | |
| SAC | 32 | 16 | | | | | |

  End with two sentences: which method wins on samples, which on wall-clock, and a prediction, using the formula in section 7 with a π0.5-sized \(t_{\text{policy}}\), of which would win in regime C.

---

## Exit criteria

You can move on when you can:

* reproduce the γ-ripple values by hand and explain why the max change per sweep is \(\gamma^k\)
* prove that \(\mathcal{T}\) is a γ-contraction, and say exactly why the proof breaks with a neural network
* explain overestimation with the Jensen argument and show it in your DQN plot
* write the SAC critic target, actor loss, and temperature loss from memory, including the tanh correction
* use your own table to explain why PPO wins wall-clock in regime B and why off-policy reuse wins in regime C
* explain why π0.5's action space rules out an argmax, and describe sample-and-rank with a critic

---

## Self-check

1. In the deterministic gridworld, the values stop changing after 7 sweeps (sweep 8 reports zero change) for both γ = 0.9 and γ = 0.99. Why does γ not matter here? In the stochastic "slip" version from Lab 7a, roughly how many sweeps does the contraction bound say you need before the max change drops below 0.01, for each γ? Relate the answer to the effective horizon \(1/(1-\gamma)\).
2. Your DQN's mean predicted Q on CartPole climbs to 250 while returns stay around 150. Name the two most likely causes, and the two flags from Lab 7b you would check first.
3. Your SAC run at UTD = 16 needs 4× fewer env steps than UTD = 1 to reach 80%, but takes 3× longer in wall-clock. Use the \(t_{\text{iter}}\) formula to explain why. What would have to change about the environment for UTD = 16 to win?
4. A teammate stores 128×128 RGB observations as `float32` tensors on the GPU, with both `obs` and `next_obs`, and runs out of memory at 100 k transitions on a 24 GB card. Estimate what they were trying to allocate, and give two changes that make 1 M transitions fit somewhere.
5. You want to improve π0.5 on LIBERO with a critic without fine-tuning the VLA. Describe the sample-and-rank procedure, its per-decision inference cost in terms of \(K\), and why a competition that accepts only model weights forces one more step.

---

## References

* Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., ch. 4 (value iteration), 6 (Q-learning), 11.3 (the deadly triad) — [online](http://incompleteideas.net/book/the-book-2nd.html)
* Mnih et al., "Playing Atari with Deep Reinforcement Learning," 2013 — [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
* van Hasselt, Guez, Silver, "Deep Reinforcement Learning with Double Q-learning," 2015 — [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
* Lillicrap et al., "Continuous control with deep reinforcement learning" (DDPG), 2015 — [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
* Fujimoto, van Hoof, Meger, "Addressing Function Approximation Error in Actor-Critic Methods" (TD3), 2018 — [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
* Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor," 2018 — [arXiv:1801.01290](https://arxiv.org/abs/1801.01290); "Soft Actor-Critic Algorithms and Applications" (automatic temperature) — [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)
* Kalashnikov et al., "QT-Opt: Scalable Deep RL for Vision-Based Robotic Manipulation," 2018 — [arXiv:1806.10293](https://arxiv.org/abs/1806.10293)
* Chen et al., "Randomized Ensembled Double Q-Learning" (REDQ), 2021 — [arXiv:2101.05982](https://arxiv.org/abs/2101.05982)
* Hiraoka et al., "Dropout Q-Functions for Doubly Efficient RL" (DroQ), 2021 — [arXiv:2110.02034](https://arxiv.org/abs/2110.02034)
* Nikishin et al., "The Primacy Bias in Deep RL," 2022 — [arXiv:2205.07802](https://arxiv.org/abs/2205.07802)
* Nakamoto et al., "Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance" (V-GPS), 2024 — [arXiv:2410.13816](https://arxiv.org/abs/2410.13816)
* Li, Zhou, Levine, "Reinforcement Learning with Action Chunking" (Q-chunking), 2025 — [arXiv:2507.07969](https://arxiv.org/abs/2507.07969)

---

## Next in this special course

* Next: [Lecture 08 — The Probabilistic Toolkit: ELBO, VAEs, Control as Inference, Inverse RL](Lecture-08.md)
* Previous: [Lecture 06 — PPO, Trust Regions, and the KL Leash](Lecture-06.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
