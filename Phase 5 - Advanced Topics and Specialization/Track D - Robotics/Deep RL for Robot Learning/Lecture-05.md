# Lecture 05: Actor-Critic, GAE, and Privileged Critics

## Overview

A football team can learn from the final score alone: win, and everything you did was good; lose, and everything was bad. That signal is honest, but it reaches you late and is hopelessly blurry. A good **coach** changes this. The coach watches each play and says "that pass put us in a much better position than I expected," long before the final whistle. The coach can be wrong, but their judgment is immediate and steady.

In RL, the coach is the **critic**: a network \(V_\phi(s)\) that predicts the expected future score from state \(s\). The player is the **actor**, the policy \(\pi_\theta\). After each move, the critic compares where things stand now with where they stood before:

```text
  advantage  ≈  (reward now + coach's value of the new state)  −  coach's value of the old state
```

Positive means the move turned out better than the coach expected, so make it more likely. Lecture 04's Monte Carlo baselines needed many restarts from the *same* state to estimate this. A learned critic generalizes across states, so it can judge every step of every episode.

The cost is bias. The real final score is honest but noisy; the coach is calm but can be wrong. **Generalized Advantage Estimation (GAE)** puts a dial \(\lambda\) between the two. This lecture derives that dial, shows how the discount \(\gamma\) sets how far ahead the coach looks, fixes the most common bug in every actor-critic codebase (truncation vs termination), and then gives the coach an unfair advantage. In simulation, the critic can see exact object poses while the actor sees only pixels.

By the end you should be able to:

* write the TD error \(\delta_t\) and show that its expectation is the advantage when \(V = V^\pi\)
* derive GAE as an exponentially weighted average of n-step advantages and state what \(\lambda = 0\) and \(\lambda = 1\) recover
* choose \(\gamma\) from the task's horizon, and explain why a small \(\gamma\) can make a sparse success reward invisible
* implement bootstrapping that treats **truncation** and **termination** differently, and catch the bug when it is wrong
* build an **asymmetric** actor-critic (pixel actor, privileged-state critic) and measure what it saves in compute and memory

---

## 1. Why it matters: a baseline for every state

Lecture 04 ended with two problems:

| Problem | Monte Carlo fix (L04) | Learned critic (this lecture) |
|---|---|---|
| Baseline needs \(V(s_t)\) for every \(t\), not only \(s_0\) | Restart from \(s_t\) many times: impossible on a real robot, costly in sim | \(V_\phi(s_t)\) from one forward pass |
| Return is a sum of many random rewards | Big batches | Replace most of the sum with \(V_\phi\): lower variance, some bias |
| Group baseline wastes rollouts when all tries agree | Pick better initial states (L12) | Critic still gives non-zero advantages mid-episode, because it compares states |

With a sparse binary success reward and \(\gamma \approx 1\), \(V^\pi(s)\) is **the probability of eventually succeeding from \(s\)** (Lecture 01). That is something you can sanity-check by eye: it should rise when the gripper closes on the bowl and fall when the bowl slips.

> **Robot connection.** For π0.5 on a LIBERO layout, a critic that outputs "chance of success from here" turns one sparse 0/1 per episode into a per-chunk signal: this chunk raised the success estimate from 0.4 to 0.7. In simulation the critic can also **peek**. It can take exact object poses, contact flags, and the goal predicate as input, even though π0.5 itself only ever sees cameras, the instruction, and proprioception. The critic is discarded after training, so it is allowed to cheat. Section 6 makes this precise, and it is step 4 of the capstone game plan.

---

## 2. Mental model: the actor-critic loop

```text
         ┌────────── obs (pixels + proprio) ─────────┐
         ▼                                            │
   ┌───────────┐  a_t   ┌────────────┐  s_{t+1}, r_t  │
   │  ACTOR    │ ─────► │ SIMULATOR  │ ───────────────┤
   │  πθ(a|o)  │        └────────────┘                │
   └─────▲─────┘                                      ▼
         │  advantage Â_t                       ┌───────────┐
         └───────────────────────────────────── │  CRITIC   │  regress Vφ(s_t)
            "better or worse than I expected?"  │  Vφ(s)    │  toward return targets
                                                └───────────┘
```

Two losses, trained together:

* **Critic:** regression. Make \(V_\phi(s_t)\) match a target built from observed rewards (and its own predictions; see Section 3).
* **Actor:** Lecture 04's policy gradient with the weight replaced by \(\hat A_t\): \(\;\nabla_\theta J \approx \frac{1}{N}\sum_{i,t} \nabla_\theta \log \pi_\theta(a_t^i \mid o_t^i)\, \hat A_t^i\).

Everything interesting is in **how \(\hat A_t\) is built**.

---

## 3. Value targets: Monte Carlo, TD, and everything between

### 3.1 Two ways to train the coach

| Target for \(V_\phi(s_t)\) | Formula | Bias | Variance |
|---|---|---|---|
| **Monte Carlo** | \(\sum_{t' \ge t} \gamma^{t'-t} r_{t'}\) | None, given enough data | High: sums every future random reward |
| **TD(0)** | \(r_t + \gamma V_\phi(s_{t+1})\) | Whatever error \(V_\phi\) has | Low: one random reward plus a smooth estimate |
| **n-step** | \(\sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V_\phi(s_{t+n})\) | Shrinks as \(n\) grows | Grows with \(n\) |

Using your own prediction inside your own target is **bootstrapping**. It propagates information backward one step per update without waiting for the episode to finish. It is also the origin of every instability you will meet in Lecture 07.

### 3.2 The TD error is an advantage estimate

Define the **TD error**:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

If \(V = V^\pi\) exactly, then conditioned on \((s_t, a_t)\):

$$
\mathbb{E}[\delta_t \mid s_t, a_t] = r_t + \gamma\, \mathbb{E}_{s_{t+1}}\big[V^\pi(s_{t+1})\big] - V^\pi(s_t) = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)
$$

So \(\delta_t\) is a one-sample estimate of the advantage, with only one step of randomness in it. If \(V \ne V^\pi\), it is biased by exactly the critic's error. That is the trade.

### 3.3 n-step advantages telescope

The n-step advantage estimate is a sum of TD errors. The intermediate \(V\) terms cancel:

$$
\hat A_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n}) - V(s_t) = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}
$$

\(n = 1\) is TD(0). \(n \to \infty\) is the Monte Carlo return minus \(V(s_t)\), which is Lecture 04's REINFORCE with a learned baseline.

---

## 4. GAE: the λ dial

Rather than picking one \(n\), average all of them with geometric weights \((1-\lambda)\lambda^{n-1}\):

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat A_t^{(n)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}
$$

The second equality comes from substituting the telescoped form and swapping sums: \(\delta_{t+l}\) appears in every \(\hat A^{(n)}\) with \(n > l\), with total weight \((1-\lambda)\sum_{n>l}\lambda^{n-1}\gamma^l = (\gamma\lambda)^l\). It gives a one-line backward recursion, which is what every implementation computes:

$$
\hat A_t = \delta_t + \gamma \lambda\, (1 - d_t)\, \hat A_{t+1}
$$

where \(d_t = 1\) if the episode ended at step \(t\). The critic's regression target is then \(\hat R_t = \hat A_t + V(s_t)\), the **λ-return**.

| \(\lambda\) | \(\hat A_t\) becomes | Trust | Typical use |
|---|---|---|---|
| 0 | \(\delta_t\) (TD(0)) | The coach completely | Low variance; biased while the critic is bad |
| 0.9-0.97 | Mostly short-range \(\delta\)'s, decaying | Mostly the coach, checked against real rewards | PPO defaults (0.95 is common) |
| 1 | \(\sum_l \gamma^l r_{t+l} - V(s_t)\) | Real returns; coach is only a baseline | Unbiased; Lecture 04's variance |

### 4.1 The discount γ is a horizon knob

\(\sum_t \gamma^t = 1/(1-\gamma)\) is the **effective horizon**: roughly how many steps ahead the agent cares about. GAE's credit decays faster, with horizon \(1/(1-\gamma\lambda)\).

| \(\gamma\) | Effective horizon | Weight on a reward 50 steps away (\(\gamma^{50}\)) |
|---|---|---|
| 0.8 | 5 steps | ~1e-5 |
| 0.95 | 20 steps | ~0.08 |
| 0.99 | 100 steps | ~0.61 |
| 0.999 | 1000 steps | ~0.95 |

This table matters for robot tasks. ManiSkill's PPO baseline script defaults to \(\gamma = 0.8\), \(\lambda = 0.9\) for its 50-step tasks. That works because it trains on a **dense shaped reward**, which pays out every step. With a **sparse success reward** at step 50, \(\gamma = 0.8\) means the success is worth ~1e-5 from the start of the episode, and the critic sees essentially nothing. The rule of thumb: **with sparse reward, \(1/(1-\gamma)\) must be at least the number of decisions between the action and the success.**

Action chunks change the count. openpi's LIBERO setup (`pi05_libero`) predicts 10-action chunks and executes 5 before re-planning, so a 300-step LIBERO episode is 60 decisions. A \(\gamma\) applied **per chunk** therefore covers five times more wall time than the same \(\gamma\) applied per control step. Decide whether your critic runs per step or per chunk, and set \(\gamma\) accordingly.

---

## 5. The bugs that live in every actor-critic codebase

### 5.1 Truncation is not termination

An episode can end for two very different reasons:

| | **Terminated** | **Truncated** |
|---|---|---|
| Cause | The MDP really ended: success, failure, bowl fell off the table | A time limit you imposed for convenience (50 steps) |
| True value of the next state | 0 (nothing more will happen) | \(V(s_{T})\): the world would have continued |
| Bootstrap target | \(r_t\) | \(r_t + \gamma V(s_{t+1})\) |
| Gymnasium flag | `terminated` | `truncated` |

Treating truncation as termination tells the critic that states near the time limit are worth 0. It then learns a value that depends on time, which it cannot see, and the advantage estimates near the end of every episode become wrong. Pardo et al. (2017) analyze this. Gymnasium split `done` into `terminated` and `truncated` precisely so code can tell them apart.

There are **two masks**, and conflating them is the bug:

* `terminated` decides **whether to bootstrap** with \(V(s_{t+1})\)
* `terminated or truncated` decides **where the GAE trace stops** (the next row of the buffer belongs to a new episode)

A third trap comes from auto-resetting vector envs. After a done, the returned `obs` is the **first observation of the next episode**, not \(s_{t+1}\). ManiSkill's `ManiSkillVectorEnv` (like Gymnasium's pre-1.0 vector envs) puts the real final observation in `info["final_observation"]`. You must evaluate \(V\) on *that* for truncated envs. Gymnasium 1.x vector envs default to next-step autoreset instead: the done step returns the real final observation, and the following `step()` call performs the reset and its transition must be masked out. Check which mode your vector env uses.

### 5.2 Other details that move the needle

* **Advantage normalization.** Normalize \(\hat A\) to zero mean and unit std per batch. This is a cheap, reliable learning-rate stabilizer. Note that normalizing re-introduces a batch baseline.
* **Explained variance** of the critic, \(\mathrm{EV} = 1 - \mathrm{Var}[\hat R - V]/\mathrm{Var}[\hat R]\), is the single most useful critic diagnostic. Near 1 means the critic predicts returns well. Near 0 means it is no better than a constant. Negative means it is actively harmful, and you should check the bootstrapping masks first.
* **Shared vs separate trunks.** A shared trunk saves one forward pass, but the value loss and policy loss pull the features in different directions and need a relative weight (`vf_coef`). CleanRL's continuous PPO and ManiSkill's state PPO use **separate** networks. For pixel policies, sharing the image encoder is common because the encoder is the expensive part.
* **Value-loss scale.** With dense rewards, returns can be in the tens or hundreds. Normalize rewards or returns, or the value loss will swamp a shared trunk.

---

## 6. Asymmetric actor-critic: give the coach the answers

The policy must run on the robot, so it can only use deployable inputs: cameras, instruction, and proprioception. The critic only exists during training, **in simulation**, where the full state is available. Pinto et al. (2017) proposed feeding the critic privileged state while the actor sees images. OpenAI's in-hand cube manipulation work used the same split for its value network.

Why it is legitimate: Lecture 04 showed that any baseline \(b(s_t)\) that does not depend on \(a_t\) leaves the gradient unbiased:

$$
\mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid o_t)}\big[\nabla_\theta \log \pi_\theta(a_t \mid o_t)\, b(s_t)\big] = b(s_t) \cdot 0 = 0
$$

The action distribution depends only on what the actor sees, so the inner expectation over \(a_t\) is still zero however much the baseline knows about \(s_t\). A better-informed baseline is a **lower-variance** baseline.

| | Actor | Critic |
|---|---|---|
| Input | RGB from the task camera, proprio, goal | Full sim state: cube pose and velocity, goal, joint state |
| Size | CNN + MLP, ~0.5-1 M params | MLP 3×256, ~0.15-0.2 M params |
| Needed at deployment | Yes | No |
| Learning problem | Hard (perception + control) | Easy (low-dimensional regression) |

**The subtlety.** Once the critic is used for **bootstrapping** (GAE with \(\lambda < 1\)), not only as a baseline, it plays a different role. The actor's true value under partial observability depends on what it *knows*, which is its observation history, not the true state. Baisero & Amato (2022) show that a critic conditioned on state alone can bias the gradient in genuinely partially observable tasks, and propose conditioning on both history and state. In practice: for manipulation where the cameras see almost everything that matters, state critics work well. For tasks with hidden information (a closed drawer, an occluded object), give the critic the actor's features *plus* the privileged state, and compare.

> **Robot connection.** For π0.5 on LIBERO, the privileged critic can read every object's pose from the simulator: the bowl, the plate, and the distractors. It can also read whether the success predicate is close to true. Learning "probability of success" from 30 numbers is far easier than learning it from two camera images. This is why a privileged critic is the default first critic for VLA RL in simulation. It also sidesteps the hard question of what network to use for a critic at VLA scale (Section 7).

---

## 7. The hardware view: what the critic costs

A critic adds **parameters, optimizer state, a forward pass per step, and a backward pass per update**. How much depends on what it looks at:

| Critic design | Extra params | Extra rollout compute | Extra learner memory | When |
|---|---|---|---|---|
| Small MLP on privileged state | ~0.1-1 M | negligible | negligible | Sim training, any policy size: **the default** |
| Separate pixel CNN (same as actor) | ≈ actor encoder | ≈ +1 actor forward per step | ≈ +actor activations and Adam state | When no privileged state exists |
| Value head on a shared trunk | tiny | ~0 (same forward) | ~0 extra weights, but value gradients flow into the trunk | Small pixel policies; risky for a pretrained VLA |
| Separate critic the size of the VLA | ≈ VLA param count | +1 VLA forward per step | ≈ +16 bytes/param with Adam in mixed precision | Rarely worth it |

Do the arithmetic for the last row. Adam in mixed precision stores roughly 16 bytes per trainable parameter: bf16 weights and gradients, plus fp32 master weights and two fp32 moments. A separate 3B-parameter critic therefore adds on the order of 48 GB before activations. That is a whole GPU. This single number explains two industry choices you will meet later:

* **GRPO drops the critic entirely** (Lecture 11) and uses the group mean from Lecture 04 instead.
* **Privileged-state critics** make the critic nearly free in simulation, whatever the size of the actor.

In regime B (ManiSkill, MLP policies), the critic is a rounding error, and the decision is purely about sample efficiency. In regime C (VLA), critic choice is a **memory decision** first. With pixels, measure where memory goes. The rollout buffer of `uint8` frames, the CNN activations during the update, and a duplicated encoder for a pixel critic each show up separately in `torch.cuda.max_memory_allocated`.

---

## 8. Build it

Code lives in `actor_critic/`. All labs use ManiSkill3 on GPU (regime B).

### Lab 5a — A2C + GAE on PickCube (state)

```python
# actor_critic/a2c_gae.py
import argparse, time, torch, torch.nn as nn, gymnasium as gym
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

def make_env(env_id, n, obs_mode="state", reward_mode="normalized_dense", control_mode="pd_joint_delta_pos"):
    return gym.make(env_id, num_envs=n, obs_mode=obs_mode, reward_mode=reward_mode,
                    control_mode=control_mode, sim_backend="physx_cuda")

def mlp(i, o, h=256, out_gain=1.0):
    layers, d = [], i
    for _ in range(3):
        layers += [nn.Linear(d, h), nn.Tanh()]; d = h
    last = nn.Linear(d, o); nn.init.orthogonal_(last.weight, out_gain); nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)

def gae(r, v, v_next, terminated, done, gamma, lam):
    """All (K, N). v_next[t] = V(true s_{t+1}), using the final obs for envs that just ended.
    `terminated` stops bootstrapping; `done` = terminated | truncated stops the trace."""
    adv, last = torch.zeros_like(r), torch.zeros_like(r[0])
    for t in reversed(range(r.shape[0])):
        delta = r[t] + gamma * (1.0 - terminated[t]) * v_next[t] - v[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + v                                       # advantages, lambda-return targets

def explained_variance(v, ret):
    return (1 - (ret - v).var() / ret.var().clamp_min(1e-8)).item()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=1024); p.add_argument("--rollout", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.8); p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--iters", type=int, default=1000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward", default="normalized_dense"); p.add_argument("--ent-coef", type=float, default=0.0)
    a = p.parse_args(); torch.manual_seed(a.seed); N, K = a.num_envs, a.rollout

    base = make_env("PickCube-v1", N, reward_mode=a.reward)
    envs = ManiSkillVectorEnv(base, N, ignore_terminations=False, record_metrics=True)  # partial resets
    dev = envs.unwrapped.device
    od, ad = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
    actor, critic = mlp(od, ad, out_gain=0.01).to(dev), mlp(od, 1).to(dev)
    log_std = nn.Parameter(torch.full((ad,), -0.5, device=dev))
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters(), log_std], lr=3e-4, eps=1e-5)

    obs, _ = envs.reset(seed=a.seed); succ = float("nan")
    for it in range(a.iters):
        buf = {k: [] for k in ("obs", "act", "r", "v", "v_next", "term", "done")}
        t0 = time.perf_counter()
        for t in range(K):
            with torch.no_grad():
                mu = actor(obs); act = mu + log_std.exp() * torch.randn_like(mu)
                v = critic(obs).squeeze(-1)
            nobs, r, term, trunc, info = envs.step(act.clamp(-1, 1))
            with torch.no_grad():
                v_next = critic(nobs).squeeze(-1)                         # valid where the episode continues
                if "final_observation" in info:                           # where it ended: use the real s_{t+1}
                    m = info["_final_info"]
                    v_next[m] = critic(info["final_observation"][m]).squeeze(-1)
                    succ = info["final_info"]["episode"]["success_once"][m].float().mean().item()
            for k, x in zip(buf, (obs, act, r.float(), v, v_next, term.float(), (term | trunc).float())):
                buf[k].append(x)
            obs = nobs
        torch.cuda.synchronize(); t1 = time.perf_counter()
        B = {k: torch.stack(x) for k, x in buf.items()}
        adv, ret = gae(B["r"], B["v"], B["v_next"], B["term"], B["done"], a.gamma, a.lam)
        o, act, adv, ret = B["obs"].flatten(0, 1), B["act"].flatten(0, 1), adv.flatten(), ret.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        dist = torch.distributions.Normal(actor(o), log_std.exp())
        pg_loss = -(dist.log_prob(act).sum(-1) * adv).mean()
        v_loss = 0.5 * (critic(o).squeeze(-1) - ret).pow(2).mean()
        loss = pg_loss + 0.5 * v_loss - a.ent_coef * dist.entropy().sum(-1).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 0.5); opt.step()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        if it % 20 == 0:
            print(f"it {it} succ {succ:.2f} EV {explained_variance(B['v'].flatten(), ret):+.2f} "
                  f"rollout {t1-t0:.2f}s learn {t2-t1:.3f}s sps {N*K/(t1-t0):,.0f}")
```

Two deliberate checks:

* **Inject the bug.** Replace `(1.0 - terminated[t])` with `(1.0 - done[t])`, so truncation is treated as termination, and rerun with \(\gamma = 0.99\). Compare explained variance and success. Write down what you see. With \(\gamma = 0.8\) the effect is smaller. Explain why using the table in Section 4.1.
* **Value sanity plot.** Roll out one episode and plot \(V_\phi(s_t)\) next to `is_grasped` and the success flag over time. It should step up at the grasp.

### Lab 5b — Pixel actor: privileged-state critic vs pixel critic

Switch the actor to pixels and compare two critics on identical actor architectures. To keep bootstrapping simple, run **aligned episodes**: no auto-reset, all envs reset together, and the rollout length equals the 50-step episode. Every rollout then ends in a truncation at which you can still read the privileged state before resetting.

```python
# actor_critic/asym_pixels.py  (key differences from a2c_gae.py)
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

base = make_env("PickCube-v1", N, obs_mode="rgb")
base = FlattenRGBDObservationWrapper(base, rgb=True, depth=False, state=True)  # obs = {"rgb", "state"}
envs = ManiSkillVectorEnv(base, N, auto_reset=False, ignore_terminations=True, record_metrics=True)
priv = lambda: envs.unwrapped.get_state()      # flat sim state: all actor poses/vels + robot joints

class Encoder(nn.Module):                       # actor's (and pixel critic's) image + proprio encoder
    def __init__(self, c, sd):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(c, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                 nn.Linear(64 * 16, 256), nn.ReLU())
        self.state = nn.Sequential(nn.Linear(sd, 256), nn.ReLU())
    def forward(self, rgb, state):
        return torch.cat([self.cnn(rgb.permute(0, 3, 1, 2).float() / 255.0), self.state(state)], -1)

# critic = "state": V(s) = mlp(priv_dim, 1)(priv())               -- privileged, tiny
# critic = "pixel": V(o) = Linear(512, 1)(Encoder(c, sd)(rgb, st)) -- separate encoder, same size as actor's
# Rollout: obs, _ = envs.reset(); for t in range(50): store rgb (uint8), state, priv() if needed, act, r.
# The last step is a truncation: v_next[49] = V(final s or o) computed BEFORE the next envs.reset().
# gae(..., terminated=zeros, done=zeros except done[49]=1, ...)
```

The PickCube state observation includes `goal_pos` and `tcp_pose` but, in `rgb` mode, not the cube pose. The actor must find the cube in pixels. The privileged critic gets it for free from `get_state()`. Run both critics for 3 seeds at `num_envs=256`, and record success vs env steps, explained variance over training, peak GPU memory, and rollout/learn time per iteration.

### Lab 5c — λ and γ sweep

On the state task from Lab 5a, sweep \(\lambda \in \{0, 0.9, 0.95, 1.0\}\) × \(\gamma \in \{0.8, 0.95, 0.99\}\), 3 seeds each (36 short runs; at regime B speeds this is minutes to hours on one GPU). Then repeat the \(\gamma\) sweep with `--reward sparse`, starting from your Lecture 02 BC policy as the actor so that success is non-zero. Pass `make_env` the same `control_mode` the BC policy was trained with. Predict the result from Section 4.1 before you look.

---

## 9. Use it in the real stack

* **CleanRL** `ppo_continuous_action.py` and **ManiSkill**'s `examples/baselines/ppo/ppo.py` contain the exact GAE recursion above. ManiSkill's version also evaluates the critic on `final_observation` for envs that ended mid-rollout, and has a comment discussing the termination mask. Read it and compare with your `gae()`.
* **Isaac Lab + rsl_rl** support separate observation groups for the policy and the critic. This is the standard way legged-locomotion and manipulation tasks feed privileged terms (true friction, contact forces, base velocity) to the critic only.
* **VLA RL**: frameworks such as [RLinf](https://github.com/RLinf/RLinf) support PPO-style training of VLAs, where the choice between a value head on the VLA and a separate critic is exactly the Section 7 memory trade-off. Check what your framework of choice does before assuming.

---

## 10. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Success vs env steps / vs wall-clock** | ≥ 3 seeds per configuration | Sample and time efficiency of each critic and λ |
| **Explained variance** | Per iteration, from `explained_variance()` | Is the critic helping? Negative means check masks |
| **Advantage std before normalization** | Per iteration | Tracks the variance that λ trades for bias |
| **Value at reset vs empirical return** | Mean \(V(s_0)\) vs mean discounted return from \(s_0\) | Calibration; with sparse reward, a check on "probability of success" |
| **Peak GPU memory** | `torch.cuda.max_memory_allocated`, per critic type | The asymmetric critic's memory saving, measured |
| **Rollout / learn time split** | `t1 - t0`, `t2 - t1` | Pixel critics add forward passes in rollout and backward passes in learn |
| **Params: actor vs critic** | `sum(p.numel() for p in m.parameters())` | Ties memory to architecture |

---

## 11. Ship it

Commit `actor_critic/` containing:

* `a2c_gae.py`, `asym_pixels.py`, and `sweep.sh`
* `results/lambda_gamma_sweep.png`: final success (mean ± range over seeds) as a heatmap over \(\lambda\) × \(\gamma\), dense reward; plus a second panel for the sparse-reward \(\gamma\) sweep
* `results/asym_vs_pixel.csv` and `.png`: success curves, explained variance, peak memory, and s/iteration for the two critics
* `results/truncation_bug.png`: EV and success with the correct mask vs the injected bug
* `NOTES.md`: which \(\gamma, \lambda\) you would choose for (a) PickCube with dense reward, (b) a 300-step LIBERO task with sparse reward and \(H = 10\) chunks, with one sentence of justification each, and your measured memory saving from the privileged critic

---

## Exit criteria

You can move on when you can:

* derive \(\mathbb{E}[\delta_t \mid s_t, a_t] = A^\pi\) and the GAE sum-of-δ form from the n-step definition
* state the effective horizon for a given \(\gamma\) and use it to reject a bad \(\gamma\) for a sparse-reward task
* point to the two masks in your GAE code, explain each, and show the measured effect of swapping them
* justify feeding privileged state to the critic but not the actor, and name the case where a state-only critic can bias learning
* estimate the memory cost of a critic for a 3B-parameter policy under each design in Section 7

---

## Self-check

1. Your PPO run on a 50-step task with sparse success reward uses \(\gamma = 0.8\) copied from a ManiSkill dense-reward config. Explained variance hovers around 0, and success never improves even when starting from a BC policy that succeeds 30% of the time. What is the most likely cause, and what \(\gamma\) would you try?
2. A colleague's critic has explained variance of −0.4 and gets worse over training, but only on the task with a time limit. The same code is fine on CartPole with `terminated`-only endings. Name the bug and the one-line fix.
3. You set \(\lambda = 0\) to minimize variance, and early training is much worse than with \(\lambda = 0.95\), though it catches up later. Explain both halves of that observation in terms of critic error.
4. For π0.5 on LIBERO you must choose a critic: (a) a value head on the VLM backbone, (b) a separate 3B-parameter copy, or (c) a 3-layer MLP on simulator object poses. Rank them by memory and by expected advantage quality, and say what each would need in order to be used on a real robot.
5. In a task where the target object starts inside a closed drawer, your state-only privileged critic trains well, but the pixel actor learns a worse policy than with a history-conditioned critic. Why can a better-informed critic hurt here, and what input would you give the critic instead?
6. Your pixel-critic run uses 2.3× the peak memory of the privileged-critic run, though both actors are identical. List the specific tensors responsible, and which of them would also grow if you doubled `num_envs`.

---

## References

* CS 285 Lecture 6 "Actor-Critic Algorithms" — [slides and video](https://rail.eecs.berkeley.edu/deeprlcourse/)
* Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., ch. 6 (TD), 7 (n-step), 12 (eligibility traces / λ-returns), 13 (actor-critic) — [online](http://incompleteideas.net/book/the-book-2nd.html)
* Schulman, Moritz, Levine, Jordan, Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," 2015 — [arXiv](https://arxiv.org/abs/1506.02438)
* Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (A3C), 2016 — [arXiv](https://arxiv.org/abs/1602.01783)
* Pinto, Andrychowicz, Welinder, Zaremba, Abbeel, "Asymmetric Actor Critic for Image-Based Robot Learning," 2017 — [arXiv](https://arxiv.org/abs/1710.06542)
* OpenAI et al., "Learning Dexterous In-Hand Manipulation," 2018 — [arXiv](https://arxiv.org/abs/1808.00177)
* Baisero & Amato, "Unbiased Asymmetric Reinforcement Learning under Partial Observability," 2021 — [arXiv](https://arxiv.org/abs/2105.11674)
* Pardo, Tavakoli, Levdik, Kormushev, "Time Limits in Reinforcement Learning," 2017 — [arXiv](https://arxiv.org/abs/1712.00378)
* Gymnasium, "Terminated / Truncated step API" — [Farama blog](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)
* Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study," 2020 — [arXiv](https://arxiv.org/abs/2006.05990)
* Haarnoja et al., "Soft Actor-Critic," 2018 — [arXiv](https://arxiv.org/abs/1801.01290) (the replay-buffer + learned-Q branch of actor-critic; Lecture 07)

---

## Next in this special course

* Next: [Lecture 06 — PPO, Trust Regions, and the KL Leash](Lecture-06.md)
* Previous: [Lecture 04 — Policy Gradients: Do More of What Worked](Lecture-04.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
