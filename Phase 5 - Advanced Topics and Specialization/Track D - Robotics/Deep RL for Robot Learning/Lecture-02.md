# Lecture 02: Imitation Learning: Behavioral Cloning and DAgger

## Overview

The simplest way to teach a robot is to **show it**. Record an expert doing the task, then train a network to answer one question: *"when the camera picture looks like this, what did the expert do?"* That is **behavioral cloning (BC)** — plain supervised learning, with observations as inputs and expert actions as labels.

It works surprisingly well, and it is how almost every VLA policy, π0.5 included, gets its skills. But it has one built-in flaw. A student driver who has only ever watched perfect driving has never seen the car drift toward the curb. The first small mistake puts them somewhere the teacher never was, they have no idea what to do there, and the next mistake is bigger. Errors don't just add up; they **compound**.

This lecture makes that intuition precise (the \(O(\epsilon T^2)\) bound), shows the fix that changes it to \(O(\epsilon T)\) (**DAgger**: let the robot drive, then ask the expert what it *should* have done in the places it actually went), and then asks the hardware question: what does expert data cost to produce, and what changes when the expert is a GPU-simulated script instead of a person with a teleop rig?

By the end you should be able to:

* write the BC objective as maximum likelihood and explain why MSE regression is a special case of it
* explain covariate shift and sketch the proof that BC's worst-case cost grows as \(\epsilon T^2\)
* implement DAgger, including the \(\beta\) mixing schedule, and explain why it gets \(O(\epsilon T)\)
* write a **privileged, stateless scripted expert** for a GPU-parallel simulator and use it as a DAgger oracle
* estimate demo-collection and expert-labeling throughput for teleop, motion planners, and scripted experts

---

## 1. Why it matters: the cheapest skill you will ever get

For our 7-DoF arm on the kitchen table, "put the bowl on the plate" has a sparse 0/1 reward at the end of a few hundred steps. RL from scratch on that signal almost never sees a success (Lecture 12 deals with that). A few dozen human demonstrations, cloned, often get a policy that succeeds a good fraction of the time on the layouts it was shown.

That is why the modern recipe from Lecture 01 puts imitation **before** RL:

| Stage | Data | Objective | Lecture |
|---|---|---|---|
| Pre-training | Large, messy, multi-robot + web data | Imitation (next-action / next-token likelihood) | 03 |
| Post-training (SFT) | Small, clean task demos | Imitation | 02-03 |
| RL post-training | The policy's own rollouts + a score | Policy improvement | 04-11 |

Two engineering facts make BC the default first step:

1. **It is supervised learning.** No reward, no critic, no rollouts during training. Every trick from supervised deep learning (large batches, mixed precision, data loaders, LoRA) transfers unchanged.
2. **Its failure mode is predictable.** BC fails when the policy reaches states the demos never covered. Once you know that, you know where to spend data.

**Robot connection.** A π0.5 checkpoint fine-tuned on LIBERO demonstrations is a BC policy. Robustness tests that move objects to new positions are exactly the "states the demos never covered" problem. Much of the headroom left on top of a strong LIBERO checkpoint is distribution shift, not missing skill.

---

## 2. Mental model: copying, and where copying breaks

### 2.1 BC as maximum likelihood

Given a dataset of expert pairs \(\mathcal{D} = \{(o_i, a_i)\}\) collected by running the expert policy \(\pi^*\), BC fits

$$
\theta^\star = \arg\max_\theta \; \mathbb{E}_{(o,a) \sim \mathcal{D}} \big[ \log \pi_\theta(a \mid o) \big].
$$

If the policy is a Gaussian with fixed variance, \(\pi_\theta(a \mid o) = \mathcal{N}(a;\, \mu_\theta(o),\, \sigma^2 I)\), then

$$
-\log \pi_\theta(a \mid o) = \frac{1}{2\sigma^2} \lVert a - \mu_\theta(o) \rVert^2 + \text{const},
$$

so maximum likelihood **is** mean-squared-error regression. Keep that equivalence in mind: Lecture 03 shows that a unimodal Gaussian, and therefore plain MSE, is the wrong model when the expert sometimes goes left and sometimes goes right.

### 2.2 Covariate shift

The training inputs come from the expert's state distribution \(p_{\pi^*}(o)\). At test time the inputs come from the learner's own distribution \(p_{\pi_\theta}(o)\). Supervised learning guarantees small error only on the training distribution:

$$
\mathbb{E}_{o \sim p_{\pi^*}} \big[ \ell(\pi_\theta, o) \big] \text{ small} \;\;\not\Rightarrow\;\; \mathbb{E}_{o \sim p_{\pi_\theta}} \big[ \ell(\pi_\theta, o) \big] \text{ small}.
$$

This is **covariate shift**, and in sequential decision-making the learner *causes* it. Each action changes the next input, so a small error moves the learner off the expert's distribution, where its errors are larger, which moves it further off.

```text
   expert path   ●──●──●──●──●──●──●──●──● goal
   learner path  ●──●──●─╮
                          ╰─●─╮          (small error → unfamiliar state)
                               ╰──●──╮   (bigger error → more unfamiliar)
                                      ╰──● crash
```

### 2.3 The \(\epsilon T^2\) bound (Ross & Bagnell, 2010)

Setup: horizon \(T\), per-step cost \(c \in [0, 1]\) (think "1 if not in a good state"). Suppose the learner disagrees with the expert with probability at most \(\epsilon\) **on states drawn from the expert's distribution** — the quantity supervised training actually controls.

**Proof intuition.** As long as the learner hasn't made a mistake yet, it is on the expert's distribution. The probability that it has made no mistake in the first \(t\) steps is at least \((1-\epsilon)^t\). Once it has made a mistake, assume the worst: it pays cost 1 on every remaining step. So the extra expected cost at step \(t\) is at most

$$
1 - (1-\epsilon)^t \;\le\; \epsilon t ,
$$

and summing over the episode,

$$
J(\pi_\theta) \;\le\; J(\pi^*) + \sum_{t=1}^{T} \epsilon t \;=\; J(\pi^*) + \epsilon \frac{T(T+1)}{2} \;=\; J(\pi^*) + O(\epsilon T^2).
$$

Two readings matter:

* **Double the horizon, up to 4× the trouble.** A per-step error rate that looks fine on a 50-step task can be disastrous on a 500-step task.
* **The bound is tight in the worst case.** There are MDPs (a tightrope: one wrong step and you can never get back) where BC really does pay \(\epsilon T^2\). Many real tasks are kinder — the arm can re-approach a missed bowl — which is why BC often works better than the bound suggests. Your lab measures where PickCube sits.

### 2.4 DAgger: label the states the learner actually visits

The fix is to train on the learner's distribution instead of the expert's. **DAgger** (Dataset Aggregation; Ross, Gordon & Bagnell, 2011):

```text
D ← expert demonstrations
for i = 1 … N:
    π_i ← train BC on D
    run the mixture  β_i·π*  +  (1 − β_i)·π_i  in the environment
    for every visited observation o: ask the expert  a* = π*(o)
    D ← D ∪ {(o, a*)}                       # aggregate, never discard
return best π_i on validation rollouts
```

* **\(\beta\) mixing.** At step \(i\), the executed action comes from the expert with probability \(\beta_i\) and from the learner otherwise. The labels are always the expert's. The analysis allows any schedule with \(\beta_i \to 0\) (for example \(\beta_i = p^{i-1}\)). The paper notes that the parameter-free choice — \(\beta_1 = 1\) (plain BC first) and \(\beta_i = 0\) afterwards — often works well in practice.
* **Aggregation.** Training on the union of all iterations' data, not only the latest, is what turns DAgger into a no-regret online learner (Follow-the-Leader). That reduction is where its guarantee comes from.
* **Guarantee.** The training error \(\epsilon\) is now measured on the learner's own distribution. If one mistake can increase the remaining cost by at most \(u\) (the task is *recoverable*), the cost grows as \(O(u\,\epsilon\,T)\), linear in the horizon. On a tightrope, \(u\) is large and nothing saves you. On a kitchen table, \(u\) is small.

**Why it works, in one sentence:** the expert's labels are spent on the mistakes the learner actually makes, not on states it would never reach.

### 2.5 Other ways to get recovery data

DAgger needs an expert who can be queried at arbitrary states. When that is expensive, the same goal — demos that contain mistakes **and** recoveries — can be approached in other ways:

| Technique | Idea | Cost |
|---|---|---|
| **Noise injection (DART; Laskey et al., 2017)** | Add noise to the *expert's* executed actions while recording the expert's intended action as the label. The expert drifts and corrects itself, so the data contains recoveries. The noise level is tuned to resemble the learner's error. | Same as BC collection, off-policy |
| **Synthetic viewpoints** | NVIDIA's end-to-end driving system (Bojarski et al., 2016) used left/right-shifted camera views labeled with corrective steering | Cheap; specific to geometry you can fake |
| **Intervention-based (HG-DAgger; Kelly et al., 2019)** | The learner drives; a human takes over only when they judge things unsafe; only the takeover segments are added | Human stays in control, which is how humans label well |
| **More capable models** | The bounds are linear in \(\epsilon\). A bigger model with better visual features lowers \(\epsilon\) directly. | Compute (Lecture 03) |
| **More diverse data** | Many tasks, layouts, and operators widen \(p_{\pi^*}\) so the learner's distribution overlaps it more | Collection hours |

Why HG-DAgger exists: humans are poor at labeling a recorded state they were not controlling. Watching a video of the arm drifting and saying what the correct 7-D action would have been at frame 137 does not produce good labels. A human teleoperator in control does. Spencer et al. (2021) analyze when interactive feedback is needed at all and when offline data is enough.

---

## 3. The privileged scripted expert

In simulation, you can write an expert that cheats. It reads the true pose of every object from the simulator — information the learner will never get from pixels — and computes a good action with a few lines of geometry. For DAgger, it has to satisfy one requirement that a recorded human demo cannot:

> **The expert must be a policy, not a trajectory.** It has to return a sensible action from *any* state, including the strange states the learner wanders into. Replaying a stored trajectory, or a controller whose internal phase variable assumes it started at reset, is not enough.

So write it **stateless**: decide the phase from the current state (Is the cube grasped? Is the gripper above it?) instead of from a counter. This is the rung-2 version of what the capstone needs on LIBERO, where a scripted expert reading object poses from the simulator can label states a π0.5 rollout visits without a human in the loop.

---

## 4. The hardware view: where the cost of imitation goes

Imitation splits into three costs with very different hardware profiles:

| Cost | Teleop human | Motion-planner expert | Scripted privileged expert (GPU sim) |
|---|---|---|---|
| **Demo collection rate** | Real time: episode length ÷ control rate, plus resets. On the order of tens of demos per operator-hour. | Planning time per episode (often CPU-bound, ms-s per plan) + sim | Bounded by env-steps/s: with 10⁵ steps/s and 50-step episodes, on the order of 10³ episodes/s |
| **Expert-label rate (DAgger)** | Poor. Off-policy labeling is unreliable, so use interventions. | One plan per queried state, often the loop's bottleneck | A few batched tensor ops per step, effectively free next to the sim step |
| **Who pays** | People-hours | CPU cores | GPU seconds |
| **Label quality** | Natural, multimodal, inconsistent | Consistent but can be jerky | Perfectly consistent and unimodal (can be *too* easy; see Lecture 03) |

(Order-of-magnitude figures. Replace them with your measurements from the lab.)

**Training is the cheap regime.** BC is supervised learning: one forward and one backward pass per sample, and no environment in the loop. An MLP on state processes millions of samples per second on one GPU. For a VLA, BC fine-tuning is where the memory goes: full fine-tuning in mixed precision with Adam needs about 16 bytes per parameter (bf16 weights and gradients, fp32 master weights, two fp32 Adam moments) before activations. For a ~3B-parameter π0.5 that is on the order of 50 GB, which is why LoRA is the single-GPU path.

**DAgger brings rollouts back.** One DAgger iteration costs

$$
t_{\text{iter}} \approx \underbrace{t_{\text{rollout}}(M \text{ episodes})}_{\text{regime A/B/C from Lecture 01}} + \underbrace{t_{\text{label}}(M T)}_{\text{expert}} + \underbrace{t_{\text{train}}(\lvert \mathcal{D}_i \rvert)}_{\text{supervised}} .
$$

Two traps follow:

* \(\lvert \mathcal{D}_i \rvert\) grows every iteration. Retraining from scratch on the full aggregate makes total training cost quadratic in the number of iterations. Use a fixed step budget per iteration, warm-start from \(\pi_{i-1}\), or both.
* For a VLA, \(t_{\text{rollout}}\) is regime C: the policy forward pass dominates, exactly like RL. DAgger is not "free imitation". It pays the same rollout bill as on-policy RL, and only the labels are cheaper than rewards.

---

## 5. Build it: BC and DAgger on PickCube

Everything lives in `bc_dagger/`. We use ManiSkill3 `PickCube-v1` with `obs_mode="state"` and `control_mode="pd_ee_delta_pose"`. Its action is 7-D (3 translation deltas, 3 rotation deltas, gripper), matching the running example. With the default normalization, ±1 on a translation axis means ±0.1 m per control step, and gripper −1/+1 means close/open. Check these against your ManiSkill version's Panda controller config.

### Lab 2a — Scripted privileged expert (stateless, batched)

```python
# bc_dagger/expert.py
import torch

@torch.no_grad()
def expert_action(env, k: int = 1, gain: float = 0.5) -> torch.Tensor:
    """Privileged expert for PickCube-v1 (pd_ee_delta_pose). Reads true poses from the sim.
    Stateless: the phase is inferred from the current state, so it can label ANY state.
    k > 1 slows the expert by capping per-step motion at 1/k (used for the horizon sweep)."""
    u = env.unwrapped
    cube, goal = u.cube.pose.p, u.goal_site.pose.p          # (N, 3) world frame
    tcp = u.agent.tcp.pose.p                                # gripper tool-center point
    grasped = u.agent.is_grasping(u.cube)                   # (N,) bool

    xy_err = torch.linalg.norm((cube - tcp)[:, :2], dim=1)
    z_err = (tcp - cube)[:, 2].abs()
    above = cube.clone(); above[:, 2] += 0.05

    target = torch.where((xy_err > 0.01)[:, None], above, cube)   # align over cube, then descend
    target = torch.where(grasped[:, None], goal, target)          # once grasped, carry to goal
    close = grasped | ((xy_err < 0.01) & (z_err < 0.01))

    a = torch.zeros(cube.shape[0], 7, device=cube.device)
    a[:, :3] = (gain * (target - tcp) / 0.1).clamp(-1.0 / k, 1.0 / k)
    a[:, 6] = torch.where(close, -1.0, 1.0)                      # -1 close, +1 open
    return a
```

Rotation stays at zero, so the gripper never aligns with the cube's yaw. The expert will therefore fail on some layouts. That is realistic: **filter demos by success**, and report the expert's own success rate as the ceiling.

### Lab 2b — Rollout, collect, and label in one function

```python
# bc_dagger/rollout.py
import torch, gymnasium as gym
import mani_skill.envs  # registers PickCube-v1
from expert import expert_action

def make_env(n: int, k: int = 1):
    # We drive exactly T = 50k steps ourselves; max_episode_steps only affects the truncation flag.
    return gym.make("PickCube-v1", num_envs=n, obs_mode="state",
                    control_mode="pd_ee_delta_pose", max_episode_steps=50 * k)

@torch.no_grad()
def rollout(env, policy=None, beta: float = 1.0, k: int = 1, seed: int = 0):
    """Run the beta-mixture of expert and learner. Labels are ALWAYS the expert's action."""
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    N, dev = u.num_envs, obs.device
    init_xy = u.cube.pose.p[:, :2].clone()
    O, A = [], []
    succ = torch.zeros(N, dtype=torch.bool, device=dev)
    for _ in range(50 * k):
        a_exp = expert_action(env, k)
        if policy is None or beta >= 1.0:
            a_run = a_exp
        else:
            use_exp = torch.rand(N, 1, device=dev) < beta
            a_run = torch.where(use_exp, a_exp, policy(obs))
            a_run[:, :3] = a_run[:, :3].clamp(-1.0 / k, 1.0 / k)   # same motion cap as expert
        O.append(obs); A.append(a_exp)
        obs, _, _, _, info = env.step(a_run)
        succ |= info["success"]
    return torch.stack(O, 1), torch.stack(A, 1), succ, init_xy   # (N, T, ...) per episode
```

Measure the collection rate now: episodes/s and labels/s at `n = 256, 1024, 4096`. That number goes into the hardware note.

### Lab 2c — BC

```python
# bc_dagger/bc.py
import torch, torch.nn as nn

class Normalized(nn.Module):        # obs normalization baked in, so torch.save(policy) works (a lambda can't be pickled)
    def __init__(self, net, mu, sd):
        super().__init__(); self.net = net
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
    def forward(self, o):
        return self.net((o - self.mu) / self.sd)

def train_bc(O, A, steps=5000, bs=4096, lr=3e-4, h=256):
    O, A = O.reshape(-1, O.shape[-1]), A.reshape(-1, A.shape[-1])   # flatten episodes
    mu, sd = O.mean(0), O.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(O.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                        nn.Linear(h, A.shape[1]), nn.Tanh()).to(O.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        i = torch.randint(0, O.shape[0], (bs,), device=O.device)
        loss = ((net((O[i] - mu) / sd) - A[i]) ** 2).mean()        # Gaussian MLE == MSE
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return Normalized(net, mu, sd)
```

(The gripper is really a binary decision. Training it with BCE as a separate head is a fair improvement. Keep MSE first so the baseline is simple.)

Evaluate any policy with `rollout(make_env(1024), policy, beta=0.0)` on a **fixed list of seeds** shared by every run, and report `succ.float().mean()` alongside the expert's success on the same seeds.

### Lab 2d — Three degradation experiments

Train each with ≥ 3 seeds and keep only successful expert episodes as demos.

1. **Number of demos.** \(\{10, 30, 100, 300, 1000\}\) episodes at \(k = 1\). Plot success vs demos on a log x-axis.
2. **Horizon.** Slow the task down with \(k \in \{1, 2, 4, 8\}\) (\(T = 50k\)). Expert and learner have the same per-step motion cap, so the same task now takes \(k\) times as many decisions. Keep the number of demo *episodes* fixed. Also measure the **per-step error on the expert's distribution**, \(\hat\epsilon = \mathbb{E}_{o \sim \mathcal{D}_{\text{val}}}\lVert \pi_\theta(o) - \pi^*(o)\rVert^2\), at each \(k\). Time dilation changes action magnitudes as well as \(T\), so you need \(\hat\epsilon\) to separate the two effects.
3. **Initial-pose shift.** Keep only demos whose initial cube position lies in the inner half of the spawn square (\(\lVert xy - c\rVert_\infty \le 0.05\) m, with \(c\) the mean initial position). Evaluate on the full square and bucket results by initial \(\lVert xy - c\rVert_\infty\). The outer ring is your rung-2 "perturbed layout".

### Lab 2e — DAgger

```python
# bc_dagger/dagger.py  (sketch of the loop; reuse rollout/train_bc above)
D_O, D_A, succ, xy = rollout(make_env(256))                 # iteration 0: expert demos
D_O, D_A = D_O[succ], D_A[succ]
queries, log = D_O.shape[0] * D_O.shape[1], []
for it in range(1, 9):
    pi = train_bc(D_O, D_A)
    O, A, _, _ = rollout(make_env(256), pi, beta=0.0, seed=1000 + it)   # learner drives
    D_O, D_A = torch.cat([D_O, O]), torch.cat([D_A, A])                 # aggregate
    queries += O.shape[0] * O.shape[1]
    _, _, s_eval, xy_eval = rollout(make_env(1024), pi, beta=0.0, seed=0) # fixed eval seeds
    log.append((it, queries, s_eval.float().mean().item()))
```

Run DAgger from a deliberately weak start (10-30 demos, inner-square layouts only, \(k = 4\)) and plot:

* success vs DAgger iteration
* success vs **cumulative expert queries**, on the same axes as **BC trained with the same number of expert labels** (more demos instead of DAgger data). This is the fair comparison. In sim, labels are nearly free, so DAgger's value is *which* states get labeled, not how many.
* success on the outer ring, before and after DAgger

Optional: \(\beta_i = 0.5^{i-1}\) vs \(\beta_1 = 1,\ \beta_{i>1} = 0\), and DART-style noise (add Gaussian noise to `a_run` while labeling with `a_exp`) as a non-interactive baseline at equal label count.

---

## 6. Use it in the real stack

* **Demo tooling.** [robomimic](https://github.com/ARISE-Initiative/robomimic) (BC, BC-RNN and the study of what matters in learning from human demos), [LeRobot](https://github.com/huggingface/lerobot) (dataset format, teleop recording, BC policies), and ManiSkill3's own demo datasets and motion-planning solutions for its tabletop tasks (see the ManiSkill docs).
* **LIBERO.** Ships human teleoperation demonstrations for every task. openpi's LIBERO fine-tuning is behavioral cloning of those demos into π0.5. When you fine-tune it, you are running this lecture's Lab 2c at 10⁹ parameters.
* **Interactive correction at scale.** Real-robot pipelines use intervention-based collection (HG-DAgger-style takeovers) because a human in control produces usable labels. In simulation, the privileged scripted expert replaces the human entirely, as long as it can be written. For long-horizon household tasks, writing it is real engineering work.
* **Robot connection.** On rung 3, DAgger becomes: roll out π0.5 on LIBERO layouts (including moved objects), label the visited states with a scripted expert that reads object poses from the simulator, and fine-tune on the aggregate. Two caveats carry into Lecture 03. The expert's action space and chunking must match the policy's. And a scripted expert's style (perfectly consistent, unimodal) differs from the human demos the policy learned from, so mixing them creates exactly the multimodal targets MSE handles badly.

---

## 7. Measure it

| Metric | Definition | Why it matters |
|---|---|---|
| **Expert ceiling** | Expert success on the fixed eval seeds | Upper bound for BC; report it next to every BC number |
| **Success vs demos** | Learner success at each demo count | Data efficiency of copying |
| **\(\hat\epsilon\) (on-distribution error)** | Action MSE vs expert on held-out *expert* states | The \(\epsilon\) in the bound, measured |
| **\(\hat\epsilon_{\text{on-policy}}\)** | Action MSE vs expert on states the *learner* visits | The gap between this and \(\hat\epsilon\) *is* covariate shift |
| **Success vs horizon \(T\)** | Learner success at each \(k\) | Tests the \(T\) vs \(T^2\) story |
| **Success vs expert queries** | DAgger vs equal-label BC | Whether coverage beats quantity |
| **Demo collection rate** | Expert episodes/s and labels/s at each `num_envs` | Data cost per hour of GPU vs per hour of human |
| **BC training throughput** | Samples/s and peak GPU memory | Confirms training is the cheap term |

---

## 8. Ship it

Commit `bc_dagger/` containing:

* `expert.py`, `rollout.py`, `bc.py`, `dagger.py`, `run_sweeps.sh`
* `demos.pt` — the expert dataset later lectures load: `torch.save({"O": O, "A": A, "success": succ}, "bc_dagger/demos.pt")` with `O: (N, T, obs_dim)`, `A: (N, T, 7)`, `success: (N,)` straight from `rollout()`
* `bc_policy.pt` — your best policy (final DAgger iterate) as a whole module, `torch.save(policy, "bc_dagger/bc_policy.pt")` (keep `bc.py` importable when loading, since pickle stores the `Normalized` class by reference), mapping state obs → mean 7-D action; Lecture 06 fine-tunes it with PPO
* `results.csv` — one row per (experiment, setting, seed) with success, \(\hat\epsilon\), \(\hat\epsilon_{\text{on-policy}}\), expert queries, wall-clock
* `success_vs_demos.png`, `success_vs_horizon.png` (log-log of failure rate vs \(T\), with reference slopes 1 and 2), `success_vs_radius.png`, `dagger_vs_bc_queries.png`
* `COMPOUNDING.md` — half a page. Answer: does your measured failure rate vs \(T\) look closer to linear or quadratic, after accounting for how \(\hat\epsilon\) changed with \(k\)? How large is \(\hat\epsilon_{\text{on-policy}} / \hat\epsilon\) for BC vs the final DAgger policy? What did one hour of GPU buy in demos, compared with your estimate for a human teleoperator?

The demos and the best DAgger policy are inputs to Lecture 03 (flow-matching policy) and Lecture 06 (BC-initialized PPO).

---

## Exit criteria

You can move on when you can:

* derive MSE regression from Gaussian maximum likelihood in two lines
* reproduce the \(\epsilon T^2\) argument on a whiteboard and say what assumption makes it worst-case
* explain why DAgger labels must come from an expert that is a policy, and why aggregation (not replacement) matters
* show, with your own plots, where BC degrades on PickCube (demos, horizon, initial pose) and how much DAgger recovers per expert query
* estimate the cost of 1,000 demos via teleop, a motion planner, and a scripted GPU-sim expert

---

## Self-check

1. Your BC policy reaches 95% success on 50-step PickCube. You change the controller so the same motion takes 200 steps, keep the same per-step error rate, and success collapses. Is that consistent with the \(\epsilon T^2\) bound? What else changed that you would need to measure before blaming compounding?
2. A colleague proposes DAgger with human labelers who watch videos of the robot's failed rollouts and type in the correct action for each frame. Why will the labels likely be poor, and what would you do instead?
3. Your scripted expert has a `phase` counter that advances from "approach" to "grasp" to "lift" based on elapsed steps. Why does this break DAgger, and how do you fix it without changing what the expert does on its own rollouts?
4. DAgger and BC both reach 80% after 200,000 expert labels in simulation, but DAgger is far better on the outer-ring initial poses. Explain the difference in terms of which states received labels.
5. Fine-tuning π0.5 on 500 more LIBERO demos takes 4 GPU-hours. One DAgger iteration needs 2,000 π0.5 rollouts of 300 steps, with replanning every 5 steps, at 80 ms per batched chunk call at batch 32. Estimate the rollout GPU-hours per DAgger iteration, ignoring simulator time, and say which term of \(t_{\text{iter}}\) dominates.

---

## References

* CS 285 Lecture 2 "Supervised Learning of Behaviors" — [slides and video](https://rail.eecs.berkeley.edu/deeprlcourse/)
* Ross & Bagnell, "Efficient Reductions for Imitation Learning," AISTATS 2010 — [PMLR](https://proceedings.mlr.press/v9/ross10a.html) (the \(\epsilon T^2\) analysis)
* Ross, Gordon & Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," AISTATS 2011 — [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) (DAgger)
* Laskey et al., "DART: Noise Injection for Robust Imitation Learning," CoRL 2017 — [arXiv:1703.09327](https://arxiv.org/abs/1703.09327)
* Kelly et al., "HG-DAgger: Interactive Imitation Learning with Human Experts," ICRA 2019 — [arXiv:1810.02890](https://arxiv.org/abs/1810.02890)
* Spencer et al., "Feedback in Imitation Learning: The Three Regimes of Covariate Shift," 2021 — [arXiv:2102.02872](https://arxiv.org/abs/2102.02872)
* Bojarski et al., "End to End Learning for Self-Driving Cars," 2016 — [arXiv:1604.07316](https://arxiv.org/abs/1604.07316)
* Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network," Advances in Neural Information Processing Systems 1, 1989 (the original BC-for-driving system)
* Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation," CoRL 2021 — [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)
* ManiSkill3 — [paper](https://arxiv.org/abs/2410.00425), [docs](https://maniskill.readthedocs.io/)
* LIBERO — [paper](https://arxiv.org/abs/2306.03310)

---

## Next in this special course

* Next: [Lecture 03 — Modern Imitation: Multimodality, Action Chunks, Flow Matching](Lecture-03.md)
* Previous: [Lecture 01 — The RL Problem and the Rollout Engine](Lecture-01.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
