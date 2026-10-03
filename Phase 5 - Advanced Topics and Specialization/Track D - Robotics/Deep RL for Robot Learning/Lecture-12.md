# Lecture 12: Exploration, Curricula, Skills, and Multi-Task RL

## Overview

Every Friday you can go to your favorite restaurant, which is reliably good, or try the new place down the street, which might be better or might be awful. Always going to the favorite is **exploitation**. Sometimes trying the new place is **exploration**. If you never explore you never find out the new place is better, and if you always explore you never enjoy the good meal you already know about.

For a robot with a sparse 0/1 reward the problem is harsher than choosing a restaurant. If the arm never stumbles onto "bowl on plate" by chance, every episode scores 0, every advantage is 0, and **the algorithm learns nothing at all**. No amount of compute fixes a zero gradient. This lecture covers the tools for getting *some* signal:

* **bonuses for novelty**, so the robot is paid to go somewhere new (RND)
* **curricula**, so it practices where it sometimes succeeds and moves to harder setups as it improves
* **skills without a score**, so it builds a repertoire before any task exists (DIAYN, Skew-Fit)
* **hindsight**, so a failed attempt at one goal counts as a success at another (HER)
* **multi-task balance**, so the easy tasks don't crowd out the hard ones and every task group gets reported separately

From the hardware side, this lecture is about **compute allocation**. A rollout that produces no gradient is a GPU-second thrown away, and in regime C each one is a VLA forward pass. A curriculum is a scheduling policy for that compute.

By the end you should be able to:

* compute what fraction of rollouts carry no signal for a given success rate and group size, and design a sampler that reduces it
* implement RND and explain why its bonus shrinks with visits and why it survives pixel observations where counts fail
* build an initial-state curriculum from demonstration states in a GPU simulator
* implement HER relabeling and say why it requires an off-policy learner
* write down DIAYN's objective and the reward it induces
* run a multi-task sampler driven by per-task success and report per-task results

---

## 1. Why it matters: no success, no signal

With a binary success reward and the group baseline from Lectures 04 and 11, a group of \(G\) episodes from a layout with success probability \(p\) carries signal only if its outcomes are mixed:

$$
P(\text{informative group}) = 1 - p^G - (1-p)^G
$$

| Per-layout success \(p\) | 0.01 | 0.1 | 0.5 | 0.9 | 0.99 |
|---|---|---|---|---|---|
| Informative groups, \(G = 8\) | ~8% | ~57% | > 99% | ~57% | ~8% |

The curve is symmetric. A layout the policy almost always solves is exactly as useless for learning as one it never solves. The per-episode reward variance \(p(1-p)\) peaks at \(p = 0.5\), and that is where policy-gradient signal lives. Call layouts with \(0 < p < 1\) **goldilocks** layouts. With sparse binary rewards, only goldilocks layouts carry gradient signal.

Random exploration does not fix the \(p \approx 0\) end. If success needs \(n\) roughly independent "right moves" (reach, align, close gripper, lift, move, release) and dithering gets each one right with probability \(q\), then \(p \approx q^n\), which falls exponentially with task length. Adding Gaussian noise to a 7-D action rarely produces a deliberate grasp.

**Robot connection.** Base π0.5 on nominal LIBERO layouts sits near the \(p \approx 1\) end, which gives little signal. On aggressively perturbed layouts some setups sit near \(p \approx 0\), also little signal. The capstone's gradient comes from the band in between, so finding that band and keeping rollouts in it is the whole game.

---

## 2. Mental model: undirected vs directed exploration

| Strategy | What it does | Fails when |
|---|---|---|
| ε-greedy / Boltzmann | Random action with probability ε, or sample \(\propto \exp(Q/T)\) | Reward is many coordinated steps away |
| Gaussian action noise / entropy bonus | Jitter around the current action (PPO std, SAC's α) | Same: jitter is local, a random walk covers distance ~ \(\sqrt{T}\) |
| Optimism (UCB, count bonuses) | Pretend untried things are good until proven otherwise | States never repeat (pixels, continuous state) |
| Novelty (pseudo-counts, RND, ICM) | Learned "have I seen this?" signal as bonus reward | Novelty ≠ task progress; noisy observations |
| Curricula / demo resets | Change *where* episodes start, not the policy | Requires a resettable simulator or demos |

### 2.1 Optimism and counts

For bandits, **UCB** picks \(\arg\max_a \hat\mu_a + c\sqrt{\ln t / N_a}\): the bonus is large for rarely tried arms and shrinks as \(N_a\) grows. The MDP version adds \(r^+(s) = \beta / \sqrt{N(s)}\) to the reward. With continuous state or pixels, \(N(s)\) is 0 or 1 forever (you never see the same image twice), so you need a *generalizing* notion of count.

**Pseudo-counts** (Bellemare et al.) derive one from a density model \(\rho\): if \(\rho(s)\) is the model's probability of \(s\) before training on it and \(\rho'(s)\) after one update on \(s\), then

$$
\hat N(s) = \frac{\rho(s)\,\big(1 - \rho'(s)\big)}{\rho'(s) - \rho(s)}
$$

which reduces to the true count for an empirical-frequency model.

### 2.2 Random Network Distillation (RND)

A cheaper novelty signal: fix a **randomly initialized** network \(f\) and train a predictor \(\hat f_\phi\) to match it on visited states. The bonus is the prediction error:

$$
r^{\text{int}}(s) = \big\lVert \hat f_\phi(s) - f(s) \big\rVert^2
$$

On states seen often, the predictor has been trained there and the error is small. On new states it has not, and the error is large. That is the guessing game: a bad guess means a new place, which earns a curiosity bonus, and the bonus shrinks with visits automatically. Because \(f\) is a deterministic function of the observation, the target is learnable in principle. That distinguishes RND from forward-model curiosity (**ICM**, prediction error of the next state's features), which stays high forever on genuinely random transitions (the "noisy TV" problem). Practical details that matter: normalize observations, normalize the intrinsic reward by a running std, and keep intrinsic and extrinsic value estimates separate (the RND paper uses two value heads).

**Robot connection.** RND on the full state is dominated by arm joint configurations: the arm can be "novel" by flailing. RND on the *object* state (cube or bowl pose) pays the robot for moving objects to new places, which is much closer to task progress. What you feed the novelty model is a design decision. Lab 2 measures it.

---

## 3. Curricula: spend rollouts where the signal is

### 3.1 Success-rate-adaptive sampling

Keep a running success estimate \(\hat p_i\) for every initial state (seed) or task \(i\), and sample proportionally to its signal:

$$
w_i \propto \hat p_i (1 - \hat p_i) + \epsilon
$$

The floor \(\epsilon\) keeps "solved" and "hopeless" items occasionally sampled, so their estimates can change. The expected **wasted-rollout fraction** under a sampler \(w\) with group size \(G\) is

$$
\text{waste}(w) = \sum_i w_i \big(p_i^G + (1 - p_i)^G\big)
$$

and that is the number to put in your plot. Uniform sampling over mostly-solved layouts can waste half the budget. This sampler is how the "start on setups it sometimes solves, then go harder" idea gets implemented.

### 3.2 Start-state curricula

**Reverse curriculum generation** (Florensa et al.) starts episodes near the goal and moves the start states backward as the policy succeeds from them. In a GPU simulator that can save and restore state, you can do this with a demonstration: record the state at every step of a scripted-expert trajectory ([Lecture 02](Lecture-02.md)), then reset episodes to "\(k\) steps before the end" and grow \(k\) whenever success from that offset exceeds a threshold. You turn one long sparse problem into a sequence of short ones, each in the goldilocks band.

### 3.3 Demonstration-augmented RL

When some setups are never solved, show the robot first. Nair et al. combine three mechanisms on top of DDPG + HER: demonstrations kept in the replay buffer, an auxiliary BC loss with a **Q-filter** (apply BC only where the critic rates the demo action above the policy's), and resets to demonstration states. The result solves block-stacking tasks that neither pure RL nor BC solves. The VLA analogs are the RLPD-style symmetric sampling from [Lecture 10](Lecture-10.md) and advantage-conditioned training on interventions from [Lecture 11](Lecture-11.md).

**Robot connection.** In simulation you have a privileged scripted teacher, so "show demos first for never-solved setups" costs GPU time, not human time. Run the teacher on the perturbed layouts where π0.5 sits at \(p \approx 0\), fine-tune on those demos to push the layouts into the goldilocks band, then let RL take over.

---

## 4. Skills without a score

Before a task exists, a robot can still practice: a playground before the game. Two objectives make that concrete.

### 4.1 DIAYN: skills that are distinguishable

Sample a skill \(z \sim p(z)\) (uniform over \(n\) discrete skills), run \(\pi(a \mid s, z)\), and train a discriminator \(q_\phi(z \mid s)\) to guess which skill produced each visited state. DIAYN maximizes

$$
\mathcal{F} = I(S; Z) + \mathcal{H}(A \mid S) - I(A; Z \mid S) = \mathcal{H}(Z) - \mathcal{H}(Z \mid S) + \mathcal{H}(A \mid S, Z)
$$

Lower-bounding \(-\mathcal{H}(Z \mid S)\) with the discriminator gives a per-step reward, optimized with SAC (max-entropy, Lecture 07):

$$
r_z(s) = \log q_\phi(z \mid s) - \log p(z)
$$

A skill is rewarded when the judge can tell from the states alone which skill it is, so the skills spread apart. Mutual information splits as \(I(S;Z) = \mathcal{H}(S) - \mathcal{H}(S \mid Z)\): **coverage** (visit many states overall) plus **control** (each skill reliably goes to its own region).

### 4.2 Skew-Fit: practice rare places as goals

For goal-conditioned policies, sample practice goals from previously visited states, reweighted toward rare ones: weight \(\propto \hat p(s)^\alpha\) with \(\alpha \in [-1, 0)\) under a learned density \(\hat p\). The paper shows the goal distribution converges toward uniform over valid states, which maximizes coverage, and demonstrates door opening on a real robot from pixels without a hand-designed reward.

**Robot connection.** Practicing more on unusual table setups is the Skew-Fit idea applied to initial states. Broad practice is insurance against perturbation tests you have not seen.

---

## 5. Goals, hindsight, and successor features

**Goal-conditioned RL.** Condition the policy on a goal, \(\pi(a \mid s, g)\), with reward \(r = \mathbb{1}\big[\lVert \text{achieved}(s') - g \rVert < \varepsilon\big]\). For PushCube, the goal is the target position and "achieved" is the cube position.

**Hindsight Experience Replay (HER).** The arm aimed for the red target but pushed the cube somewhere else. That episode failed for goal \(g\), but it is a perfect demonstration for the goal "wherever the cube ended up". HER stores each transition again with substituted goals \(g'\) taken from states achieved **later in the same episode** (the "future" strategy) and recomputes the reward for \(g'\):

```text
for each sampled transition (s_t, a_t, s_{t+1}, g) from episode e:
    with prob p_relabel: g' ← achieved(s_{t'}) for random t' in (t, T]   else g' ← g
    r' ← 1[ ||achieved(s_{t+1}) − g'|| < eps ]
    train Q / actor on (s_t‖g', a_t, r', s_{t+1}‖g')
```

The relabeled transition was **not** generated by \(\pi(\cdot \mid s, g')\), so HER needs an off-policy learner (DDPG, TD3, SAC). PPO and GRPO cannot use it without importance weights. Relabeling is an implicit curriculum: early on, goals near where the cube happens to go are easy, so the policy gets signal from step one.

**Successor features.** If rewards are linear in features, \(r = \phi(s, a, s')^\top w\), then any policy's Q-function factors as \(Q^\pi_w(s,a) = \psi^\pi(s,a)^\top w\) with \(\psi^\pi = \mathbb{E}_\pi[\sum_t \gamma^t \phi_t]\). A new task (a new \(w\)) is evaluated without new rollouts, and **generalized policy improvement** acts greedily with respect to the best of several old policies: \(\pi(s) = \arg\max_a \max_i \psi^{\pi_i}(s,a)^\top w\). It works well when tasks share dynamics and differ only in reward.

---

## 6. Multi-task RL, hierarchy, and meta-RL

**Telling the robot which task.** A multi-task policy conditions on a task identifier: a one-hot vector, a goal, or (for π0.5) a language instruction. Three things go wrong when tasks share one network:

| Problem | Symptom | Fix |
|---|---|---|
| Reward-scale imbalance | Tasks with large or dense returns dominate the gradient | Per-task return/advantage normalization (PopArt adapts it automatically) |
| Gradient conflict | Improving task A hurts task B; per-task gradients have negative cosine | Gradient surgery (PCGrad projects out conflicting components), or more capacity |
| Sampling imbalance | Easy tasks soak up rollouts long after they are solved | Success-rate-adaptive task sampling (Section 3.1) |

Under GRPO with binary rewards, solved tasks drop out of the *gradient* automatically (zero-variance groups), but they still consume *rollouts* unless the sampler stops picking them. With PPO and a learned critic they keep contributing low-signal noise.

**Report each group separately.** An average over suites hides regressions. Improving LIBERO-Spatial by 4 points while losing 3 on LIBERO-Long shows up as a small gain in the mean. [Lecture 13](Lecture-13.md) adds confidence intervals per group.

**Hierarchy.** A high-level policy picks sub-goals ("open the drawer", "pick up the bowl") and a low-level policy executes motions. π0.5 is structured this way: at inference it first predicts a high-level semantic subtask in language, then the action expert generates low-level actions conditioned on that subtask, both from the same model. For RL this raises a choice: reward the low level for subtask completion (denser, needs subtask predicates) or the whole stack for task success (sparser, simpler). The hierarchy also shortens the horizon each level has to plan over.

**Meta-RL: learning to learn.** Train across a distribution of tasks so that the policy adapts quickly to a new one. **RL²** puts the "learning algorithm" in an RNN's hidden state carried across episodes. **MAML** learns an initialization that adapts in a few gradient steps. **PEARL** infers a probabilistic latent task variable from recent experience and conditions an off-policy actor-critic on it. The VLA analog is in-context adaptation from a history of recent frames. For this course it is a pointer, not a tool.

**Robot connection.** π0.5 is a multi-task, language-conditioned policy. The capstone must balance rollouts across LIBERO suites and perturbation types, and it must report each group's score separately, nominal and perturbed.

---

## 7. The hardware view: curricula are compute allocation

* **Wasted rollouts are the metric.** Track the fraction of rollouts that produce zero policy gradient: zero-variance groups for group-baseline methods, zero-return episodes for single-episode methods. In regime C each wasted episode is \(T/H\) VLA forward passes. Cutting waste from 50% to 15% is about a 1.7× effective speedup with no kernel work. Compare with what a week of inference optimization buys you.
* **Novelty models add networks to the loop.** RND is two forward passes per env step plus a predictor update per minibatch. On state observations that is negligible next to the simulator. On pixels it is a second CNN pair. At VLA scale, compute the bonus on an embedding you already have (pooled vision features) rather than running a new image encoder.
* **Hindsight is learner-side and cheap if vectorized.** Relabeling and reward recomputation are tensor ops on the replay sample. Store episodes contiguously on the GPU, plus achieved goals, so relabeling never touches the host. The extra cost shows up in replay memory (full episodes, not single transitions).
* **State resets are a simulator feature.** Demo-state curricula need save/restore of the full simulator state. ManiSkill3 exposes `get_state_dict` / `set_state_dict` (with per-env indexing). On a real robot this is the expensive part, which is why curricula are a simulation advantage.
* **Multi-task GPU sims cost memory.** A GPU-parallel sim instance usually holds one task type. Training on \(m\) tasks means \(m\) instances (memory scales with \(m\)) or time-slicing between them. Per-task evaluation multiplies evaluation cost by \(m\) at fixed confidence (Lecture 13).
* **Sampler bookkeeping is free.** A success table over 10⁴ seeds is a few kB, so there is no excuse to sample uniformly.

---

## 8. Build it

Work on rung 2. All environments use `reward_mode="sparse"` (ManiSkill3 then rewards success only).

### Lab 1 — Make PPO fail

Run your [Lecture 06](Lecture-06.md) PPO on `gym.make("PickCube-v1", num_envs=1024, obs_mode="state", reward_mode="sparse")` with a fixed budget (e.g., 20 M env steps). Log success and the **zero-return episode fraction**. If PPO still solves it within budget, make the problem harder (`StackCube-v1`, or a smaller budget) until it doesn't. That failing configuration is the control for every other lab.

### Lab 2 — RND

```python
# explore/rnd.py
import torch, torch.nn as nn

def mlp(i, o): return nn.Sequential(nn.Linear(i, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, o))

class RND(nn.Module):
    def __init__(self, in_dim, feat=128):
        super().__init__()
        self.target, self.pred = mlp(in_dim, feat), mlp(in_dim, feat)
        self.target.requires_grad_(False)
        self.register_buffer("mu", torch.zeros(in_dim)); self.register_buffer("var", torch.ones(in_dim))
        self.register_buffer("r_var", torch.ones(()))

    def update_stats(self, x, m=0.01):                      # running obs + bonus normalization
        self.mu.lerp_(x.mean(0), m); self.var.lerp_(x.var(0) + 1e-6, m)

    def error(self, x):
        z = ((x - self.mu) / self.var.sqrt()).clamp(-5, 5)
        return (self.pred(z) - self.target(z)).pow(2).mean(-1)

    @torch.no_grad()
    def bonus(self, x, m=0.01):
        e = self.error(x)
        self.r_var.lerp_(e.var() + 1e-8, m)
        return e / self.r_var.sqrt()
```

In the PPO rollout, call `bonus(feats)` each step and use \(r = r^{\text{ext}} + c\, r^{\text{int}}\), with \(c \in \{0.01, 0.1, 1\}\). After each rollout, take a few optimizer steps on `rnd.error(feats).mean()` (gradients reach only `pred`). Ablate the input `feats`: the full state vector vs only the object pose slice. Plot mean bonus over training. It should decay on familiar states and spike when the cube first moves.

### Lab 3 — Initial-state curriculum from demo states

Record a scripted-expert trajectory per seed with your L02 expert, saving `env.unwrapped.get_state_dict()` at every step. Then reset into the tail of the demonstration and walk backward:

```python
# explore/reverse_curriculum.py
def reset_into_demo(env, seeds, demo_states, back):
    """demo_states[t]: batched state dict at step t of expert demos recorded for these seeds."""
    env.reset(seed=seeds)
    t = max(0, len(demo_states) - 1 - back)
    env.unwrapped.set_state_dict(demo_states[t])
    return env.unwrapped.get_obs()

back, step = 2, 2
for it in range(iters):
    obs = reset_into_demo(env, seeds, demo_states, back)
    succ = run_ppo_iteration(obs, ...)                      # your L06 loop, starting from obs
    if succ.mean() > 0.6:
        back += step                                        # move the start further from the goal
    log(back=back, success=succ.mean())
```

Plot `back` vs iteration. The run is finished when `back` reaches the full demo length and success holds from true initial states.

### Lab 4 — HER + SAC on goal-conditioned PushCube

Use your [Lecture 07](Lecture-07.md) SAC on `PushCube-v1`, sparse. Find which entries of the flat state observation hold the goal position and which hold the cube position by reading the task's observation code in the ManiSkill source, and take the success threshold from the task's `evaluate`. Store fixed-length episodes on the GPU:

```python
# explore/her.py  — obs: (E, T+1, D), act: (E, T, A), ach: (E, T+1, 3) cube xyz per step
def sample_her(obs, act, ach, n_eps, B, goal_idx, eps, p_relabel=0.8):
    T, dev = act.shape[1], act.device
    e = torch.randint(0, n_eps, (B,), device=dev)
    t = torch.randint(0, T, (B,), device=dev)
    o, o2, a = obs[e, t].clone(), obs[e, t + 1].clone(), act[e, t]
    fut = t + 1 + (torch.rand(B, device=dev) * (T - t)).long()       # t' in [t+1, T]
    g = torch.where((torch.rand(B, device=dev) < p_relabel)[:, None], ach[e, fut], o[:, goal_idx])
    o[:, goal_idx], o2[:, goal_idx] = g, g
    # PushCube's evaluate() tests xy distance only (goal z ≠ cube-centre z): compare xy, with its threshold.
    r = ((ach[e, t + 1, :2] - g[:, :2]).norm(dim=-1) < eps).float()
    return o, a, r, o2                                      # treat time-limit ends as truncation: always bootstrap
```

Compare SAC with `p_relabel = 0` (plain) vs `0.8` (HER). Plot success vs env steps and the fraction of sampled transitions with \(r = 1\).

### Lab 5 — Goldilocks sampler for seeds and tasks

```python
# explore/goldilocks.py
class GoldilocksSampler:
    def __init__(self, n, floor=0.05, decay=0.995, device="cuda"):
        self.s = torch.zeros(n, device=device); self.f = torch.zeros(n, device=device)
        self.floor, self.decay = floor, decay

    def p(self):  return (self.s + 1) / (self.s + self.f + 2)          # Beta(1,1) posterior mean

    def weights(self):
        w = self.p() * (1 - self.p())
        return (1 - self.floor) * w / w.sum() + self.floor / len(w)

    def sample(self, k):  return torch.multinomial(self.weights(), k, replacement=True)

    def update(self, idx, succ):                                       # decayed counts track a moving policy
        self.s.mul_(self.decay); self.f.mul_(self.decay)
        self.s.index_add_(0, idx, succ); self.f.index_add_(0, idx, 1 - succ)
```

(a) **Seeds:** with your [Lecture 11](Lecture-11.md) GRPO setup, draw `init_seeds` from a pool of 4096 seeds (including perturbed-pose seeds) via `sample(num_groups)` vs uniformly. Log the zero-variance group fraction and success on held-out seeds.
(b) **Tasks:** train one state-based policy on `PickCube-v1`, `PushCube-v1`, `StackCube-v1` (pad observations to a common size and append a one-hot task id; one env instance per task). Each iteration, choose which task to collect from with the sampler vs uniformly. Report **per-task** success curves and the worst-task success, not only the mean.

---

## 9. Use it in the real stack

* [CleanRL](https://github.com/vwxyzjn/cleanrl) has a single-file PPO + RND variant to diff against your Lab 2.
* [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) ships a HER replay buffer for its off-policy algorithms. Read it to check your relabeling logic.
* [Isaac Lab](https://github.com/isaac-sim/IsaacLab) manager-based environments include curriculum terms, the standard way terrain curricula are run for legged robots.
* [ManiSkill3](https://github.com/haosulab/ManiSkill) provides demonstration datasets and motion-planning tooling for generating expert trajectories, plus the state save/restore used in Lab 3.
* For the capstone: put the goldilocks sampler over (LIBERO task, perturbation type, seed) triples in front of whatever VLA RL framework you use.

---

## 10. Measure it

| Metric | Definition | Why it matters |
|---|---|---|
| Wasted-rollout fraction | Rollouts in zero-variance groups (or zero-return episodes) / total | The compute-allocation metric; compare with \(\sum_i w_i(p_i^G + (1-p_i)^G)\) |
| Success vs env steps and vs wall-clock | Per method, ≥ 3 seeds | Bonuses and samplers cost compute; check they pay for themselves |
| Intrinsic/extrinsic ratio | Mean \(c\,r^{\text{int}}\) vs mean \(r^{\text{ext}}\) | Novelty swamping the task reward is a common failure |
| Curriculum frontier | `back` offset (Lab 3), or mean \(\hat p\) of sampled items (Lab 5) | Is the curriculum actually advancing? |
| HER positive-reward fraction | Fraction of sampled transitions with \(r = 1\) | Direct measure of how much signal relabeling creates |
| Per-task / per-group success | Separately for each task, suite, and perturbation type | Averages hide regressions |
| Overhead | Extra ms per env step (RND), replay memory (HER), per-task instances (multi-task) | The hardware cost of each fix |

---

## 11. Ship it

Commit `explore/` containing:

* `rnd.py`, `reverse_curriculum.py`, `her.py`, `goldilocks.py`, plus run scripts for Labs 1-5
* `results.csv`: one row per (method, seed, iteration) with every metric above
* `WASTED_ROLLOUTS.md`: the before/after table of wasted-rollout fraction (uniform vs goldilocks; with and without curriculum), with the measured success curves, and one paragraph converting the saving into GPU-hours for a regime C run using your Lecture 01 numbers
* `per_task.png`: per-task success curves, uniform vs adaptive sampling
* `rnd_input_ablation.png`: full state vs object-only RND input

---

## Exit criteria

You can move on when you can:

* compute the informative-group fraction for any \(p\) and \(G\), and explain why \(p \approx 1\) is as bad as \(p \approx 0\)
* explain why RND's bonus decays with visits and why it avoids the noisy-TV failure that forward-model curiosity has
* show a sparse-reward task where your PPO baseline fails and at least one of RND / curriculum / HER succeeds
* explain why HER needs an off-policy learner
* show a measured drop in wasted-rollout fraction from adaptive sampling, with per-task results reported separately

---

## Self-check

1. Your GRPO run on LIBERO uses \(G = 8\). On 60% of layouts base π0.5 succeeds 97% of the time, and on the rest about 50%. Estimate the wasted-rollout fraction under uniform sampling. What does the goldilocks sampler change, and why keep a floor \(\epsilon > 0\)?
2. You add RND on the full state to sparse PickCube. Success stays at zero, but the intrinsic reward stays high and the arm sweeps wildly. Diagnose and propose two changes.
3. A teammate wants to add HER to the PPO pipeline "because it worked for SAC." What breaks, and what would you need to make it correct?
4. Your multi-task policy's mean success rose from 70% to 74%, but one suite dropped from 65% to 52%. Name two mechanisms from Section 6 that could cause this and the measurement that tells them apart.
5. For a never-solved perturbed layout, compare (a) RND, (b) a reverse curriculum from a privileged scripted teacher's demo, (c) fine-tuning on teacher demos, in terms of simulator features required, GPU cost, and what the final policy weights learn.
6. DIAYN skills on a tabletop all end up as different arm poses that never touch the cube. Using \(I(S;Z) = \mathcal{H}(S) - \mathcal{H}(S \mid Z)\), explain why this is a valid optimum, and how you would change the discriminator's input to get object-centric skills.

---

## References

* CS 285 lectures on exploration, unsupervised skill discovery, and multi-task / meta-RL — [course site](https://rail.eecs.berkeley.edu/deeprlcourse/)
* Burda et al., "Exploration by Random Network Distillation", 2018 — [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)
* Bellemare et al., "Unifying Count-Based Exploration and Intrinsic Motivation", 2016 — [arXiv:1606.01868](https://arxiv.org/abs/1606.01868); Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction" (ICM), 2017 — [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)
* Florensa et al., "Reverse Curriculum Generation for Reinforcement Learning", 2017 — [arXiv:1707.05300](https://arxiv.org/abs/1707.05300)
* Nair et al., "Overcoming Exploration in Reinforcement Learning with Demonstrations", 2017 — [arXiv:1709.10089](https://arxiv.org/abs/1709.10089)
* Eysenbach et al., "Diversity is All You Need: Learning Skills without a Reward Function" (DIAYN), 2018 — [arXiv:1802.06070](https://arxiv.org/abs/1802.06070)
* Pong et al., "Skew-Fit: State-Covering Self-Supervised Reinforcement Learning", 2019 — [arXiv:1903.03698](https://arxiv.org/abs/1903.03698)
* Andrychowicz et al., "Hindsight Experience Replay", 2017 — [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)
* Barreto et al., "Successor Features for Transfer in Reinforcement Learning", 2016 — [arXiv:1606.05312](https://arxiv.org/abs/1606.05312)
* Hessel et al., "Multi-task Deep Reinforcement Learning with PopArt", 2018 — [arXiv:1809.04474](https://arxiv.org/abs/1809.04474); Yu et al., "Gradient Surgery for Multi-Task Learning" (PCGrad), 2020 — [arXiv:2001.06782](https://arxiv.org/abs/2001.06782)
* Physical Intelligence, "π0.5: a Vision-Language-Action Model with Open-World Generalization" (hierarchical subtask → action inference), 2025 — [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
* Meta-RL: Duan et al., "RL²", 2016 — [arXiv:1611.02779](https://arxiv.org/abs/1611.02779); Finn et al., "MAML", 2017 — [arXiv:1703.03400](https://arxiv.org/abs/1703.03400); Rakelly et al., "PEARL", 2019 — [arXiv:1903.08254](https://arxiv.org/abs/1903.08254)

---

## Next in this special course

* Next: [Lecture 13 — Theory, Evaluation Rigor, and Sim-to-Real](Lecture-13.md)
* Previous: [Lecture 11 — RL for LLMs and VLAs](Lecture-11.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
