# Lecture 06b: Actor-Critic PPO in Practice: Inside RSL-RL

## Overview

[Lecture 05](Lecture-05.md) built the actor-critic: a policy (the player) and a value function (the coach), tied together by GAE. [Lecture 06](Lecture-06.md) added PPO's clipped surrogate and the trust-region argument behind it. This lecture opens the code that turns those equations into walking robots: **RSL-RL**, the PPO library from ETH Zurich's Robotic Systems Lab and NVIDIA. It is the default RL library in Isaac Lab, and it is also used by legged_gym, mjlab, and MuJoCo Playground. When a legged-locomotion or manipulation paper says "we train with PPO", this is usually the PPO it means.

RSL-RL is small on purpose. The whole library is about 4,800 lines, and the algorithm itself, `rsl_rl/algorithms/ppo.py`, is under 500. You can read all of it in an afternoon. This lecture reads it with you, line by line against the equations from Lectures 05 and 06, then trains it on a batched pendulum you write yourself and runs the ablations that show which of its choices matter.

**Version pin:** `rsl-rl-lib==5.5.1` (tag `v5.5.1`, September 2026), the version Isaac Lab 3.0 pins. Line numbers below refer to that tag. The 4.x and 5.x releases reorganized the library: older code and tutorials use a single `ActorCritic` module and a `policy=` config, which you will learn to recognize in Section 9.

By the end you should be able to:

* draw RSL-RL's runner → algorithm → models → storage structure and say what each piece owns
* point to the exact lines that implement GAE, the clipped surrogate, the clipped value loss, and the adaptive learning rate
* explain the time-out bootstrapping trick, what approximation it makes, and when Isaac Lab turns it on
* configure an asymmetric actor-critic with observation groups and per-model normalization
* contrast RSL-RL's defaults with a CleanRL-style PPO and predict which differences matter for a given task
* train RSL-RL on a custom vectorized environment and read its training logs

---

## 1. Why it matters: one library, most robot PPO

The same 500 lines of PPO train ANYmal and Unitree quadrupeds, humanoids, and Franka manipulation tasks across several simulators. Three reasons to know it well:

* **Reproducing papers.** Most published locomotion results inherit RSL-RL's defaults: adaptive learning rate, clipped value loss, separate actor and critic networks, a short rollout over thousands of environments. If you reimplement PPO "from the paper", you will not match their curves until you match these choices.
* **Debugging your own training.** When an Isaac Lab run plateaus, the fix is usually in this file: a value target off by a time-out, a reward scale fighting the value clip, a learning rate the KL rule drove to its floor. You need to know which line is responsible.
* **Extending it.** The library's purpose is to be modified. Asymmetric critics, symmetry augmentation, curiosity, and student-teacher distillation are all small, readable additions to the same loop.

---

## 2. Mental model: four pieces and a loop

```text
 OnPolicyRunner.learn()                                   rsl_rl/runners/on_policy_runner.py
 └─ for it in iterations:
      ├─ with torch.inference_mode():                     ── ROLLOUT (collect_time)
      │    for t in range(num_steps_per_env):
      │        a = alg.act(obs)                  actor samples a; critic predicts V(s); log π(a|s) stored
      │        obs, r, done, extras = env.step(a)
      │        alg.process_env_step(obs, r, done, extras)   time-out bootstrap; store transition
      │    alg.compute_returns(obs)              GAE, backward over the T stored steps
      └─ alg.update()                                     ── LEARN (learn_time)
           for epoch × minibatch: recompute log π, V, H → adapt lr from KL → loss → clip grads → Adam step
           then update observation normalizers; clear storage

 PPO                        rsl_rl/algorithms/ppo.py         the algorithm
 MLPModel ×2                rsl_rl/models/mlp_model.py       actor (+ GaussianDistribution), critic
 RolloutStorage             rsl_rl/storage/rollout_storage.py  T × N tensors, on the training device
 VecEnv                     rsl_rl/env/vec_env.py            get_observations() → TensorDict; step()
```

This is Lecture 01's try → judge → improve loop, with the boxes named. Four structural choices define the library:

1. **The actor and critic are separate networks** (two `MLPModel`s, `ppo.py:92-93`) trained by **one optimizer** over both parameter sets (`ppo.py:100-102`). There is no shared trunk by default. This lets the critic take different inputs from the actor (Section 3). The exception is image encoders, which can be shared with `share_cnn_encoders`.
2. **Everything lives on the training device.** The environment returns GPU tensors, the storage is preallocated on the GPU, and the rollout runs under `torch.inference_mode()`. Nothing crosses to the CPU in the hot loop.
3. **Short rollouts over many environments.** A typical Isaac Lab config collects 24 steps from each of thousands of environments per iteration, then runs 5 epochs × 4 minibatches. Lecture 06 §8.1 explained why that batch shape suits GPU simulation.
4. **Observations are dictionaries.** The environment returns a `TensorDict` of named observation groups, and a config maps groups to models.

---

## 3. Observation groups: the asymmetric critic in one dictionary

[Lecture 05 §6](Lecture-05.md) argued that in simulation the critic should see privileged state the actor cannot: exact object poses, friction coefficients, contact forces. RSL-RL makes that a configuration choice. The environment returns several groups:

```python
obs = TensorDict({"policy": proprio_and_commands, "privileged": sim_state}, batch_size=[num_envs])
```

and the runner config maps **observation sets** (what each model needs) to **groups** (what the environment provides):

```python
"obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]}
```

Each model concatenates its groups in order (`mlp_model.py:113-118`). `resolve_obs_groups` (`utils/utils.py:192`) fills in a missing set: first with a group of the same name, then with the `"policy"` group, and otherwise it raises an error. The sets RSL-RL knows are `actor`, `critic`, `student`, `teacher`, and `rnd_state`.

**Normalization is per model.** With `obs_normalization=True`, each model owns an `EmpiricalNormalization` module (`modules/normalization.py`), a running mean and variance over the whole batch (not per environment). It normalizes as \((x - \mu) / (\sigma + 10^{-2})\). Two details matter:

* The statistics are updated **once per iteration, after** `update()`, from that iteration's rollout (`ppo.py:327-330`), and only in training mode. During one PPO update the normalizer is frozen, so the old and new policies see identically normalized inputs. That keeps the importance ratio meaningful.
* The normalizer is copied into the exported JIT and ONNX policies (`mlp_model.py:196-242`). A deployed policy takes raw observations, so there is no separate normalization step to forget on the robot.

---

## 4. The rollout: `act()` and `process_env_step()`

`act()` (`ppo.py:124-135`) does four things per step: sample an action from the actor's distribution, compute \(V(s_t)\) with the critic, compute \(\log \pi_{\theta_\text{old}}(a_t \mid s_t)\), and store the **distribution parameters** (mean and std for a Gaussian). The parameters are stored so that the update can later compute an exact KL divergence between the old and new policies, not just a sample estimate.

The default action distribution is `GaussianDistribution`: the MLP outputs the mean, and the standard deviation is a **state-independent learned parameter**, stored as a raw scale by default (`std_type="scalar"`) and clamped to a range. Log-probabilities and entropies are summed over action dimensions. Unbounded Gaussian samples go straight to the environment. Isaac Lab's wrapper can clamp them with `clip_actions`, but there is no tanh squashing and no Jacobian correction.

### 4.1 Time-out bootstrapping

`process_env_step()` (`ppo.py:137-164`) contains the most important line in the file for robot tasks:

```python
# Bootstrapping on time outs                                          ppo.py:153-158
if "time_outs" in extras:
    self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1)
```

Robot episodes are cut by a time limit (say 20 s of walking), and the time limit is not a real terminal state. Lecture 05 §5.1 showed that treating a truncation as termination teaches the critic that the world ends at 20 s. RSL-RL's fix is to fold the missing future into the last reward:

$$
\tilde r_t = r_t + \gamma\, V(s_t)\cdot \mathbb{1}[\text{time-out at } t]
$$

and then let the `done` flag cut the GAE recursion as usual. Notice that it uses \(V(s_t)\), the value of the state **before** the step, not \(V(s_{t+1})\). That is an approximation forced by the environment API. GPU environments reset in place, so the observation returned at a time-out is already the next episode's first state, and the true \(s_{t+1}\) is gone. Using the previous state's value is accurate when \(V\) changes slowly over one control step, which is usually true at 50 Hz.

Isaac Lab's `RslRlVecEnvWrapper` sets `dones = terminated | truncated`, and it reports `extras["time_outs"] = truncated` **only for infinite-horizon tasks** (`cfg.is_finite_horizon == False`). For a task that really ends at the time limit (the robot is scored at the final step), the time-out is a real terminal state and must not be bootstrapped. Lab 6b-c measures what happens when bootstrapping is missing.

---

## 5. Returns: GAE in sixteen lines

`compute_returns()` (`ppo.py:166-191`) is Lecture 05's GAE written as a backward loop:

```python
for step in reversed(range(T)):
    next_values = last_values if step == T - 1 else values[step + 1]
    next_is_not_terminal = 1.0 - dones[step].float()
    delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]    # TD error
    advantage = delta + next_is_not_terminal * gamma * lam * advantage                  # GAE recursion
    returns[step] = advantage + values[step]                                            # value target
advantages = returns - values
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)               # whole batch
```

That is exactly

$$
\delta_t = \tilde r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t), \qquad
\hat A_t = \delta_t + \gamma \lambda (1 - d_t)\hat A_{t+1}, \qquad
\hat R_t = \hat A_t + V(s_t)
$$

with the critic's value of the final observation, `last_values`, bootstrapping the end of the rollout. That final bootstrap is what makes 24-step rollouts work: GAE never needs an episode to finish. One detail differs from many CleanRL-style implementations. Advantages are normalized **once over the whole batch** by default. Set `normalize_advantage_per_mini_batch=True` to normalize each minibatch instead.

---

## 6. The update: one loss, line by line

`update()` (`ppo.py:193-358`) loops over minibatches from `mini_batch_generator` (`rollout_storage.py:225-259`). That generator draws **one** random permutation per update and reuses the same minibatch partition in every epoch. The minibatch size is \(N \cdot T / \texttt{num\_mini\_batches}\). For each minibatch it recomputes \(\log \pi_\theta\), \(V_\phi\), and the entropy under the current parameters, then builds four pieces.

### 6.1 The adaptive learning rate (`ppo.py:240-266`)

CleanRL-style PPO uses a fixed or annealed learning rate and optionally stops an epoch early when the KL gets too large. RSL-RL instead **steers the learning rate** with the exact Gaussian KL between the stored old distribution and the current one, measured on every minibatch **before** its gradient step:

$$
\text{lr} \leftarrow
\begin{cases}
\max(10^{-5},\ \text{lr}/1.5) & \text{if } \overline{\mathrm{KL}} > 2\,\mathrm{KL}_\text{target} \\
\min(10^{-2},\ 1.5\,\text{lr}) & \text{if } 0 < \overline{\mathrm{KL}} < \mathrm{KL}_\text{target}/2 \\
\text{lr} & \text{otherwise}
\end{cases}
$$

with `desired_kl=0.01` by default (`schedule="adaptive"`). This is a soft trust region in the spirit of Lecture 06 §5. The step size grows while updates are timid and shrinks when the policy moves too far. It also explains a log pattern you will see: the learning rate rises quickly at the start of training, then oscillates. If it sits at the \(10^{-5}\) floor, something is pushing the policy too hard (a reward spike, a bad normalizer); if it is pinned at \(10^{-2}\), the policy has stopped changing. Set `schedule="fixed"` to disable it.

### 6.2 The clipped surrogate (`ppo.py:268-274`)

```python
ratio = torch.exp(actions_log_prob - old_actions_log_prob)
surrogate = -advantages * ratio
surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

The maximum of the negated terms is the minimum of Lecture 06's clipped objective, so this is exactly \(-L^{\text{CLIP}}\).

### 6.3 The clipped value loss (`ppo.py:276-283`)

$$
V^{\text{clip}} = V_\text{old} + \operatorname{clip}\!\big(V_\phi - V_\text{old},\, -\varepsilon,\, \varepsilon\big), \qquad
L^V = \mathbb{E}\Big[\max\big((V_\phi - \hat R)^2,\ (V^{\text{clip}} - \hat R)^2\big)\Big]
$$

RSL-RL turns this on by default and uses the **same** \(\varepsilon\) (0.2) as the policy clip. Lecture 06 §8 noted that large studies did not find value clipping reliably helpful. In RSL-RL it has a specific side effect you should know: \(\varepsilon\) is in **reward units**, and once a sample's prediction has moved more than 0.2 from its rollout-time value toward the target, the clipped branch has the larger error, the `max` selects it, and that sample stops contributing gradient. The critic can move each prediction by at most about 0.2 per update. With small, well-scaled rewards the cap rarely binds. With returns in the hundreds it binds on many samples, and the critic needs more updates to catch up. Whether that slows *learning* depends on the task. In the Pendulum runs of Section 12, unscaled rewards made no difference beyond seed noise, with or without value clipping, because advantage normalization and \(\lambda = 0.95\) make the policy update fairly tolerant of a lagging critic. So treat reward scale as a hyperparameter to check on your own task, using `Loss/value` and the critic's explained variance, not as a known failure.

### 6.4 Total loss, gradients, step (`ppo.py:285-311`)

$$
L = -L^{\text{CLIP}} + c_V\, L^V - c_H\, \mathcal{H}[\pi_\theta]
$$

with `value_loss_coef` \(c_V = 1.0\) and `entropy_coef` \(c_H\) (0.01 class default; 0.005 in Isaac Lab's ANYmal-D config). One backward pass covers both networks. Then each network's gradient is clipped **separately** to `max_grad_norm` (1.0), and a single Adam step updates both. Separate clipping keeps a large critic gradient, common early in training, from shrinking the actor's step.

---

## 7. The defaults, and how they differ

| Choice | RSL-RL 5.5.1 (class default · Isaac Lab ANYmal-D config) | CleanRL-style (ManiSkill `ppo.py`, from Lecture 06 §8) |
|---|---|---|
| Networks | Separate actor and critic, ELU · [512, 256, 128] | Separate actor and critic MLPs |
| Rollout shape | `num_steps_per_env` 24, thousands of envs | 50 steps × 512 envs |
| Epochs × minibatches | 5 × 4 | 4 × 32 |
| Learning rate | 1e-3, **adaptive** from KL (target 0.01) | 3e-4 fixed, early stop at `target_kl` 0.1 |
| Advantage normalization | Whole batch (default) | Per minibatch |
| Value loss | **Clipped**, coefficient 1.0 | Unclipped, coefficient 0.5 |
| Entropy coefficient | 0.01 · 0.005 | 0.0 |
| Gradient clipping | 1.0 per network | 0.5 global |
| Policy std | State-independent, raw scale, init 1.0 | State-independent log-std, init −0.5 |
| \(\gamma\), \(\lambda\) | 0.99, 0.95 | 0.8, 0.9 (dense reward, 50-step tasks) |
| Time-outs | Bootstrapped through the reward | Bootstrapped from the stored final observation |

Neither column is "correct". The RSL-RL column was tuned on long-horizon locomotion with thousands of environments; the CleanRL column on short manipulation episodes. When porting a task between libraries, change one row at a time and keep your seeds (Lecture 13).

---

## 8. Beyond vanilla PPO: what else is in the box

Each of these is a small, optional addition to the same loop:

| Feature | Where | What it does | Lecture |
|---|---|---|---|
| Recurrent models | `models/rnn_model.py` (LSTM/GRU) | Memory for partial observability. The recurrent minibatch generator splits by **environment**, not by time, so each minibatch holds whole trajectories (size `num_envs // num_mini_batches`) | 03 |
| CNN encoders | `models/cnn_model.py`, `share_cnn_encoders` | Pixel observations, optionally one encoder shared by actor and critic | 05 |
| Other distributions | `HeteroscedasticGaussianDistribution`, `BetaDistribution` | State-dependent std; bounded actions | 04 |
| Symmetry | `extensions/symmetry.py` | Mirror-augmented minibatches and/or a mirror loss (not with recurrent models) | — |
| Curiosity (RND) | `extensions/rnd.py` | Intrinsic reward added in `process_env_step`, its own optimizer | 12 |
| Student-teacher distillation | `algorithms/distillation.py`, `DistillationRunner` | Train a deployable student on onboard observations to imitate a teacher trained with privileged ones | 02, 05 |
| Multi-GPU | `torchrun`; NCCL | One process per GPU: gradients averaged in buckets, KL all-reduced, learning rate broadcast from rank 0, normalizer statistics combined | 12 |
| Speed | `torch_compile_mode`, `use_mixed_precision` (bf16 autocast) | Faster update for larger networks | — |
| Export | `export_policy_to_jit`, `export_policy_to_onnx` | Deterministic (mean) policy with the normalizer included | — |

---

## 9. The hardware view

**Where an iteration's time goes.** The logger writes `Perf/collection_time` (rollout plus GAE), `Perf/learning_time` (the update), and `Perf/total_fps` = \(N T / (t_\text{collect} + t_\text{learn})\). In Lecture 01's regime B (GPU simulator, small MLP), collection is mostly simulator time. Learning is `num_learning_epochs` × `num_mini_batches` = 20 optimizer steps per iteration, each on a minibatch of \(NT/4\) samples. With 4,096 environments × 24 steps, a minibatch is 24,576 samples, large enough to keep a GPU busy even with a small network. Watch the ratio of the two times. If learning dominates, larger networks, more epochs, or more minibatches cost wall-clock directly, and `torch_compile_mode` or bf16 are the levers. If collection dominates, the simulator is the bottleneck, and the PPO settings barely matter for speed.

**Storage memory** is preallocated:

$$
\text{bytes} \approx 4 \cdot T \cdot N \cdot \big(d_\text{obs} + d_a \,(1 + 2) + 5\big)
$$

for float32 observations (all groups, stored once), actions, two distribution-parameter tensors, and the scalars (reward, done, value, log-prob, return, advantage). With \(T = 24\), \(N = 4096\), \(d_\text{obs} = 256\), \(d_a = 12\), that is about 117 MB. That is small next to the simulator's footprint. Rollout storage only becomes a problem with image observations, where Isaac Lab Lecture 10 shows the camera Cartpole buffer reaching gigabytes.

**On an 8 GB card** (the reference GPU for the [Isaac Sim and Isaac Lab](../Isaac%20Sim%20and%20Isaac%20Lab/README.md) course), state-based RSL-RL training is limited by the simulator, not by PPO. The actor, critic, Adam state, and storage for a locomotion task fit in a few hundred MB. Reduce `--num_envs` for the simulator's sake, and expect the KL rule to compensate partly for the smaller batch. Smaller batches give noisier gradients, which tend to produce larger KL per step and push the learning rate down.

---

## 10. Build it

Everything goes in `rslrl_lab/`. You need only PyTorch and `pip install rsl-rl-lib==5.5.1`. The labs run on CPU in minutes, and faster on any GPU.

### Lab 6b-a — Map the equations to the code

Clone the tag and read `ppo.py`, `on_policy_runner.py`, `rollout_storage.py`, and `mlp_model.py` in full:

```bash
git clone --depth 1 --branch v5.5.1 https://github.com/leggedrobotics/rsl_rl.git
```

Fill in `RSLRL_MAP.md`, one row per equation from Lectures 05-06: the equation, the file and lines, and one sentence on any difference from the lecture's version. Starter rows, which you should verify:

| Equation | Where in v5.5.1 |
|---|---|
| Sample \(a \sim \pi_\theta\), store \(V\), \(\log\pi\), distribution params | `ppo.py:124-135` |
| Time-out bootstrap \(\tilde r_t = r_t + \gamma V(s_t)\) | `ppo.py:153-158` |
| GAE and value targets | `ppo.py:166-191` |
| KL-adaptive learning rate | `ppo.py:240-266` |
| Clipped surrogate | `ppo.py:268-274` |
| Clipped value loss | `ppo.py:276-283` |
| Total loss, per-network gradient clipping | `ppo.py:285-311` |
| Normalizer update after the epoch loop | `ppo.py:327-330` |

### Lab 6b-b — RSL-RL on a vectorized environment you wrote

RSL-RL trains anything that implements `VecEnv`: `get_observations()` returning a `TensorDict`, and `step()` returning `(obs, rewards, dones, extras)`. This environment is Gymnasium's Pendulum-v1, vectorized in torch. Episodes **only** end by time-out after 200 steps, which makes it a clean test of Section 4.1.

```python
# rslrl_lab/pendulum_rsl.py — train RSL-RL PPO on a batched, torch-native Pendulum.
import argparse, math, time, torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner


class TorchPendulum(VecEnv):
    """Gymnasium Pendulum-v1 dynamics, vectorized in torch. Episodes only end by time-out (200 steps)."""

    def __init__(self, num_envs, device="cpu", reward_scale=1.0, report_timeouts=True):
        self.num_envs, self.num_actions, self.device = num_envs, 1, device
        self.max_episode_length, self.cfg = 200, {}
        self.reward_scale, self.report_timeouts = reward_scale, report_timeouts
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.th = torch.zeros(num_envs, device=device)
        self.thdot = torch.zeros(num_envs, device=device)
        self._reset(torch.arange(num_envs, device=device))

    def _reset(self, ids):
        self.th[ids] = (torch.rand(len(ids), device=self.device) * 2 - 1) * math.pi
        self.thdot[ids] = torch.rand(len(ids), device=self.device) * 2 - 1
        self.episode_length_buf[ids] = 0

    def get_observations(self):
        obs = torch.stack([torch.cos(self.th), torch.sin(self.th), self.thdot], dim=-1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions):
        u = actions.squeeze(-1).clamp(-2.0, 2.0)
        ang = (self.th + math.pi) % (2 * math.pi) - math.pi                 # angle_normalize
        cost = ang**2 + 0.1 * self.thdot**2 + 0.001 * u**2
        self.thdot = (self.thdot + (3 * 10.0 / 2 * torch.sin(self.th) + 3.0 * u) * 0.05).clamp(-8.0, 8.0)
        self.th = self.th + self.thdot * 0.05
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        self._reset(time_outs.nonzero().squeeze(-1))                       # auto-reset: obs below is post-reset
        extras = {"time_outs": time_outs.float()} if self.report_timeouts else {}
        return self.get_observations(), -cost * self.reward_scale, time_outs.long(), extras


def train_cfg(args):
    return {
        "num_steps_per_env": args.steps, "save_interval": 10_000,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [64, 64], "activation": "elu",
                  "obs_normalization": True,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [64, 64], "activation": "elu",
                   "obs_normalization": True},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4,
                      "clip_param": 0.2, "gamma": 0.99, "lam": 0.95, "learning_rate": 1e-3,
                      "schedule": args.schedule, "desired_kl": 0.01, "entropy_coef": 0.0,
                      "value_loss_coef": 1.0, "use_clipped_value_loss": bool(args.clip_value),
                      "max_grad_norm": 1.0},
    }


@torch.inference_mode()
def evaluate(policy, device, episodes=256):
    env = TorchPendulum(episodes, device)                                   # unscaled reward
    obs, ret = env.get_observations(), torch.zeros(episodes, device=device)
    for _ in range(env.max_episode_length):                                 # exactly one episode per env
        obs, r, _, _ = env.step(policy(obs))                               # deterministic: mean action
        ret += r
    return ret.mean().item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward_scale", type=float, default=0.1)
    p.add_argument("--no_timeouts", action="store_true")        # ablation: treat time-outs as terminal
    p.add_argument("--clip_value", type=int, default=1)
    p.add_argument("--schedule", default="adaptive")             # or "fixed"
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    env = TorchPendulum(args.num_envs, args.device, args.reward_scale, report_timeouts=not args.no_timeouts)
    runner = OnPolicyRunner(env, train_cfg(args), log_dir=None, device=args.device)
    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    wall = time.perf_counter() - t0
    score = evaluate(runner.get_inference_policy(args.device), args.device)
    print(f"RESULT timeouts={not args.no_timeouts} clip_value={bool(args.clip_value)} "
          f"reward_scale={args.reward_scale} schedule={args.schedule} iters={args.iters} seed={args.seed} eval_return={score:.1f} wall={wall:.1f}s")
```

```bash
pip install rsl-rl-lib==5.5.1
python rslrl_lab/pendulum_rsl.py                       # about 2-3 minutes on a laptop CPU
```

The evaluation uses the unscaled Pendulum reward, so scores compare directly with Gymnasium's. A random policy scores around −1,200 per episode, and a policy that swings up and balances scores around −150. Things to notice:

* `init_at_random_ep_len=True` staggers the environments' episode clocks, so time-outs spread evenly across iterations instead of all landing on the same step.
* The rollout (32 steps) is much shorter than an episode (200 steps). The final-value bootstrap and the time-out bootstrap together make that safe.
* Pass a real `log_dir` to `OnPolicyRunner` to get TensorBoard logs (`Loss/surrogate`, `Loss/value`, `Loss/entropy`, `Loss/learning_rate`, `Policy/mean_std`, `Perf/*`, `Train/mean_reward`). Plot the learning rate. You should see the adaptive rule at work.

### Lab 6b-c — Ablations: which choices matter?

Run 3 seeds of each configuration at two budgets, `--iters 30` (learning speed) and `--iters 150` (final performance), and report the mean and spread of the evaluation return:

| Run | Flags | Tests |
|---|---|---|
| Baseline | (defaults) | — |
| No time-out bootstrapping | `--no_timeouts` | Section 4.1: time-outs treated as terminal states |
| Unscaled reward, clipped value | `--reward_scale 1.0` | Section 6.3: the value clip in reward units |
| Unscaled reward, unclipped value | `--reward_scale 1.0 --clip_value 0` | Isolates the value clip from the reward scale itself |
| Fixed learning rate | `--schedule fixed` | Section 6.1: what the KL rule buys |

The runs are independent, so run them in parallel (`xargs -P` with `OMP_NUM_THREADS=1` on CPU). Results from one CPU test run of all five rows are in Section 12. Treat them as a reference point and measure your own.

### Lab 6b-d — The same PPO in Isaac Lab

If you have an RTX GPU with Isaac Lab 3.0 installed ([Isaac Lab Lecture 10](../Isaac%20Sim%20and%20Isaac%20Lab/Lecture-10.md)), train a task whose agent config is the RSL-RL config from Section 7:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD --viz none --num_envs 1024
```

Open `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py` and match each field to the `PPO.__init__` arguments you read in Lab 6b-a. Then open the TensorBoard logs and check that the learning-rate curve shows the same adaptive pattern as in your Pendulum runs. On an 8 GB card, lower `--num_envs` until it fits and note how `Perf/total_fps` and the learning-rate curve change.

---

## 11. Use it in the real stack

* **Isaac Lab 3.0** wraps RSL-RL in `isaaclab_rl.rsl_rl`. A task's agent config is an `RslRlOnPolicyRunnerCfg` with `actor` and `critic` set to `RslRlMLPModelCfg` (or `RslRlRNNModelCfg` / `RslRlCNNModelCfg`). It carries the action distribution as `distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=...)` and an `RslRlPpoAlgorithmCfg` whose fields are the `PPO.__init__` arguments. `RslRlVecEnvWrapper` adapts the environment and applies `clip_actions`.
* **You will see this in older code.** Before RSL-RL 4.0, configs had a single `policy=RslRlPpoActorCriticCfg(init_noise_std=..., actor_hidden_dims=..., critic_hidden_dims=...)` and a runner-level `empirical_normalization` flag. In 5.x the equivalents are separate `actor` and `critic` configs, `distribution_cfg.init_std`, and per-model `obs_normalization`. Isaac Lab 3.0 still accepts the old fields with deprecation warnings.
* **Other users:** legged_gym (Isaac Gym, the original RSL-RL client, on an older API), mjlab (MuJoCo-Warp), and MuJoCo Playground. Your Pendulum `VecEnv` from Lab 6b-b is the template for plugging in any other simulator, including ManiSkill.

---

## 12. Measure it

| Metric | Source | What it tells you |
|---|---|---|
| `Train/mean_reward`, `Train/mean_episode_length` | Logger, from completed episodes | Learning progress (noisy; compare with seeds) |
| `Loss/learning_rate` | Adaptive schedule | Floor means updates are too aggressive; ceiling means the policy has stopped moving |
| `Loss/value` vs `Loss/surrogate` | `update()` | A value loss that stays high usually points to reward scale or missing time-outs |
| `Policy/mean_std` | Gaussian std | Exploration collapsing too fast or never shrinking |
| `Perf/collection_time` vs `Perf/learning_time` | Runner | Whether the simulator or PPO bounds wall-clock time |
| `Perf/total_fps` | \(NT / (t_c + t_l)\) | Throughput to compare across hardware and `num_envs` |
| Evaluation return, ≥ 3 seeds | Lab script | The number to report, with spread (Lecture 13) |

**Reference results from one CPU test run** of Lab 6b-c (512 environments, 32 steps, 150 iterations, rsl-rl-lib 5.5.1, PyTorch 2.14 CPU, 3 seeds). These are one machine's measurements, not benchmarks:

| Configuration | Eval return after 30 iterations | Eval return after 150 iterations |
|---|---|---|
| Baseline | −296 ± 159 (−183, −478, −228) | −149 ± 3 (−147, −153, −148) |
| No time-out bootstrapping | −688 ± 81 (−743, −726, −595) | −149 ± 4 (−146, −153, −148) |
| Fixed learning rate | −760 ± 97 (−680, −733, −868) | −152 ± 4 (−155, −153, −147) |
| Unscaled reward, clipped value | −361 ± 260 (−204, −217, −662) | −152 ± 1 (−151, −152, −154) |
| Unscaled reward, unclipped value | −406 ± 227 (−158, −457, −602) | −148 ± 4 (−144, −152, −147) |

Mean ± sample std over seeds 0, 1, 2, with the individual seeds in parentheses. Evaluation is the deterministic policy over 256 episodes, in Gymnasium Pendulum units (random ≈ −1,200). Each 150-iteration run took about 1 minute with four runs sharing a 4-core CPU.

What the numbers say, and what they don't:

* **Every configuration solves Pendulum by 150 iterations.** On an easy task, none of these choices changes the final result. They change how fast you get there, which is what you pay for in GPU-hours on a hard task.
* **Time-out bootstrapping and the adaptive learning rate both speed up early learning by a wide margin.** Their 30-iteration returns are more than 2× worse than the baseline, and the seed ranges do not overlap. Without time-outs, the critic is taught that the world ends every 200 steps, and its values carry a time dependence it cannot observe. The fixed-rate gap is the adaptive rule at work. In baseline seed 0, it pushed the learning rate from 1e-3 to its 1e-2 ceiling within the first iteration and held it there for 27 iterations, 10× the fixed rate, before backing off to 2e-3 as the KL grew.
* **Reward scale and value clipping showed no effect** beyond seed noise on this task (Section 6.3).
* **Three seeds is a small sample.** The baseline's 30-iteration spread (−183 to −478) shows how much early learning varies. Use more seeds before claiming a difference smaller than these (Lecture 13).

---

## 13. Ship it

Commit `rslrl_lab/` with:

* `RSLRL_MAP.md` — the completed equation-to-code table, with one paragraph on every place where RSL-RL differs from Lectures 05-06
* `pendulum_rsl.py` and `run_ablations.sh`
* `ablations.csv` — one row per (configuration, seed): evaluation return, wall-clock, final learning rate
* `ablations.png` — evaluation return per configuration with seed spread, plus the learning-rate curves of one baseline seed
* `NOTES.md` — half a page: which RSL-RL choice mattered most on Pendulum, why, and whether you expect the same ranking on a locomotion task

---

## Exit criteria

You can move on when you can:

* name the four classes in the RSL-RL loop and the file each lives in
* point to the lines implementing GAE, the clipped surrogate, the clipped value loss, and the adaptive learning rate, and write each as an equation
* explain the time-out bootstrap, why it uses \(V(s_t)\), and why Isaac Lab only reports time-outs for infinite-horizon tasks
* configure an asymmetric actor-critic with `obs_groups`, and say where its normalizers are updated and saved
* read an RSL-RL training log and diagnose a stuck learning rate, a high value loss, or collapsing policy std
* show your ablation results with seed spread and explain the ranking

---

## Self-check

1. Your Isaac Lab locomotion run's `Loss/learning_rate` drops to \(10^{-5}\) within 50 iterations and stays there, while the reward stops improving. Which lines of `ppo.py` are involved, what does this tell you about the per-minibatch KL, and name two things you would check first.
2. A teammate ports a task to RSL-RL and multiplies every reward by 100 "to make the signal stronger". Using Section 6.3, explain how this could slow the critic, which logged metrics would confirm or rule that out, and why the Pendulum runs in Section 12 showed no effect. Give two fixes that don't involve changing the reward back.
3. A manipulation task ends after a fixed 5 s, and success is scored only at the last step. Should `time_outs` be reported to RSL-RL? What would bootstrapping do to the value targets at the end of each episode?
4. You add privileged friction and mass observations to the critic only. Write the `obs_groups` entry and the environment's observation dictionary. Then explain why the exported ONNX policy still runs on the real robot, and what would break if you had put the privileged group in the actor's set.
5. Lab 6b-b uses 32-step rollouts for 200-step episodes. Walk through what happens to the GAE recursion at step 31 of a rollout when the episode is still running, and at a step where a time-out occurs.
6. Your 8 GB GPU forces you from 4,096 to 1,024 environments. Predict how `Perf/total_fps`, the minibatch size, and the adaptive learning-rate trajectory change. Which RSL-RL setting would you adjust to keep the update statistics similar?

---

## References

* Schwarke, Mittal, Rudin, Hoeller, Hutter, "RSL-RL: A Learning Library for Robotics Research," 2025 — [arXiv:2509.10771](https://arxiv.org/abs/2509.10771)
* RSL-RL source, tag v5.5.1 — [GitHub](https://github.com/leggedrobotics/rsl_rl/tree/v5.5.1) · [PyPI `rsl-rl-lib`](https://pypi.org/project/rsl-rl-lib/)
* Schulman et al., "Proximal Policy Optimization Algorithms," 2017 — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
* Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation," 2015 — [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
* Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," 2021 — [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)
* Pinto et al., "Asymmetric Actor Critic for Image-Based Robot Learning," 2017 — [arXiv:1710.06542](https://arxiv.org/abs/1710.06542)
* Andrychowicz et al., "What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study," 2020 — [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)
* Huang et al., "The 37 Implementation Details of Proximal Policy Optimization" — [ICLR blog track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
* Isaac Lab 3.0, reinforcement learning concepts — [docs](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/reinforcement_learning.html)

---

## Next in this special course

* Next: [Lecture 07 — Value-Based and Off-Policy RL: From Q-Learning to SAC](Lecture-07.md)
* Previous: [Lecture 06 — PPO, Trust Regions, and the KL Leash](Lecture-06.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
