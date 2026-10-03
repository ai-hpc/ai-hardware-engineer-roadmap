# Lecture 03: Modern Imitation: Multimodality, Action Chunks, Flow Matching

## Overview

Imagine a dataset of a hundred drivers passing a tree in the middle of the road. Half went left and half went right. Ask a regression model "what did the drivers do here?" and it answers with the average: straight ahead, into the tree. Every single demonstration was good. The average of them is a crash.

Human demonstrations are full of this. One operator grasps the bowl from the left, another from the right. The same operator pauses on some trials and not others. Lecture 02's scripted expert hid the problem because it always did the same thing in the same state. Real data does not.

This lecture covers the four ideas that turned behavioral cloning into the recipe behind today's VLAs:

1. **Model the whole distribution of good actions**, not its mean. Either discretize actions and predict them token by token, like spelling a word, or start from random noise and shape it into a good move (diffusion / **flow matching**). Different noise gives different, equally good answers.
2. **Predict action chunks.** Plan the next ~50 small moves at once and execute some of them before re-planning. That means fewer decisions, less compounding, and less mode-switching.
3. **Use history carefully.** Memory helps under partial observability, but extra inputs open the door to **causal confusion**.
4. **Pre-train on huge messy data, post-train on small clean data.** π0.5 puts all of this together: cameras + instruction → a vision-language model → a flow-matching *action expert* → the next chunk of actions.

And the hardware question running through all of it: a flow head costs **\(K\) network evaluations per chunk**, a tokenized head costs **one sequential decode step per token**, and chunking divides both by \(H\). Those three numbers set your control-loop latency and your RL rollout throughput.

By the end you should be able to:

* prove that MSE regression learns the conditional mean, and show why that fails for multimodal demonstrations
* compare GMM, tokenized-autoregressive, diffusion, and flow-matching action heads on expressiveness and inference cost
* derive the flow-matching training target, implement training and Euler sampling, and explain why **1-step sampling collapses back to the mean**
* choose a chunk length and execution horizon \(H\), and explain receding-horizon execution and ACT's temporal ensembling
* describe the π0 / π0.5 architecture at the level of the papers and estimate per-chunk latency from prefill, NFE, and \(H\)

---

## 1. Why it matters: what Lecture 02 got away with

| Lecture 02 assumption | Reality for "put the bowl on the plate" | What breaks | Fix in this lecture |
|---|---|---|---|
| One consistent expert | Many operators, many styles | MSE averages modes | Distributional heads (§3-4) |
| Full state observation | Cameras; occlusions; the drawer's inside is hidden | Markov assumption | History, with care (§2) |
| One action per decision | 10-50 Hz control, long tasks | Compounding over many decisions; jitter | Action chunks (§5) |
| Small MLP, microseconds | Multi-billion-parameter VLA | Latency budget | NFE, chunking, KV reuse (§7) |

**Robot connection.** π0.5 is the end product of this lecture. Improving it in the capstone is **post-training**: you start from a model that already knows a lot about objects, language, and motion, and the first rule is not to erase that knowledge while teaching it something new.

---

## 2. History, memory, and causal confusion

### 2.1 Why add history

With observations instead of states (a POMDP, Lecture 01), the current frame may not determine the right action. Was the gripper moving toward the bowl or away from it? Is the drawer already open? The standard fix is to feed a short window of past observations, usually through a transformer over frame tokens, or to carry a recurrent state.

### 2.2 Why history can hurt: causal confusion

de Haan, Jayaraman & Levine (2019) give the canonical example. A driving policy sees the dashboard, including a **brake indicator light**. In the demos, the light is on exactly when the expert is braking, because it is a *consequence* of braking. A cloner can reach near-zero training loss with the rule "brake when the light is on" and ignore the pedestrian, the actual cause. At test time the light is on only if the policy is already braking, so the rule is useless. The paper shows that giving the cloner *more* information can make it *worse* in closed loop, and proposes resolving the confusion with targeted interventions (expert queries or policy execution).

History creates a version of this for free. The previous action is an excellent predictor of the current action, because demos are smooth. A policy given its own past actions or a long frame history can learn to **copy its previous action** (the "copycat" problem; Wen et al., 2020). Training loss looks excellent, and the policy never reacts.

Practical mitigations:

* **Don't feed past actions** unless you have a reason. Feed past *observations* sparingly.
* **History dropout:** randomly drop or blank past frames during training so the policy cannot rely on them as a shortcut.
* **Predict chunks from the current observation** (§5). Chunking gives temporal consistency without making the past an input.
* **Check in closed loop.** Validation loss cannot detect causal confusion. Only rollouts, or DAgger-style queries on the learner's states, can.

**Robot connection.** π0 conditions on the current observation (camera images, instruction, proprioceptive state) and gets temporal coherence from action chunks rather than a long frame history. If you add history while post-training, measure closed-loop success before and after. Don't rely on loss.

---

## 3. Multimodal actions

### 3.1 MSE learns the conditional mean

For any predictor \(f\), split the error around \(\bar a(o) = \mathbb{E}[a \mid o]\):

$$
\mathbb{E}\big[\lVert a - f(o) \rVert^2 \mid o\big] = \mathbb{E}\big[\lVert a - \bar a(o) \rVert^2 \mid o\big] + \lVert \bar a(o) - f(o) \rVert^2 .
$$

The cross term vanishes because \(\mathbb{E}[a - \bar a(o) \mid o] = 0\). The first term does not depend on \(f\), so the minimizer is \(f^\star(o) = \mathbb{E}[a \mid o]\). With demos split evenly between "steer \(-1\)" and "steer \(+1\)", \(f^\star = 0\): into the tree. The same argument holds for the Gaussian MLE of Lecture 02. A unimodal model puts its mode at the mean.

### 3.2 Ways to represent a multimodal action distribution

| Head | How it represents \(p(a \mid o)\) | Strength | Weakness | Inference cost per decision |
|---|---|---|---|---|
| **MSE / Gaussian** | Single mode | Simple, fast | Averages modes | 1 head pass |
| **Mixture of Gaussians (MDN; Bishop 1994)** | \(\sum_k w_k \mathcal{N}(\mu_k, \Sigma_k)\) | Exact likelihood, cheap | Mode collapse in training; scales badly to 350-D chunks | 1 head pass |
| **Binned + autoregressive** (RT-2, OpenVLA) | Each dimension discretized (e.g. 256 bins) and predicted as a token, one after another: \(p(a) = \prod_d p(a_d \mid a_{<d}, o)\) | Reuses the LLM's softmax and decoding; captures correlations | One sequential decode step per token; a 50×7 chunk is 350 tokens | \(L\) sequential decode steps |
| **Compressed tokens** (FAST: DCT + byte-pair encoding of chunks) | Fewer, more informative tokens per chunk | Makes autoregressive chunking practical | Still sequential; tokenizer is fixed | fewer decode steps |
| **k-means bins + offset** (Behavior Transformers; Shafiullah et al., 2022) | Classify the cluster, regress a residual | Handles a few modes well | Coarse in high dimension | 1 pass |
| **Diffusion** (Diffusion Policy) | Iterative denoising from Gaussian noise | Very expressive; works on whole chunks | Many denoising steps | \(K\) denoiser passes |
| **Flow matching** (π0, π0.5) | ODE from noise to action along a learned velocity field | Expressive, simple loss, few steps | No cheap exact log-likelihood (matters for RL; Lecture 11) | \(K\) velocity passes |

Why autoregression matters for tokenized heads: if dimensions are predicted **independently**, the model can only represent products of marginals. Demos at \((-1,-1)\) and \((+1,+1)\) would also give mass to \((-1,+1)\). Conditioning each token on the previous ones, "spelling" the action, restores the joint distribution.

---

## 4. Flow matching

### 4.1 The four-step picture

1. Take a demo action \(a\) and a sample of pure noise \(\epsilon\).
2. Mix them partway: \(x_t\) is \(t\) of the way from noise to action.
3. Train a network to point from the mix toward the action: predict the direction \(a - \epsilon\).
4. To act: start from fresh noise and follow the arrows, about 10 small steps.

Different starting noise flows to different good actions. That is the multimodality.

### 4.2 The math (convention used in this lecture)

We use \(t \in [0,1]\) with **\(t = 0\) = noise, \(t = 1\) = action** (the rectified-flow / linear-path convention). Given an observation \(o\), demo action (or chunk) \(a\), and \(\epsilon \sim \mathcal{N}(0, I)\):

$$
x_t = (1 - t)\,\epsilon + t\,a, \qquad u_t = \frac{d x_t}{dt} = a - \epsilon .
$$

Train the velocity network \(v_\theta\) by regression:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(o,a) \sim \mathcal{D},\; \epsilon,\; t \sim \mathcal{U}[0,1]} \Big[ \big\lVert v_\theta(x_t, t, o) - (a - \epsilon) \big\rVert^2 \Big].
$$

Sample with \(K\) Euler steps of size \(1/K\):

$$
x_0 \sim \mathcal{N}(0, I), \qquad x_{k+1} = x_k + \tfrac{1}{K}\, v_\theta\!\big(x_k,\, \tfrac{k}{K},\, o\big), \qquad \hat a = x_K .
$$

**Why is MSE fine here when it failed in §3?** By the same argument as §3.1, the regression learns a conditional mean, but of the *velocity* at a noisy point:

$$
v^\star(x, t, o) = \mathbb{E}\big[a - \epsilon \;\big|\; x_t = x,\; o\big].
$$

The averaging is over the (noise, action) pairs whose straight paths pass through the same point \(x\) at time \(t\), not over all actions for the observation. Lipman et al. (2022) and Liu et al. (2022) show that integrating this averaged field from \(x_0 \sim \mathcal{N}(0,I)\) carries the noise distribution to the data distribution \(p(a \mid o)\). Paths from different noise samples separate as \(t\) grows, and each one ends in some mode.

**Why 1-step sampling collapses to the mean.** At \(t = 0\), \(x_0 = \epsilon\) is independent of \(a\), so \(v^\star(\epsilon, 0, o) = \mathbb{E}[a \mid o] - \epsilon\). One Euler step gives

$$
\hat a = \epsilon + v^\star(\epsilon, 0, o) = \mathbb{E}[a \mid o] .
$$

With \(K = 1\), a perfectly trained flow policy *is* the MSE policy, and drives into the tree. Multimodality lives in the later integration steps. This is the first thing to remember when someone proposes cutting denoising steps to speed up rollouts: fewer steps is not just "slightly less accurate". In the limit, it removes the property you chose flow matching for. (Distillation and consistency-style methods train a model specifically for 1-2 steps; that is a different model, not the same model sampled with fewer steps.)

**Convention warning.** The π0 paper writes the path with \(\tau\) weighting the action and samples \(\tau\) from a shifted Beta distribution that emphasizes noisier timesteps. The openpi code uses the opposite labeling: \(x_t = t\,\epsilon + (1-t)\,a\), target \(\epsilon - a\), and integrates from \(t = 1\) down to \(0\) with \(\delta = -1/K\), \(K = 10\) by default. These are the same family of models. When you port weights, write a sampler, or add noise injection for RL (Lecture 11), follow the code's convention, not a paper's or this lecture's.

### 4.3 Relation to diffusion

Diffusion Policy (Chi et al., 2023) trains a denoiser on a DDPM-style noising process over whole action sequences and samples by iterative denoising, often with DDIM-style samplers to cut steps. Flow matching with a linear path is a close cousin with a simpler loss, and its straighter paths tolerate fewer integration steps. Lecture 08 makes the family resemblance precise: both are stacks of small latent-variable steps, relatives of the VAE.

---

## 5. Action chunks

### 5.1 Predict \(P\), execute \(H\), re-plan

The policy outputs a chunk \(\hat a_{t:t+P} \sim \pi_\theta(\cdot \mid o_t)\) of \(P\) future actions. You execute the first \(H \le P\), then query again. This is **receding-horizon** execution, the same pattern as MPC (Lecture 09).

```text
 t:   0    5    10   15   20   25 ...
      [ chunk 1: P=16 ─────────]
      [exec H=5]
           [ chunk 2: P=16 ─────────]
           [exec H=5]
                [ chunk 3 ...
```

Why chunking helps:

1. **Fewer decisions, less compounding.** The Lecture 02 bound counts decisions. With \(T/H\) decisions per episode instead of \(T\), the \(T^2\) factor shrinks accordingly (as a heuristic — within-chunk errors still accumulate open-loop). Lecture 13 formalizes this as a shorter effective horizon.
2. **Commitment to one mode.** A single-step multimodal policy can sample "left" at step 3 and "right" at step 4 and dither into the tree. A chunk commits to one coherent plan for \(H\) steps.
3. **Non-Markovian demos.** Human pauses and hesitations aren't predictable from the current frame. They are much more predictable as part of a sequence (an argument made in the ACT paper).
4. **Throughput.** \(T/H\) policy calls instead of \(T\) (Lecture 01, §3).

The cost is reactivity. Within the \(H\) executed steps, the robot is open-loop. If someone nudges the bowl at step 2 of a 50-step open-loop chunk, the policy does not see it until the chunk ends.

### 5.2 Execution schemes

| Scheme | Policy calls per episode | Reactivity | Smoothness |
|---|---|---|---|
| \(H = 1\) (re-plan every step) | \(T\) | Best | Can dither between modes |
| Receding horizon, \(1 < H < P\) | \(T/H\) | Good | Good, small seams at chunk boundaries |
| \(H = P\) (open-loop chunks) | \(T/P\) | Worst | Seams at every boundary |
| **Temporal ensembling** (ACT) | \(T\) | Good | Best: averages overlapping predictions |

**ACT** (Action Chunking with Transformers; Zhao et al., 2023) trains a transformer as a conditional VAE to output chunks. With temporal ensembling, the policy is queried **every step**. All chunks that cover the current timestep are averaged with exponential weights \(w_i = \exp(-m\, i)\), where \(i = 0\) is the oldest prediction. That gives smoothness without open-loop execution, but at the full \(T\) calls per episode, so no amortization. Note also that averaging chunks drawn from *different modes* reintroduces the mean problem. Ensembling is safe only when consecutive chunks agree.

---

## 6. Pre-train → post-train, and the π0 / π0.5 architecture

**The recipe.** Pre-train on a huge, heterogeneous mixture (many robots, many tasks, web vision-language data) to learn general visual, semantic, and motor knowledge. Post-train on a small, clean, task-focused dataset to get high-quality, consistent behavior. It is the same split as LLM pre-training and SFT.

**π0** (Black et al., 2024), as the paper describes it:

```text
  cameras (1-3 images) ─┐
  instruction text ─────┼──► PaliGemma VLM (~3B)  ── prefix: KV cache ──┐
  proprio state ────────┘                                              │ attends
                                                                       ▼
                     noisy action chunk x_t (P=50) + t ──► action expert (~300M) ──► velocity
                                                       ▲                               │
                                                       └──── 10 Euler steps ◄──────────┘
```

* Total ~3.3B parameters. The action expert is a separate set of transformer weights (~300M) for the action tokens. It shares attention with the VLM tokens through a block-wise causal mask.
* Chunks of \(P = 50\) actions, sampled with 10 integration steps.
* Because the prefix (images, language, state) does not attend to the action tokens, its keys and values are **computed once per chunk and cached**. Only the action expert's suffix is recomputed at each integration step. The paper reports on-board inference of about 73 ms on an RTX 4090 for its setup.

**π0.5** (Physical Intelligence, 2025) keeps this structure and changes the training and inference recipe:

* **Co-training** on heterogeneous data: mobile and static manipulators in many homes, cross-embodiment lab data, web data (captioning, question answering, object localization), and high-level subtask annotations.
* **Two training stages.** Pre-training represents actions as discrete FAST tokens. A post-training stage adds the flow-matching action expert that produces continuous chunks.
* **Hierarchical inference.** The model first predicts a high-level subtask as text ("pick up the plate"), then the action expert produces the low-level chunk conditioned on that subtask.

Don't fill in details the papers don't give. For the exact chunk length, action dimension padding, and normalization of a specific checkpoint, read its openpi config.

**Robot connection: post-training without erasing.** Fine-tuning on a narrow dataset with a high learning rate can quietly destroy the general skills that make the policy robust to moved objects and new phrasing (catastrophic forgetting). Standard protections: low learning rate, LoRA, mixing in broader data, short training, and evaluating on held-out suites *and* perturbed layouts after every change. RL post-training adds a KL leash to the original policy (Lecture 06).

---

## 7. The hardware view: where action-head latency goes

Per-chunk latency for a VLA policy decomposes as:

$$
t_{\text{chunk}} \approx t_{\text{vision}} + t_{\text{prefill}} + \begin{cases} K \cdot t_{\text{expert}} & \text{flow / diffusion head} \\ L \cdot t_{\text{decode}} & \text{autoregressive tokens} \end{cases}
$$

and the cost per control step is \(t_{\text{chunk}} / H\).

| Lever | Affects | Hardware character |
|---|---|---|
| **NFE \(K\)** (flow steps) | Multiplies \(t_{\text{expert}}\), not the prefix (KV cached) | Small transformer over \(P\) action tokens, all in parallel, so compute-light per step. \(K\) runs are sequential and often **launch-bound** at batch 1 (CUDA Graphs help). |
| **Tokens per chunk \(L\)** (autoregressive) | Multiplies \(t_{\text{decode}}\) | Each decode step reads all LM weights, so it is **memory-bandwidth-bound**. 350 binned tokens per 50×7 chunk is impractical, hence FAST compression or single-step actions. |
| **Execution horizon \(H\)** | Divides everything | Amortizes vision + prefill + head over \(H\) control steps |
| **History frames** | \(t_{\text{vision}}\), \(t_{\text{prefill}}\), KV memory | Each 224×224 image through PaliGemma's SigLIP encoder is 256 tokens. Prefill grows linearly in tokens (attention quadratically). KV memory per token is \(2 \cdot n_{\text{layers}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot \text{bytes}\). |
| **Batching across envs** | Throughput, not latency | Regime C rollouts (Lecture 01): batch chunk calls across parallel simulators |

Two consequences for the rest of the course:

* **Deployment and RL want different settings.** Deployment needs \(t_{\text{chunk}}\) at batch 1 within the control deadline. RL rollouts want throughput at large batch, and would like small \(K\) — but §4.2 shows \(K\) cannot go to 1 without changing behavior. Measure success vs \(K\) before cutting it for rollouts.
* **Flow heads are parallel; token heads are sequential.** A flow head's \(K\) passes each process the whole chunk at once. A tokenized head produces one token per pass. That is a core reason the π0 line uses flow matching for continuous control, and the same choice is why Lecture 11 has to work around the missing log-likelihood.

---

## 8. Build it

Everything lives in `flow_policy/`.

### Lab 3a — The tree in 2-D: MSE-BC vs flow matching

```python
# flow_policy/toy2d.py
import torch, torch.nn as nn
OBST, R, GOAL = torch.tensor([0.0, 1.0]), 0.3, torch.tensor([0.0, 2.0])

def demos(n=2000, steps=30):
    """Each demo goes around the obstacle on a random side: a bimodal expert."""
    side = (torch.randint(0, 2, (n,)) * 2 - 1).float()           # -1 left, +1 right
    p = 0.05 * torch.randn(n, 2)                                 # start near the origin
    S, A = [], []
    for _ in range(steps):
        way = torch.stack([0.5 * side, torch.ones(n)], 1)
        tgt = torch.where((p[:, 1] < 1.0)[:, None], way, GOAL.expand(n, 2))
        a = tgt - p
        a = 0.1 * a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)  # fixed step length
        S.append(p.clone()); A.append(a * 10)                    # store actions scaled to ~unit
        p = p + a
    return torch.cat(S), torch.cat(A)

class Flow(nn.Module):
    def __init__(self, obs_dim, act_dim, h=256):
        super().__init__(); self.act_dim = act_dim
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim + 1, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, act_dim))
    def forward(self, o, x, t):
        return self.net(torch.cat([o, x, t], -1))
    def loss(self, o, a):
        eps = torch.randn_like(a)
        t = torch.rand(a.shape[0], 1, device=a.device)
        x_t = (1 - t) * eps + t * a                              # t=0 noise, t=1 action
        return ((self(o, x_t, t) - (a - eps)) ** 2).mean()
    @torch.no_grad()
    def sample(self, o, K=10):
        x = torch.randn(o.shape[0], self.act_dim, device=o.device)
        for k in range(K):
            t = torch.full((o.shape[0], 1), k / K, device=o.device)
            x = x + self(o, x, t) / K
        return x

@torch.no_grad()
def rollout(act_fn, n=500, steps=30):
    p = 0.05 * torch.randn(n, 2); hit = torch.zeros(n, dtype=torch.bool)
    for _ in range(steps):
        p = p + act_fn(p) / 10
        hit |= (p - OBST).norm(dim=1) < R
    return hit.float().mean().item(), p
```

Train an MSE MLP (`nn.Sequential` 2 → 256 → 256 → 2) and a `Flow(2, 2)` on the same `demos()` for a few thousand Adam steps each. Then report:

* collision rate for MSE-BC, and for the flow policy at \(K \in \{1, 2, 5, 10\}\)
* a plot of 50 rollouts per method, with the obstacle drawn
* the **number of side switches** per rollout for the flow policy (sign changes of \(x\) while \(y < 1\))

Expected: MSE-BC goes straight up and collides. Flow at \(K = 1\) behaves like MSE-BC (§4.2). Flow at \(K \ge 5\) splits roughly evenly between left and right, with occasional dithering near the axis. Then make the flow policy predict **chunks** of 8 steps (`act_dim = 16`), execute \(H = 8\), and show the dithering disappears.

### Lab 3b — Chunked flow policy on PickCube

Reuse the successful expert demos from Lecture 02 (`bc_dagger/demos.pt`: `O: (N, T, obs_dim)`, `A: (N, T, 7)`, filtered by `success`).

```python
# flow_policy/chunks.py
import torch

def make_chunks(O, A, P):
    """(o_t, a_{t:t+P}) pairs; pad past the episode end by repeating the last action."""
    N, T, da = A.shape
    A_pad = torch.cat([A, A[:, -1:].expand(N, P - 1, da)], 1)
    idx = torch.arange(T, device=A.device)[:, None] + torch.arange(P, device=A.device)[None]
    return O.reshape(N * T, -1), A_pad[:, idx].reshape(N * T, P * da)   # (N*T, P*7)
```

Train `Flow(obs_dim, P * 7)` on normalized observations with \(P = 16\). Then evaluate with GPU-parallel envs:

```python
# flow_policy/eval_sweep.py   (make_env from bc_dagger/rollout.py)
import time, torch

@torch.no_grad()
def evaluate(flow, mu, sd, P, H, K, n=1024, T=50, seed=0):
    env = make_env(n); obs, _ = env.reset(seed=seed)
    succ = torch.zeros(n, dtype=torch.bool, device=obs.device)
    calls, t_pol = 0, 0.0
    for t in range(T):
        if t % H == 0:                                        # receding horizon: re-plan every H
            torch.cuda.synchronize(); t0 = time.perf_counter()
            chunk = flow.sample((obs - mu) / sd, K).view(n, P, 7)
            torch.cuda.synchronize(); t_pol += time.perf_counter() - t0; calls += 1
        obs, _, _, _, info = env.step(chunk[:, t % H].clamp(-1, 1))
        succ |= info["success"]
    return succ.float().mean().item(), calls, t_pol / calls
```

Sweep \(K \in \{1, 2, 5, 10\}\) × \(H \in \{1, 2, 4, 8, 16\}\), ≥ 3 training seeds, fixed eval seeds. Separately time `flow.sample` at **batch 1** for each \(K\) (deployment latency) and at batch 1024 (rollout throughput).

Make it multimodal (recommended): give the Lecture 02 expert a random per-episode approach offset (come in from \(+y\) or \(-y\) of the cube before descending), collect new demos, and compare the MSE-BC policy against the flow policy on the same data. That is the tree, on a real task.

Optional: temporal ensembling (query every step, average overlapping chunks with \(w_i = e^{-m i}\)) at \(H = 1\). Compare its smoothness and success with receding horizon at equal \(K\), and note its call count.

---

## 9. Use it in the real stack

* **[openpi](https://github.com/Physical-Intelligence/openpi)** — π0, π0-FAST, and π0.5 code and checkpoints, including a LIBERO example. Read the flow-matching sampler in `src/openpi/models/pi0.py` (time convention, KV-cached prefix, `num_steps`) and the LIBERO client in `examples/libero/main.py`. It re-plans every `replan_steps` actions (5 at the time of writing), which is exactly the \(H\) of §5.
* **[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)** and **[ACT](https://github.com/tonyzhaozh/act)** — the reference implementations of diffusion heads and chunking with temporal ensembling.
* **[LeRobot](https://github.com/huggingface/lerobot)** — PyTorch implementations of several of these policies (including ACT and Diffusion Policy) with a common dataset format. A good place to compare heads on equal data.
* **[OpenVLA](https://github.com/openvla/openvla)** — the canonical binned-token VLA. Compare its per-action decode cost with a flow head.
* **Robot connection.** On rung 3, your Lab 3b sweep becomes: for π0.5 on LIBERO, measure success and per-chunk latency vs `num_steps` and replan horizon on nominal *and* moved-object layouts. That table decides how many denoising steps the capstone's RL rollouts can afford.

---

## 10. Measure it

| Metric | Definition | Why it matters |
|---|---|---|
| **Toy collision rate** | Fraction of rollouts hitting the obstacle | Shows mean-collapse directly |
| **Side-switch count** | Mode changes per rollout | Shows why chunks help commitment |
| **Success vs \(K\)** | PickCube success at each NFE | Quality cost of fewer integration steps |
| **Success vs \(H\)** | PickCube success at each execution horizon | Reactivity vs consistency trade-off |
| **\(t_{\text{chunk}}\) at batch 1** | `flow.sample` latency per call | Deployment latency |
| **Chunk throughput at batch \(N\)** | Chunks/s across envs | RL rollout throughput (regime C preview) |
| **Policy calls per episode** | \(\lceil T/H \rceil\) (or \(T\) with ensembling) | What you pay per episode |
| **Amortized latency per control step** | \(t_{\text{chunk}} / H\) | Must fit the control period |
| **Peak GPU memory** | Training and inference | Headroom for history frames / bigger heads |

---

## 11. Ship it

Commit `flow_policy/` containing:

* `toy2d.py`, `chunks.py`, `flow.py`, `train.py`, `eval_sweep.py`, `run_sweeps.sh`
* `toy_rollouts.png` — MSE-BC vs flow at \(K = 1\) and \(K = 10\), and chunked flow
* `results.csv` — one row per (\(K\), \(H\), seed) with success, calls/episode, \(t_{\text{chunk}}\) at batch 1 and batch \(N\), amortized latency
* `latency_success.md` — the \(K \times H\) table (success ± std across seeds, amortized ms per control step), plus three sentences. Which \((K, H)\) would you deploy under a 20 ms control period? Which would you use for RL rollouts? Where did \(K = 1\) land, and does it match §4.2?

The trained flow policy is the starting point for Lecture 10 (noise-space steering) and Lecture 11 (GRPO on a flow policy).

---

## Exit criteria

You can move on when you can:

* prove \(\arg\min_f \mathbb{E}\lVert a - f(o)\rVert^2 = \mathbb{E}[a \mid o]\) and draw the tree example
* write the flow-matching loss and sampler from memory, in your chosen time convention, and translate it to openpi's
* show from your own toy results that 1-step flow sampling reproduces MSE-BC, and explain why
* explain receding-horizon execution and temporal ensembling, and the reactivity cost of large \(H\)
* estimate \(t_{\text{chunk}}\) and calls per episode for a flow head vs a tokenized head given prefill, per-step, and per-token costs

---

## Self-check

1. Two teleoperators grasp the mug by its handle, one always approaching from the left and the other from the right. Your MSE-BC policy hovers in front of the mug and closes on nothing. Explain this with the conditional-mean result, name two heads that fix it, and say what each costs at inference.
2. To speed up RL rollouts, a teammate cuts π0.5's integration steps from 10 to 1. Success drops and the robot's grasps look "averaged". Derive why using the velocity field at \(t = 0\), and propose a safer way to reduce rollout cost.
3. With \(P = H = 50\) the robot fails whenever the bowl is nudged mid-execution. With \(H = 1\) it hesitates between two grasp approaches. What \(H\) do you try next, and how would you choose it from measurements rather than taste? What would temporal ensembling cost in policy calls per 300-step episode?
4. Adding four frames of observation history lowers validation loss but reduces closed-loop success. Give two hypotheses (one causal-confusion, one copycat) and an experiment that distinguishes them.
5. Prefill costs 40 ms, the action expert 4 ms per integration step, \(K = 10\), \(H = 5\), and the episode is 300 steps. Compute policy time per episode. Compare it with a tokenized head that needs 8 ms per token, 7 tokens per action, and 10-action chunks executed 5 at a time with the same prefill.
6. After fine-tuning π0.5 on 200 demos of a new kitchen task, success on that task is high, but success on the standard LIBERO suites and on moved-object layouts drops. What happened, and list three changes to the post-training recipe you would test.

---

## References

* CS 285 Lecture 3 (imitation learning, continued) — [slides and video](https://rail.eecs.berkeley.edu/deeprlcourse/)
* de Haan, Jayaraman & Levine, "Causal Confusion in Imitation Learning," NeurIPS 2019 — [arXiv:1905.11979](https://arxiv.org/abs/1905.11979)
* Wen et al., "Fighting Copycat Agents in Behavioral Cloning from Observation Histories," NeurIPS 2020 — [arXiv:2010.14876](https://arxiv.org/abs/2010.14876)
* Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023 — [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
* Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), RSS 2023 — [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
* Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023 — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
* Liu, Gong & Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow," ICLR 2023 — [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
* Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," 2023 — [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
* Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," 2024 — [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
* Pertsch et al., "FAST: Efficient Action Tokenization for Vision-Language-Action Models," 2025 — [arXiv:2501.09747](https://arxiv.org/abs/2501.09747)
* Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," 2024 — [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
* Physical Intelligence, "π0.5: a Vision-Language-Action Model with Open-World Generalization," 2025 — [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) · [openpi](https://github.com/Physical-Intelligence/openpi)

---

## Next in this special course

* Next: [Lecture 04 — Policy Gradients: Do More of What Worked](Lecture-04.md)
* Previous: [Lecture 02 — Imitation Learning: Behavioral Cloning and DAgger](Lecture-02.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
