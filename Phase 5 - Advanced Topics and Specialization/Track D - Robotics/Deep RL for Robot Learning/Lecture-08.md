# Lecture 08: The Probabilistic Toolkit: ELBO, VAEs, Control as Inference, Inverse RL

## Overview

You walk outside and the street is wet. You did not see what caused it, but you can reason about it: maybe it rained, maybe a sprinkler ran. The cause is **hidden**, and working out how likely each cause is from what you can see is **inference**. Doing that exactly means checking every possible hidden cause, and for anything bigger than a toy, there are far too many. So you take a simpler guess, measure how wrong that guess could be, and improve it.

That one move — replace an intractable distribution with a simple guess \(q\) and optimize a bound — is the toolkit of this lecture. It shows up all over robot learning:

* **VAEs** squeeze a camera frame or an action chunk into a small code and rebuild it. Diffusion and flow-matching heads, π0.5's action expert included, are close relatives of the VAE.
* **Control as inference** reframes "act optimally" as "infer which actions are consistent with success." It produces the entropy bonus SAC used in [Lecture 07](Lecture-07.md) and explains why a soft, stochastic policy is the principled answer, not a hack.
* **Inverse RL** runs the problem backwards: given expert behavior, infer the reward that explains it. Copy the *intention*, not the motions.
* **Learned success detectors** — the practical end of inverse RL — are how you get a reward on a real robot, where nothing checks "bowl on plate" for free. They are also how policies learn to cheat.

The common measuring tools are **KL divergence** (how different two distributions are) and **entropy** (how spread out one distribution is). You have already met the KL leash in [Lecture 06](Lecture-06.md). This lecture explains where those quantities come from.

By the end you should be able to:

* derive the ELBO two ways (Jensen, and the exact KL-gap identity) and say what the gap is
* write the VAE loss and explain why the reparameterization trick has lower variance than the score-function (REINFORCE) gradient, and when you can't use it
* explain in what sense a diffusion model is a stack of VAEs, and why a flow-matching policy has no cheap exact log-likelihood
* derive the soft Bellman backup and \(\pi \propto \exp(Q/\alpha)\) from control as inference, and show that SAC's actor step is a KL projection
* state the MaxEnt IRL gradient and the GAIL objective, and explain what each recovers
* train a success classifier, measure precision/recall, and catch a policy exploiting it

---

## 1. Why it matters: KL and entropy are the course's measuring tools

| Where | Quantity | Role |
|---|---|---|
| PPO / TRPO ([L06](Lecture-06.md)) | \(\mathrm{KL}(\pi_{\text{old}} \,\Vert\, \pi_\theta)\) | Step-size limit |
| KL leash to π0.5 ([L06](Lecture-06.md), [L11](Lecture-11.md)) | \(\mathrm{KL}(\pi_\theta \,\Vert\, \pi_{\text{ref}})\) | Keep general skills, don't game the scorer |
| SAC ([L07](Lecture-07.md)) | \(\mathcal{H}(\pi(\cdot \mid s))\) | Exploration, robustness to critic errors |
| VAE / diffusion / flow heads ([L03](Lecture-03.md)) | ELBO, KL to prior | Training generative action models |
| Exploration bonuses ([L12](Lecture-12.md)) | Novelty, mutual information | Curiosity, skill discovery |
| Inverse RL / GAIL (this lecture) | Divergence between expert and policy occupancies | Learn a reward from demonstrations |

If you can derive the ELBO and the soft Bellman equation, every row of that table becomes a variation on a theme, not a separate trick.

---

## 2. Latent variables and the ELBO

### 2.1 Mental model

A **latent variable model** explains an observation \(x\) (wet street, camera frame, action chunk) through a hidden cause \(z\) (rain or sprinkler, scene layout, "approach from the left" vs "from the right"):

$$
p_\theta(x) = \int p_\theta(x \mid z) \, p(z) \, dz
$$

The integral is the problem. With \(z\) continuous and \(p_\theta(x \mid z)\) a neural network, nobody can compute it, so nobody can maximize \(\log p_\theta(x)\) directly. The posterior \(p_\theta(z \mid x)\) ("which cause produced this?") is just as intractable.

### 2.2 Derivation 1: Jensen

Introduce any distribution \(q(z)\) (the "guesser") and multiply and divide by it:

$$
\log p_\theta(x) = \log \mathbb{E}_{q(z)}\!\left[ \frac{p_\theta(x, z)}{q(z)} \right] \;\ge\; \mathbb{E}_{q(z)}\!\left[ \log p_\theta(x, z) - \log q(z) \right] \;=\; \mathcal{L}(\theta, q)
$$

The inequality is Jensen's (log is concave). \(\mathcal{L}\) is the **evidence lower bound (ELBO)**, and it can be estimated with samples from \(q\). Rewriting the last term shows where entropy enters:

$$
\mathcal{L} = \mathbb{E}_{q}\!\left[ \log p_\theta(x, z) \right] + \mathcal{H}(q)
$$

"Explain the data well, but keep your guess spread out": the same two forces as the max-entropy RL objective in section 4.

### 2.3 Derivation 2: the exact gap

Jensen tells you there is a gap but not how big it is. Expand \(\log p_\theta(x, z) = \log p_\theta(z \mid x) + \log p_\theta(x)\) inside the ELBO:

$$
\log p_\theta(x) = \mathcal{L}(\theta, q) + \mathrm{KL}\big(q(z) \,\Vert\, p_\theta(z \mid x)\big)
$$

This identity is exact, and it says three things:

1. The ELBO is always ≤ the true log-likelihood, because KL ≥ 0.
2. The gap is exactly how wrong the guesser is about the true posterior.
3. **Raising the ELBO in \(q\) shrinks the gap. Raising it in \(\theta\) improves the model.** Joint optimization does both, which is the whole trick of variational inference.

Note the direction: \(\mathrm{KL}(q \Vert p)\) penalizes \(q\) for putting mass where \(p\) has little. So a simple \(q\) fit to a multimodal posterior tends to lock onto one mode (mode-seeking) instead of smearing across all of them. Keep that in mind when a VAE's samples look "too clean."

---

## 3. VAEs, reparameterization, and the diffusion connection

### 3.1 The VAE

Fitting a separate \(q\) per data point is slow. A **VAE** (Kingma & Welling 2013, Rezende et al. 2014) *amortizes* inference: an **encoder** \(q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \operatorname{diag}\sigma^2_\phi(x))\) predicts the guess for any \(x\), and a **decoder** \(p_\theta(x \mid z)\) rebuilds \(x\). With prior \(p(z) = \mathcal{N}(0, I)\), the per-example ELBO is

$$
\mathcal{L}(x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big]}_{\text{reconstruction}} - \underbrace{\mathrm{KL}\big(q_\phi(z \mid x) \,\Vert\, \mathcal{N}(0, I)\big)}_{\text{rate}}
$$

and the KL between diagonal Gaussians is closed-form:

$$
\mathrm{KL} = \tfrac{1}{2} \sum_{j} \big( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \big)
$$

With a Gaussian decoder of fixed variance, the reconstruction term is a scaled MSE. Weighting the KL by \(\beta\) trades reconstruction against how much information the code carries. **Posterior collapse** — the KL goes to ~0, the decoder ignores \(z\) — is the failure mode to watch for when the decoder is powerful.

### 3.2 Reparameterization vs score function

Both the VAE encoder and the RL policy need \(\nabla_\phi \mathbb{E}_{z \sim q_\phi}[f(z)]\). There are two estimators:

$$
\text{score function (REINFORCE, L04):}\quad \nabla_\phi \mathbb{E}[f(z)] = \mathbb{E}_{z \sim q_\phi}\big[ f(z) \, \nabla_\phi \log q_\phi(z) \big]
$$

$$
\text{reparameterization:}\quad z = \mu_\phi + \sigma_\phi \odot \epsilon,\;\; \epsilon \sim \mathcal{N}(0,I) \;\Rightarrow\; \nabla_\phi \mathbb{E}[f(z)] = \mathbb{E}_{\epsilon}\big[ \nabla_z f(z) \, \nabla_\phi z \big]
$$

| | Score function | Reparameterization |
|---|---|---|
| Needs | \(\log q_\phi(z)\) and its gradient; \(f\) can be a black box | \(f\) differentiable in \(z\); \(z\) a differentiable function of \(\phi\) and noise |
| Variance | High: uses only the scalar value of \(f\) (hence baselines, L04) | Low: uses the local slope of \(f\) |
| Used in | Policy gradient through an unknown simulator, GRPO, PPO | VAEs, SAC's actor (through Q), DDPG |

This is why policy gradients need baselines and huge batches while SAC's actor update is quiet. REINFORCE treats the environment as a black box. SAC replaces the environment with a differentiable critic and reparameterizes through it. The trade is variance for bias: the critic can be wrong, and the actor will follow its errors.

### 3.3 Diffusion and flow as stacked VAEs

A diffusion model (Ho et al. 2020) is a **hierarchical VAE with a fixed encoder**. The "encoder" adds a little Gaussian noise at each of many steps, \(x_0 \to x_1 \to \dots \to x_T \approx \mathcal{N}(0,I)\). Every latent has the same dimension as the data, and only the decoder (the denoiser) is learned. The training loss comes from the ELBO of that hierarchy: each KL term between Gaussians reduces to a noise-prediction MSE, which is then reweighted in practice. Kingma et al.'s Variational Diffusion Models make the continuous-time ELBO explicit.

Flow matching (Lecture 03) is trained by direct regression on a velocity field, not by maximizing an ELBO. With Gaussian probability paths it is closely related to diffusion training under a different time weighting. So "π0.5's action expert is a cousin of the VAE" holds in the precise sense: noise in, many small learned denoising steps, a Gaussian latent at the start.

The cousin has one property that matters a great deal for RL. A deterministic flow ODE *does* define an exact density, through the instantaneous change of variables:

$$
\frac{d}{dt} \log p_t(x_t) = -\operatorname{tr}\!\left( \frac{\partial v_\theta(x_t, t)}{\partial x_t} \right)
$$

Evaluating \(\log \pi(a \mid s)\) means integrating the trace of a 350×350 Jacobian (a 50×7 chunk) along the whole path, or approximating it with stochastic trace estimators — every time PPO or GRPO asks for a log-prob. That is the "flow policies have no cheap log-likelihood" problem that [Lecture 11](Lecture-11.md) works around.

---

## 4. Control as inference

### 4.1 Mental model: optimality as evidence

Treat "this step was a good step" as a binary observation \(\mathcal{O}_t\), with probability

$$
p(\mathcal{O}_t = 1 \mid s_t, a_t) = \exp\!\big( r(s_t, a_t) / \alpha \big)
$$

(this assumes \(r \le 0\), or a shifted reward). Now ask the inference question: *given that the whole trajectory was optimal, which actions were taken?* The answer is not "only the best one." Better actions become **exponentially more likely, but not certain**. That soft choice keeps randomness for exploration and keeps several good options alive, which is exactly the "stay a little unpredictable" bonus from Lecture 07, now derived.

### 4.2 The derivation (Levine 2018)

Define backward messages \(\beta_t(s,a) = p(\mathcal{O}_{t:T} \mid s_t = s, a_t = a)\), and let \(Q = \alpha \log \beta\) and \(V = \alpha \log \beta(s)\). With a uniform action prior, the message recursion becomes

$$
V(s) = \alpha \log \int \exp\!\big( Q(s,a)/\alpha \big) \, da, \qquad \pi(a \mid s) = \exp\!\big( (Q(s,a) - V(s))/\alpha \big)
$$

\(V\) is a **soft maximum** (log-sum-exp). As \(\alpha \to 0\) it becomes \(\max_a Q\), and the policy becomes greedy, which recovers ordinary RL. Exact inference has one flaw: its backup is \(Q = r + \alpha \log \mathbb{E}_{s'}[\exp(V(s')/\alpha)]\), which is **optimistic about the dynamics**. It answers "what happened, given that I succeeded?", which assumes lucky transitions. The fix is variational: choose \(q(\tau) = p(s_0)\prod_t p(s_{t+1} \mid s_t,a_t)\, \pi(a_t \mid s_t)\), keep the true dynamics fixed, and maximize the ELBO. That ELBO is (up to constants and the factor \(\alpha\))

$$
\mathbb{E}_{q}\Big[ \sum_t r(s_t,a_t) + \alpha \, \mathcal{H}(\pi(\cdot \mid s_t)) \Big]
$$

which is the max-entropy RL objective, with the familiar **soft Bellman backup** \(Q(s,a) = r + \gamma \mathbb{E}_{s'}[V(s')]\).

### 4.3 SAC is a KL projection

SAC's actor loss from Lecture 07, per state, rearranges into

$$
\mathbb{E}_{a \sim \pi}\big[ \alpha \log \pi(a \mid s) - Q(s,a) \big] = \alpha \, \mathrm{KL}\!\left( \pi(\cdot \mid s) \,\Big\Vert\, \frac{\exp(Q(s,\cdot)/\alpha)}{Z(s)} \right) - \alpha \log Z(s)
$$

\(Z(s)\) does not depend on \(\pi\), so minimizing the actor loss **projects the Boltzmann distribution \(\exp(Q/\alpha)\) onto the tanh-Gaussian family** in KL. Soft Q-learning (Haarnoja et al. 2017) and SAC are this projection, done with different policy families. The temperature \(\alpha\) is now interpretable: it is the scale on which reward differences count as "much better."

**Robot connection.** For the arm, \(\exp(Q/\alpha)\) over chunks is a distribution with several modes (around the bowl to the left, around it to the right). A Gaussian policy fit to it in reverse KL picks one mode. A flow-matching policy can represent the whole distribution, which is part of why "RL on a flow policy" is attractive and also why it is hard.

---

## 5. Inverse RL: copy the intention, not the moves

Behavioral cloning copies *what* the expert did. **Inverse RL** asks *why*: find a reward \(R_\psi\) under which the expert's behavior is optimal, then optimize it, possibly in new layouts where copying the motions fails.

**Ambiguity.** \(R \equiv 0\) makes every behavior optimal, and many different rewards explain the same demonstrations. A principle is needed to pick one.

### 5.1 MaxEnt IRL (Ziebart et al. 2008)

Assume demonstrations come from the soft-optimal distribution of section 4, \(p_\psi(\tau) \propto \exp(R_\psi(\tau))\) (dynamics factors omitted for clarity). Among all distributions that match the expert's expected features, it is the one with maximum entropy: it commits to nothing beyond what the data shows. Maximize the demos' likelihood:

$$
\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{D}_{\text{expert}}}\big[ \nabla_\psi R_\psi(\tau) \big] - \mathbb{E}_{\tau \sim p_\psi}\big[ \nabla_\psi R_\psi(\tau) \big]
$$

"Raise the reward of what the expert did, lower it for what the current soft-optimal policy does." The second expectation needs samples from the policy that is optimal for the *current* reward, so every reward update hides a full RL problem. That inner loop is the cost of IRL.

### 5.2 GAIL and AIRL

**GAIL** (Ho & Ermon 2016) skips the explicit reward. A discriminator \(D_\omega(s,a)\) (convention here: probability that \((s,a)\) came from the expert) plays against the policy:

$$
\max_\omega \; \mathbb{E}_{\text{expert}}\big[\log D_\omega(s,a)\big] + \mathbb{E}_{\pi}\big[\log(1 - D_\omega(s,a))\big]
$$

The policy is trained with any RL algorithm on the reward \(r = -\log(1 - D_\omega(s,a))\): fool the judge. At the optimal discriminator this minimizes a Jensen-Shannon divergence between the expert's and the policy's **state-action occupancies**. Unlike BC, that is a distribution-matching objective over the states the policy *actually visits*, so compounding error (Lecture 02) is penalized directly. The catch: once training ends, \(D\) is not a reusable reward. At equilibrium it outputs ~0.5 everywhere.

**AIRL** (Fu et al. 2017) gives the discriminator a structure, \(D = \exp(f)/(\exp(f) + \pi)\), with \(f\) split into a state-only reward term plus a shaping term. The recovered reward is designed to stay valid when the dynamics change, which is what you want if the reward will be reused on a shifted layout or a real robot.

---

## 6. Learned success detectors and reward hacking

In simulation, "did the bowl end up on the plate" is one line of code reading object poses. On a real robot, nothing reads poses for you. The practical descendant of IRL is a **learned success detector**: a classifier on camera frames (plus the instruction) that outputs \(p(\text{success})\), used as the reward.

* **VICE** (Fu et al. 2018) generalizes IRL to the case where you have only examples of successful *outcomes*, not full demonstrations. The success classifier is trained against the policy's own states as negatives, so it is GAIL-like in structure. Singh et al. 2019 extend this to real robots with occasional human queries for labels, so no reward engineering is needed.
* **VLM reward models** score a frame against a language description. Rocamonde et al. 2023 use CLIP-style models as zero-shot rewards. Larger VLM judges and success detectors are now common in robot-learning pipelines.

**Reward hacking.** An optimizer pointed at a learned reward searches for that reward's errors (Amodei et al. 2016 list this among the concrete safety problems). The classifier was trained on states the data collectors produced. The policy visits new states, and wherever the classifier is wrong in its favor, RL will find it: the cube held up to the camera so it *looks* placed, an occlusion that hides the failure, a lighting artifact. It is the critic-exploitation story of Lecture 07 again, and model exploitation in Lecture 09 will be the same story once more.

Defenses, roughly in order of cost:

1. **Hard negatives from the policy itself.** Periodically label policy rollouts (free in sim, human queries on a robot) and retrain.
2. **Ensembles and confidence thresholds.** Reward only when all members agree.
3. **Keep the policy close to the data** with a KL leash, so it cannot wander far into regions where the classifier is untested.
4. **Audit with a different signal.** Ground-truth sim checks, held-out human review. Never evaluate with the same detector you trained against.

---

## 7. Robot connection

For π0.5 on LIBERO, the simulator's success predicate is free and exact, so you do not need a learned reward for the capstone. But you need this lecture's tools in three places: the **KL leash** that protects general skills (an ELBO-style trade-off between reward and staying near the prior), the **entropy** that keeps exploration alive, and the honest question of **what the score actually measures**. A hidden test that moves objects is, in effect, a held-out audit of whatever shortcut your training signal allowed. A policy that learned "move the gripper to where the plate usually is" satisfies a sloppy detector and fails the moved-plate test.

---

## 8. Hardware view: another network in the loop

A learned reward turns the rollout into **policy + simulator + reward model**. Extending Lecture 01's iteration time, with a reward model invoked every \(k\) steps:

$$
t_{\text{iter}} \approx K \Big( t_{\text{env}}(N) + t_{\text{render}}(N) + t_{\text{policy}}(N) + \tfrac{1}{k}\, t_{\text{reward}}(N) \Big) + t_{\text{learn}} + t_{\text{disc}}
$$

| Component | Invoked | Cost driver | Typical regime |
|---|---|---|---|
| Camera rendering for the detector | Every step it scores | Resolution × cameras × envs; often the real cost (Lecture 01, Lab 1b) | B, with pixels |
| Small CNN success classifier | Every step, or only at episode end | Negligible next to rendering at 64×64 | B |
| VLM reward model / judge | Per chunk or per episode | Hundreds of millions to billions of parameters; a second regime-C bottleneck | C |
| GAIL / VICE discriminator training | Every iteration, on fresh policy data | Competes with the learner for GPU time and memory | B/C |
| VAE encoder (if the policy uses a latent) | Every step | Small; KL terms are closed-form and free | — |

Rules of thumb:

* **Score per episode when the task allows it.** A sparse success reward only needs the final frame. Scoring every step multiplies reward-model cost by \(T\).
* **Batch the reward model like the policy.** In regime C, a VLM judge wants the same batching and CUDA-Graph treatment as the VLA itself.
* **Budget memory.** Policy + reference + critic + reward model + discriminator may all need to be resident. Each extra network is a memory-allocation decision, and the reason GRPO-style setups that drop the critic exist (Lecture 11).
* **Classifier datasets are pixel datasets.** 200 k frames at 128×128×3 uint8 is ~9.8 GB; downsampled to 64×64 it is ~2.5 GB, which fits on the GPU.

---

## 9. Build it

All code goes in `prob_toolkit/`.

### Lab 8a — A VAE on action chunks, with the gap measured

Use the expert demonstrations from Lecture 02. Cut each episode's actions into overlapping chunks of \(H = 10\) steps × 7 dimensions.

```python
# prob_toolkit/vae_chunks.py
import math, torch, torch.nn as nn, torch.nn.functional as F
H, A, Z, SIGMA_X = 10, 7, 8, 0.1
D = torch.load("bc_dagger/demos.pt")             # saved in Lecture 02 Ship it
demos = list(D["A"][D["success"]])               # successful episodes, each (T, 7)
X = torch.cat([d.unfold(0, H, 1).transpose(1, 2).reshape(-1, H * A) for d in demos])

enc = nn.Sequential(nn.Linear(H * A, 256), nn.ReLU(), nn.Linear(256, 2 * Z))
dec = nn.Sequential(nn.Linear(Z, 256), nn.ReLU(), nn.Linear(256, H * A))
opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-3)

def log_px_given_z(x, z):                       # Gaussian decoder, fixed std SIGMA_X
    return (-0.5 * ((x - dec(z)) / SIGMA_X) ** 2 - math.log(SIGMA_X * math.sqrt(2 * math.pi))).sum(-1)

def elbo_terms(x):
    mu, logvar = enc(x).chunk(2, -1)
    z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)          # reparameterization
    rec = log_px_given_z(x, z)
    kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(-1)      # closed form vs N(0, I)
    return rec, kl, mu, logvar

@torch.no_grad()
def iwae(x, K=256):                              # tighter estimate of log p(x)
    mu, logvar = enc(x).chunk(2, -1); std = (0.5 * logvar).exp()
    z = mu + std * torch.randn(K, *mu.shape)
    log_q = (-0.5 * ((z - mu) / std) ** 2 - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_p = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1) + log_px_given_z(x, z)
    return torch.logsumexp(log_p - log_q, 0) - math.log(K)

for step in range(20_000):
    x = X[torch.randint(0, len(X), (512,))]
    rec, kl, _, _ = elbo_terms(x)
    loss = -(rec - kl).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        elbo = (rec - kl).mean().item(); lp = iwae(x[:64]).mean().item()
        print(f"{step:6d}  rec {rec.mean():8.2f}  KL {kl.mean():6.2f}  ELBO {elbo:8.2f}  "
              f"IWAE {lp:8.2f}  gap~{lp - elbo:5.2f} nats")
```

Plot reconstruction, KL, ELBO, and the IWAE estimate over training. The IWAE − ELBO difference approximates the KL gap from section 2.3. Then: (1) sweep \(\beta\) on the KL term in {0.1, 1, 4} and watch rate trade against reconstruction; (2) sample \(z \sim \mathcal{N}(0, I)\), decode, and roll the decoded chunks out open-loop in PickCube. Do they look like plausible approach motions?

### Lab 8b — A success classifier with free labels

Collect frames from a mix of policies (random, early and late checkpoints from Lectures 06-07, plus action noise) so both outcomes are well represented. The goal marker in `PickCube-v1` is hidden from the cameras, so the classifier also gets `goal_pos`, the same way a real success detector gets the instruction.

```python
# prob_toolkit/collect.py — frames + goal + ground-truth success, labels free from sim
import torch, torch.nn.functional as F, gymnasium as gym, mani_skill.envs
env = gym.make("PickCube-v1", num_envs=256, obs_mode="state_dict+rgb", control_mode="pd_ee_delta_pose")
def flat_state(obs):                            # print(obs.keys()) first; layout can differ by version
    return torch.cat([v.reshape(v.shape[0], -1).float() for part in ("agent", "extra") for v in obs[part].values()], -1)

frames, goals, labels, episode_ids = [], [], [], []
for ep_batch in range(40):                       # 40 × 256 × 50 ≈ 512 k steps; keeping every 5th stores ≈ 100 k frames
    obs, _ = env.reset(seed=ep_batch)
    policy = pick_policy(ep_batch)               # yours: random / checkpoint / checkpoint + noise
    for t in range(50):
        obs, _, _, _, info = env.step(policy(flat_state(obs)))
        if t % 5 == 4:
            rgb = obs["sensor_data"]["base_camera"]["rgb"].permute(0, 3, 1, 2).float()
            frames.append(F.interpolate(rgb, size=64, mode="area").to(torch.uint8).cpu())
            goals.append(obs["extra"]["goal_pos"].cpu()); labels.append(info["success"].cpu())
            episode_ids.append(ep_batch * 256 + torch.arange(256))
torch.save(dict(frames=torch.cat(frames), goals=torch.cat(goals), labels=torch.cat(labels),
                episode_ids=torch.cat(episode_ids)), "success_data.pt")
```

Train a small CNN on (frame, goal_pos) → success with `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`, because successes are rare. **Split train/test by episode, not by frame**: neighboring frames from one episode are near-duplicates, and a frame-level split leaks. Report precision, recall, and the confusion matrix on held-out episodes at threshold 0.5, then sweep the threshold and plot the PR curve.

### Lab 8c — Optimize against it and catch the hack

Take your Lab 7c SAC and swap the reward. The policy still sees privileged state; only the reward comes from pixels. Sketch of the change (adapt it to your loop; with the dict observation, flatten the state for the actor and buffer):

```python
# inside the SAC rollout loop, replacing the env reward
with torch.no_grad():
    rgb = F.interpolate(nobs["sensor_data"]["base_camera"]["rgb"].permute(0, 3, 1, 2).float(),
                        size=64, mode="area")
    p_success = torch.sigmoid(classifier(rgb / 255.0, nobs["extra"]["goal_pos"]))
    rew = (p_success > 0.5).float()                      # learned sparse reward
true_success = info["success"].float()                   # logged, never used for training
```

Train, and log **classifier-claimed success** and **true success** side by side. The gap between them is your reward-hacking measure. Find at least one episode where the classifier says success and the simulator says failure, save its frames, and explain what the policy found (common: cube lifted and held, but not at the goal height; gripper occluding the cube; the robot not yet static, which the sim's predicate requires). Then apply defense 1 from section 6: add the policy's false positives as labeled negatives, retrain, and repeat. Report how the gap changes over two rounds.

---

## 10. Use it in the real stack

* **Generative action heads.** Diffusion Policy and flow-matching experts (π0, π0.5 via [openpi](https://github.com/Physical-Intelligence/openpi)) are the stacked-VAE family of section 3.3. ACT uses a conditional VAE over action chunks directly.
* **Max-entropy RL.** SAC in CleanRL / Stable-Baselines3 / ManiSkill baselines is control as inference in production. The `autotune` temperature is section 4's \(\alpha\).
* **Adversarial imitation.** GAIL/AIRL implementations exist in the `imitation` library (built on Stable-Baselines3). Adversarial motion priors in legged robotics use the same discriminator-as-reward structure for style.
* **Reward models.** VLM-based success detectors and judges score robot episodes in research pipelines and data-filtering systems. Treat every one as a component with a measured precision/recall and a known failure set, never as ground truth.

---

## 11. Measure it

| Metric | How | What it tells you |
|---|---|---|
| **ELBO, reconstruction, KL (nats)** | Logged per step | Whether the model learns, and how much information \(z\) carries |
| **IWAE − ELBO** | Lab 8a | Size of the variational gap; how good the encoder's guess is |
| **Classifier precision / recall / PR curve** | Held-out episodes, sim ground truth | Detector quality *on the data distribution it was trained on* |
| **Claimed vs true success gap** | Lab 8c, per training checkpoint | Reward hacking; should grow if left unchecked |
| **Reward-model latency and share of \(t_{\text{iter}}\)** | Time render + classifier separately, with `cuda.synchronize()` | Whether the learned reward is a bottleneck |
| **Rendering cost** | env-steps/s with `obs_mode="state"` vs `"state_dict+rgb"` | What pixels for the reward cost the rollout |
| **Peak GPU memory** | `max_memory_allocated` with and without the classifier | Memory price of an extra network |

---

## 12. Ship it

Commit `prob_toolkit/` containing:

* `vae_chunks.py` + `elbo_terms.png` (reconstruction, KL, ELBO, IWAE over training; β sweep)
* `collect.py`, `train_classifier.py` + `confusion_matrix.png` + `pr_curve.png` (episode-level split)
* `sac_learned_reward.py` + `claimed_vs_true.png` (both success curves on one axis, ≥ 3 seeds)
* `REWARD_HACKING.md`: the saved frames of one classifier-fooling episode, what the policy exploited, the retraining rounds and their effect on the gap, and one paragraph on the reward model's measured share of wall-clock

---

## Exit criteria

You can move on when you can:

* derive the ELBO with Jensen and with the KL-gap identity, and say what each term means for a VAE
* explain, with an example from this course, when you must use the score-function gradient and when reparameterization is available
* explain why a diffusion model is a hierarchical VAE and why a flow policy's exact log-likelihood is expensive
* derive \(\pi \propto \exp(Q/\alpha)\) and the soft value from control as inference, and show SAC's actor loss is a KL projection
* write the MaxEnt IRL gradient and the GAIL objective, and say why GAIL's discriminator is not a reusable reward
* show, from your own runs, a learned reward being exploited and a measured gap shrinking after retraining

---

## Self-check

1. During VAE training, your ELBO improves by 5 nats but the IWAE estimate of \(\log p(x)\) does not move. Which of the two things the ELBO optimizes improved, and which didn't? What does that say about the gap?
2. You want the gradient of expected *true sim success* with respect to the parameters of a Gaussian policy. Can you use the reparameterization trick? What if you replace sim success with a learned differentiable critic? What do you gain and what new risk appears?
3. In control as inference, what happens to the policy as \(\alpha \to 0\) and as \(\alpha \to \infty\)? Why does exact inference over trajectories give an optimistic backup, and what assumption fixes it?
4. A GAIL-trained policy fools its discriminator 50% of the time (equilibrium) but succeeds at the task only 20% of the time. Give two reasons this is possible, and explain why you can't reuse the final discriminator as a reward for a new layout.
5. Your success classifier has 0.97 precision and 0.90 recall on held-out episodes. After RL against it, the classifier reports 80% success and the simulator reports 12%. Explain how both measurements can be correct, and name the cheapest fix in simulation and the cheapest fix on a real robot.
6. A VLM success judge costs 80 ms per call at batch 1. A LIBERO episode is 300 steps, executed in chunks of \(H = 10\). Compare reward-model time per episode for per-step, per-chunk, and per-episode scoring. Which does a sparse binary task need?

---

## References

* Kingma & Welling, "Auto-Encoding Variational Bayes," 2013 — [arXiv:1312.6114](https://arxiv.org/abs/1312.6114); Rezende, Mohamed, Wierstra, "Stochastic Backpropagation and Approximate Inference in Deep Generative Models," 2014 — [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)
* Blei, Kucukelbir, McAuliffe, "Variational Inference: A Review for Statisticians," 2016 — [arXiv:1601.00670](https://arxiv.org/abs/1601.00670)
* Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models," 2020 — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
* Kingma, Salimans, Poole, Ho, "Variational Diffusion Models," 2021 — [arXiv:2107.00630](https://arxiv.org/abs/2107.00630)
* Levine, "Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review," 2018 — [arXiv:1805.00909](https://arxiv.org/abs/1805.00909)
* Haarnoja, Tang, Abbeel, Levine, "Reinforcement Learning with Deep Energy-Based Policies" (soft Q-learning), 2017 — [arXiv:1702.08165](https://arxiv.org/abs/1702.08165)
* Ziebart, Maas, Bagnell, Dey, "Maximum Entropy Inverse Reinforcement Learning," AAAI 2008
* Ho & Ermon, "Generative Adversarial Imitation Learning," 2016 — [arXiv:1606.03476](https://arxiv.org/abs/1606.03476)
* Fu, Luo, Levine, "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning" (AIRL), 2017 — [arXiv:1710.11248](https://arxiv.org/abs/1710.11248)
* Fu et al., "Variational Inverse Control with Events" (VICE), 2018 — [arXiv:1805.11686](https://arxiv.org/abs/1805.11686); Singh et al., "End-to-End Robotic Reinforcement Learning without Reward Engineering," 2019 — [arXiv:1904.07854](https://arxiv.org/abs/1904.07854)
* Rocamonde et al., "Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning," 2023 — [arXiv:2310.12921](https://arxiv.org/abs/2310.12921)
* Amodei et al., "Concrete Problems in AI Safety" (reward hacking), 2016 — [arXiv:1606.06565](https://arxiv.org/abs/1606.06565)

---

## Next in this special course

* Next: [Lecture 09 — Model-Based RL and World Models](Lecture-09.md)
* Previous: [Lecture 07 — Value-Based and Off-Policy RL: From Q-Learning to SAC](Lecture-07.md)
* Back: [Deep RL for Robot Learning — Overview](README.md)
