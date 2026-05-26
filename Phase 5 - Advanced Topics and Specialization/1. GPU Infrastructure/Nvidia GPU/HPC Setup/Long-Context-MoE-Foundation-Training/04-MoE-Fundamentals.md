# Module 04 — MoE Fundamentals

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Understand the Mixture-of-Experts FFN layer at the level of math, training stability, and the auxiliary losses that keep the router from collapsing.

**Prerequisites:** Comfortable with the dense transformer FFN block. Familiarity with softmax + cross-entropy.

**Artifact:** A working top-k MoE FFN layer in PyTorch (single-GPU is fine), with an ablation showing how the auxiliary-loss weight changes per-expert utilization.

---

## Why it matters

MoE is the dominant way to scale model capacity without paying full dense compute per token. Modern frontier-class models (DeepSeek-V2/V3, Mixtral, Qwen-MoE, GPT-OSS variants) are MoE. If you do not understand the router math and the load-balancing failure modes, you will hit training instability that looks like the kernel is broken but is actually a routing pathology.

The systems-side of MoE (expert parallelism, all-to-all, capacity factor at scale) is Module 05. This module is the **algorithmic** foundation that the systems work assumes.

---

## Mental model

### A dense FFN block

```
x ∈ ℝ^H
h = gelu(x · W_up)        # W_up ∈ ℝ^{H × 4H}
y = h · W_down           # W_down ∈ ℝ^{4H × H}
```

Every token touches every parameter. Parameters per FFN block: `8 H²`. FLOPs per token: `16 H²`.

### A top-k MoE FFN block

Replace the single FFN with `E` independent expert FFNs and a router that picks `k` of them per token.

```
x ∈ ℝ^H
logits = x · W_r          # W_r ∈ ℝ^{H × E}, router weights
g = softmax(logits)       # routing distribution
topk_indices = argtopk(g, k)
topk_weights = g[topk_indices]                  # gate values for chosen experts
topk_weights = topk_weights / topk_weights.sum() # renormalize over selected
y = Σ_{i in topk_indices} topk_weights[i] · Expert_i(x)
```

Parameters per FFN block: `E · 8H² + H · E ≈ E · 8H²`. FLOPs per token (compute side): `k · 16 H²`. **Parameter count is `E`× the dense block; compute is only `k`× the dense block.**

Example: Mixtral 8×7B has `E = 8, k = 2`. Total parameters per FFN ≈ 8× dense; activated parameters per token ≈ 2× dense.

### Why MoE is unstable

The router is just a softmax over expert indices. If it learns early that "expert 3 is good", it sends everything to expert 3, expert 3 gets trained, the others starve. Without intervention this collapses to a dense-on-one-expert model with extra dead weights.

The fix is **auxiliary losses** that push the routing distribution toward uniformity.

### Load-balancing auxiliary loss

Define:

- `f_i = fraction of tokens routed to expert i` (per micro-batch)
- `P_i = mean routing probability for expert i = mean over tokens of g[i]`

Loss term: `L_aux = α · E · Σ_i f_i · P_i`. With uniform routing, `f_i = 1/E` and `P_i = 1/E`, so `L_aux = α`. With collapsed routing, `f_3 = 1` and `P_3 ≈ 1`, so `L_aux = α · E` — much larger. The gradient pushes the router toward uniform usage.

Typical `α ∈ [0.001, 0.01]`. Too low: collapse. Too high: routing becomes random, model loses the benefit of specialization.

### Router z-loss

A second stabilizer from the Switch Transformer paper. The router's logits can drift to large magnitudes that destabilize the softmax. Add:

```
L_z = β · mean_over_tokens( log(Σ_i exp(logits_i))² )
```

With small `β` (e.g. `1e-3`). Keeps the router's logsumexp bounded.

### Capacity factor and dropped tokens

When you batch many tokens, expert `i` receives some number of tokens. To enable efficient batched expert execution, you cap the per-expert capacity at:

```
capacity = ceil(capacity_factor · tokens_per_batch · k / E)
```

`capacity_factor` is typically `1.0` to `1.5`. Tokens beyond capacity for their chosen expert are **dropped** — they skip the FFN and go through with just the residual (in some implementations) or with all-zeros (in others). Dropped-token rate is a key diagnostic: should be 0–5%. If higher, the load balancer isn't doing its job or capacity is too tight.

### Two routing styles

- **Token-choice top-k** (this module's default): each token picks its top `k` experts.
- **Expert-choice**: each expert picks its top `M` tokens. Inverts the bottleneck; tokens may go to zero or many experts. Used by some recent MoE variants. Trade-offs: no dropped tokens, but no per-token guarantee of FFN coverage.

Token-choice is more common and the only style most production MoE implementations support out of the box. Mention expert-choice in your notes but use token-choice for everything else in this course.

### Why MoE matches long-context training

- Long context is already compute-heavy; you cannot afford a `2×` dense scale-up.
- MoE gives you `8×` capacity with only `2×` compute.
- Experts can specialize on long-form patterns (code, prose, math, dialog) — useful because long-context data is heterogeneous by definition.
- The systems trade-off is communication: all-to-all dispatch costs grow with token count, which is exactly what long context produces a lot of. That's Module 05.

---

## Build it

A self-contained PyTorch implementation that's small enough to read and large enough to show the routing pathologies.

```python
# minimal_moe.py
import torch, torch.nn as nn, torch.nn.functional as F

class TopKMoE(nn.Module):
    def __init__(self, H, expert_dim, num_experts=8, top_k=2,
                 aux_weight=0.01, z_weight=1e-3, capacity_factor=1.25):
        super().__init__()
        self.E, self.k = num_experts, top_k
        self.router = nn.Linear(H, num_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(H, expert_dim), nn.GELU(), nn.Linear(expert_dim, H))
            for _ in range(num_experts)
        )
        self.aux_weight, self.z_weight = aux_weight, z_weight
        self.capacity_factor = capacity_factor

    def forward(self, x):
        # x: [B, S, H]
        B, S, H = x.shape
        flat = x.reshape(-1, H)             # [T, H], T = B*S
        T = flat.size(0)

        logits = self.router(flat)          # [T, E]
        g = F.softmax(logits, dim=-1)
        topk_w, topk_idx = g.topk(self.k, dim=-1)    # [T, k]
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        # Capacity per expert
        cap = int(self.capacity_factor * T * self.k / self.E + 1)
        out = torch.zeros_like(flat)
        dropped = 0
        for e in range(self.E):
            # tokens that picked expert e in any slot, ordered by gate value
            mask = (topk_idx == e)                    # [T, k] bool
            tok_pos = mask.any(-1).nonzero(as_tuple=True)[0]
            # priority by max gate weight among the k slots for that token
            scores = (topk_w * mask).sum(-1)[tok_pos]
            order = scores.argsort(descending=True)
            keep = tok_pos[order][:cap]
            dropped += max(0, tok_pos.size(0) - cap)
            if keep.numel() == 0:
                continue
            xe = flat[keep]
            ye = self.experts[e](xe)
            # weight: pick the matching gate value from topk_w
            w_e = (topk_w * mask).sum(-1)[keep].unsqueeze(-1)
            out.index_add_(0, keep, w_e * ye)

        # Aux loss (load balancing)
        f = torch.zeros(self.E, device=x.device)
        for e in range(self.E):
            f[e] = (topk_idx == e).any(-1).float().mean()
        P = g.mean(0)                                  # [E]
        aux_loss = self.E * (f * P).sum() * self.aux_weight

        # Router z-loss
        z = torch.logsumexp(logits, dim=-1)
        z_loss = (z ** 2).mean() * self.z_weight

        return out.reshape(B, S, H), aux_loss + z_loss, dropped / max(T, 1)

if __name__ == "__main__":
    torch.manual_seed(0)
    moe = TopKMoE(H=256, expert_dim=1024).cuda()
    x = torch.randn(4, 128, 256, device="cuda")
    y, aux, drop_rate = moe(x)
    print("out", y.shape, "aux_loss", aux.item(), "dropped%", drop_rate * 100)
```

Now build the training-loop ablation:

```python
# moe_ablation.py
# Train this MoE on a toy task (e.g. learn to reconstruct shuffled MNIST flattened patches)
# Sweep aux_weight in {0, 1e-4, 1e-3, 1e-2, 1e-1}.
# For each, log per-expert token-share over training.
# Plot per-expert usage curves.
```

Expected outcomes:

- `aux_weight = 0`: one or two experts dominate after a few hundred steps; usage curve collapses.
- `aux_weight = 1e-3`: usage stays approximately uniform with small drift.
- `aux_weight = 1e-1`: usage is artificially flat, but model loss stalls because routing is essentially random.

Save the plot. It is the most useful diagnostic you will own when MoE training goes wrong.

---

## Use it in the real stack

- **Megatron-LM MoE**: `--num-experts`, `--moe-router-topk`, `--moe-aux-loss-coeff`, `--moe-z-loss-coeff`, `--moe-expert-capacity-factor`. Documented at <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md>.
- **DeepSpeed-MoE**: <https://www.deepspeed.ai/tutorials/mixture-of-experts/>. Similar concepts, different naming.
- **`transformers` MoE classes**: `MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `DeepseekV3MoE`. Read the forward methods — they implement exactly what you wrote, with the systems-side gather/scatter for expert parallelism.

Read at least one of these MoE forwards end to end. Map each variable to the math above. If a variable's purpose is unclear, you have a gap to close.

---

## Measure it

During your ablation:

- **Per-expert token share** over training (line plot, one line per expert).
- **Dropped-token rate** per step (should be < 5% with `capacity_factor = 1.25`).
- **Aux loss value** — should be near `α` if routing is balanced; much larger if collapsing.
- **Task loss** — confirms that an over-regularized router (huge `α`) hurts learning.

A healthy MoE has roughly uniform per-expert usage, near-`α` aux loss, low dropped-token rate, and a task loss curve that matches or beats a dense baseline with the same activated parameter count.

---

## Ship it

Drop into `lcm-course/`:

1. `minimal_moe.py` and `moe_ablation.py` with logs.
2. `moe_expert_usage.png` — the per-expert usage curves across the aux-weight sweep.
3. `moe_notes.md` — one paragraph each on top-k routing math, aux loss, z-loss, capacity factor, and at least one named failure mode you actually triggered in your ablation (e.g. "with aux_weight=0, expert 2 received 87% of all tokens by step 500").

---

## Related pages

- [Module 05 — MoE systems and infrastructure](05-MoE-Systems-Infrastructure.md)
- [Module 08 — Combining long-context and MoE](08-Combining-LongContext-and-MoE.md)
- Switch Transformer paper (router z-loss + load balancing): <https://arxiv.org/abs/2101.03961>
- Mixtral of Experts paper: <https://arxiv.org/abs/2401.04088>
- DeepSeek-V2 MoE design: <https://arxiv.org/abs/2405.04434>
