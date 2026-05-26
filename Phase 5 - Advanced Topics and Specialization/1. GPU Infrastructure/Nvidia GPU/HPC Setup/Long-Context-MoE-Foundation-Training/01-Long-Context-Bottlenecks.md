# Module 01 — Why Long Context Is Hard

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Build the scaling intuition that explains exactly which compute and memory costs grow with context length, where the crossovers happen, and why "make the context window bigger" is the easy half of the problem.

**Prerequisites:** Transformer fundamentals. The FlashAttention course's Lecture 1 (roofline / IO model) is strongly recommended.

**Artifact:** A table comparing per-layer compute and per-layer activation memory for attention vs MLP at `N ∈ {4K, 32K, 128K, 1M}` for one realistic model config, plus a one-paragraph conclusion identifying the dominant cost in each regime.

---

## Why it matters

Long context is **two distinct problems** that people often conflate:

1. **Making the model run** at long sequence length without running out of memory or time.
2. **Making the model actually use** the long context, instead of treating tokens past some position as background noise.

The systems work covers problem (1). Problem (2) requires positional encoding choices, training curriculum, evaluation, and data — covered in modules 03, 06, and 07. You cannot solve problem (2) without first being able to physically train at the target length.

---

## Mental model

### Per-layer compute scaling

For a transformer block with hidden size `H`, number of heads `H_q`, head dim `D`, sequence length `N`, MLP expansion `4H`:

| Component | FLOPs per token | Scales with N as |
|-----------|------------------|------------------|
| QKV projection | `~6 H²` | linear |
| Attention `QKᵀ` + `PV` | `~4 N · D · H_q = 4 N · H` (with `H = H_q · D`) | **linear per token**, **quadratic per sequence** |
| Attention softmax | `~5 N` | linear per token |
| Output projection | `~2 H²` | linear |
| MLP up + down (dense) | `~16 H²` | linear |

Per **sequence** of length `N`, total per-layer compute:

```
Attention :  ~ 4 N² · H        (the N² term)
MLP       :  ~ 24 N · H²       (linear in N, quadratic in H)
```

Crossover: attention overtakes MLP when `4N² H > 24 N H²`, i.e. `N > 6 H`. For `H = 4096`, that is `N > 24576`. For `H = 8192`, `N > 49152`. So:

- At 4K context: MLP dominates. Standard "scale H" intuition holds.
- At 32K+ context with `H = 4096`: attention dominates compute.
- At 1M context: attention is overwhelmingly dominant; MLP cost is rounding error.

### Per-layer activation memory scaling

Activations stored for the backward pass per layer per sequence:

| Component | Bytes (bf16) | Scales as |
|-----------|---------------|-----------|
| Q, K, V activations | `6 N H` | linear in N |
| Attention scores (without FlashAttention) | `2 N²` per head, summed | **quadratic** |
| Attention output | `2 N H` | linear |
| MLP intermediate | `2 N · 4H = 8 N H` | linear |

The `N²` attention scores matrix is what kills you. FlashAttention removes this term from materialized memory by computing the softmax online (see the FlashAttention course). With FlashAttention, attention activation memory becomes `O(N H)` — linear, not quadratic.

But activations are still dominated by `N H` × (number of layers). For `N = 128K, H = 8192, L = 80, bf16`, that is `128_000 · 8192 · 80 · 2 ≈ 168 GB` per sequence — already too large to keep without sequence parallel or activation recomputation.

### KV cache scaling (inference side, mentioned here for completeness)

Per-token KV size at inference: `2 · H_kv · D · 2 bytes` (K + V, bf16). For 32K context with `H_kv = 8, D = 128`: `2 · 8 · 128 · 32000 · 2 = ~130 MB` per layer per sequence. Across 80 layers that's ~10 GB per sequence — what you saw with the 72B inference work.

### What needs to change as context grows

| Regime | Bottleneck | Required techniques |
|--------|------------|---------------------|
| 4K | MLP compute | TP, ZeRO, mixed precision |
| 32K | Attention compute starts to dominate | FlashAttention |
| 128K | Activation memory, attention compute | + activation recomputation, sequence parallel |
| 1M | Attention activation, communication | + context parallel (ring/striped), longer pipeline, very careful overlap |

Long context is not one technique; it is a stack where each layer is necessary at a different N.

### The accuracy problem ("usable" context)

A separate axis from systems: a model that **accepts** 128K tokens often **uses** the first ~16K and the last ~2K and ignores the middle. This is documented in many "lost in the middle" papers. The fix is not in the kernel — it is in:

- Position encoding generalization (Module 03).
- Length curriculum during continual pretraining (Module 06).
- Training data that requires distant evidence (Module 06).
- Honest evaluation that measures position-by-position retrieval (Module 07).

The two halves of long-context work — systems and accuracy — must be solved together. A model that physically supports 1M tokens but cannot retrieve a fact at position 500K is useless.

---

## Build it

### Compute & activation table

```python
# scaling_table.py
def stats(N, H, L, H_kv, D, bf16=2):
    attn_flops_per_seq = 4 * N * N * H * L
    mlp_flops_per_seq  = 24 * N * H * H * L

    flash_attn_act = (6 * N * H + 2 * N * H) * L * bf16    # Q,K,V + O
    mlp_act        = 8 * N * H * L * bf16
    naive_attn_extra = (2 * N * N) * (H // D) * L * bf16    # the N^2 score per head

    return dict(
        N=N,
        attn_TFLOPs=attn_flops_per_seq / 1e12,
        mlp_TFLOPs=mlp_flops_per_seq / 1e12,
        attn_dom=attn_flops_per_seq > mlp_flops_per_seq,
        flash_act_GB=flash_attn_act / 1e9,
        mlp_act_GB=mlp_act / 1e9,
        naive_extra_GB=naive_attn_extra / 1e9,
    )

# Example: 8B-class model
H, L, H_kv, D = 4096, 32, 8, 128
for N in [4096, 32768, 131072, 1_000_000]:
    s = stats(N, H, L, H_kv, D)
    print(s)

# 72B-class model
H, L, H_kv, D = 8192, 80, 8, 128
for N in [4096, 32768, 131072]:
    s = stats(N, H, L, H_kv, D)
    print(s)
```

Save the table. For each model size, write one sentence: "Crossover from MLP-dominated to attention-dominated happens at N ≈ X." For each row, write the dominant activation memory category.

### Reality check

Multiply the activation memory by your batch size and pipeline stage count. For a realistic mid-training run (`batch = 1024 sequences, micro_batch_size = 2, pipeline_parallel = 4`) you can quickly land at hundreds of GB of activations per stage. Note where each row first exceeds your available HBM per GPU — that's the point at which the technique to your right in the regime table (Module 02 / 09) becomes mandatory.

---

## Use it in the real stack

NVIDIA's [MoE Long-Context Training skill](https://docs.nvidia.com/nemo/megatron-bridge/nightly/skills/perf-techniques/moe-long-context/SKILL.html) gives concrete config tables for Megatron Bridge: what context-parallel size, activation recomputation, FP8, and offload combinations are needed at each N for different model sizes. Open it. Match each entry to a row of your table above. Where it disagrees with your back-of-envelope, you found something to read more carefully.

Meta's [Effective Long-Context Scaling](https://arxiv.org/abs/2309.16039) is the canonical example of continual pretraining to extend a base model from 4K to 32K. The systems lessons in that paper (data mix, curriculum, RoPE base change) feed directly into modules 03 and 06.

---

## Measure it

- Per-row attention vs MLP TFLOPs ratio.
- Per-row activation memory in GB, broken down by (FlashAttention activations, MLP activations, naive-attn extra).
- Per-row "first technique you need to add" (recomputation, sequence parallel, context parallel, offload).

For at least one (N, H, L) combination, compute the wall-clock you would expect on your hardware: `total_FLOPs / (per_GPU_TFLOPs · num_GPUs · 0.5)` (the 0.5 is realistic utilization). This number tells you whether a training run is hours or weeks.

---

## Ship it

In your `lcm-course/` working dir:

1. `scaling_table.py` and its `scaling_table.csv` output for at least two model sizes.
2. `bottleneck_notes.md` with one paragraph per regime explaining which cost dominates and which technique you would reach for first.
3. A one-paragraph statement of how the "physically supports N tokens" axis is separate from the "actually uses N tokens" axis, in your own words.

These three are the framing for everything that follows.

---

## Related pages

- [Module 02 — Long-context attention mechanics](02-Long-Context-Attention.md)
- [FlashAttention Course — Lecture 1](../../../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/Lecture%2001%20-%20Attention%20Bottleneck%20and%20Roofline.md)
- NeMo Megatron Bridge: <https://docs.nvidia.com/nemo/megatron-bridge/nightly/>
