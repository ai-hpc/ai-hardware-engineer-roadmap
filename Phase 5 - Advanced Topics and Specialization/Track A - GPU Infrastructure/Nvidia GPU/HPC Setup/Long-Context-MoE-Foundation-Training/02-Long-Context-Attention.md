# Module 02 — Long-Context Attention Mechanics

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Make long-context attention physically tractable: FlashAttention for the inner kernel, sequence parallelism for the inter-block reduction, context parallelism for splitting one sequence across many GPUs, plus activation recomputation as the last-resort memory tool.

**Prerequisites:** Module 01. FlashAttention course Lectures 1–3.

**Artifact:** A working benchmark of FlashAttention with context-parallel attention on one 8-GPU node, plus an activation-memory plot across sequence lengths comparing (no CP) vs (CP=2) vs (CP=4) vs (CP=4 + recompute).

---

## Why it matters

You cannot train a long-context model without three things working together: a kernel that does not materialize `N²` scores, a way to split activations across many GPUs without serializing the math, and a way to spend a little extra compute to fit in HBM when activations are still too big. Each of these is a separate technique with separate failure modes. This module covers all three at the level needed to debug a real training run.

---

## Mental model

### Layer 1 — FlashAttention removes the `N²` memory term

You covered this in detail in the FlashAttention course. The summary for this module:

- Forward: tile `Q, K, V` so `S = QKᵀ` stays in SRAM; track running `(m, ℓ)` per query row; produce `O` directly.
- Backward: recompute `S, P` per tile using the saved `LSE`; same `O(N²d²/M)` HBM bound.
- For long-context training, FlashAttention is mandatory — without it the activation memory of attention alone exceeds your HBM at `N ≥ 16K`.

In practice you use `flash_attn_func` for fixed-length batches and `flash_attn_varlen_func` for packed variable-length sequences. Both work with sequence parallelism.

### Layer 2 — Sequence parallelism (Megatron-style)

When you tensor-parallelize a transformer, you split the hidden dimension `H` across TP GPUs. The LayerNorm + dropout + residual paths are **not** tensor-parallelized — by default they replicate computation and activations across TP ranks.

**Sequence parallel (SP)** changes that: split those `O(N·H)` non-matmul activations along the sequence dimension `N` instead of replicating. Now each TP rank stores `N/TP × H` activations instead of `N × H`. You pay an extra all-gather before column-parallel ops and an extra reduce-scatter after row-parallel ops, but the activation memory savings dominate.

SP is "free" memory-wise on top of TP. Enable it on any TP run for `N ≥ 8K`.

### Layer 3 — Context parallelism (CP) for splitting one sequence

SP splits non-matmul activations across TP. It does **not** split the attention computation itself — every TP rank still sees the full sequence at the attention layer. For `N` so large that even one head's worth of attention activations does not fit, you need to split the sequence itself across additional GPUs.

That is **context parallel** (CP), sometimes called sequence-dimension model parallelism. Variants:

- **Ring attention**: each CP rank owns a contiguous chunk of the sequence's KV. During attention, KV chunks rotate through ranks like a ring; each rank computes attention between its `Q` chunk and the visiting `KV` chunk; partial results are combined with the FlashAttention online-softmax recurrence.
- **Striped attention** (a refinement): instead of contiguous chunks, interleave so each rank has a striped subset of positions. Improves load balance under causal masking.
- **DistFlashAttn / Sequence-Parallel FlashAttention**: similar idea, slightly different scheduling and overlap patterns.

CP combines with FlashAttention: each rank runs the standard FlashAttention forward over its `Q_local` against each visiting `KV_chunk`, then merges via the same `(m, ℓ)` rescale you used in Module 02 of the FlashAttention course. The math is exact.

Communication cost: O(N · H) per round, O(CP_size − 1) rounds per layer. Hidden behind compute if CP_size is small and the chunk is large.

### Layer 4 — Activation recomputation (selective and full)

When you still cannot fit, recompute. Two flavors:

- **Selective recomputation**: store only specific activations (typically attention's `Q, K, V, O, LSE`), recompute the rest in backward. Cheap — costs ~10–20% extra forward FLOPs.
- **Full recomputation**: store only the layer input; recompute the entire layer forward in backward. Expensive — ~30–40% extra FLOPs.

Megatron supports both via `--recompute-granularity selective|full`. At long context, selective recomputation is usually enough alongside SP+CP.

### When to reach for what

| Symptom | Reach for |
|--------|-----------|
| OOM at TP=8, N=8K, no SP | Sequence parallel |
| OOM at TP=8 + SP, N=64K | Context parallel CP=2 |
| OOM at TP=8 + SP + CP=2, N=256K | Increase CP, then add selective recomputation |
| TFLOPs collapse after enabling CP=8 | Communication is unhidden — check overlap settings, increase chunk size |

---

## Build it

### 1. FlashAttention on a single 8-GPU node

```bash
# Megatron-LM example, 8-GPU node, TP=8, no PP, no DP
torchrun --nproc-per-node=8 pretrain_gpt.py \
  --num-layers 32 --hidden-size 4096 --num-attention-heads 32 \
  --seq-length 32768 --max-position-embeddings 32768 \
  --tensor-model-parallel-size 8 \
  --use-flash-attn --bf16 \
  --micro-batch-size 1 --global-batch-size 8 \
  --train-iters 20 --log-interval 1 \
  --data-path mock --tokenizer-type Llama3Tokenizer
```

Capture the per-iteration time and the per-GPU peak memory. This is your baseline.

### 2. Add sequence parallel

```bash
# Add to the command above
  --sequence-parallel
```

Peak memory should drop ~20–30% at N=32K. Per-iter time should be within noise.

### 3. Add context parallel

```bash
# Reduce TP to 4, add CP=2 — keeps TP×CP=8
  --tensor-model-parallel-size 4 \
  --context-parallel-size 2 \
  --sequence-parallel
```

Push the sequence length further (`--seq-length 131072`). Without CP this would OOM. With CP=2 it should fit. Measure per-iter time and peak memory. Plot vs the no-CP point at the same N (if the no-CP run is now feasible at all).

### 4. Add selective recomputation

```bash
  --recompute-granularity selective \
  --recompute-method block
```

Should buy another 20–30% memory at the cost of ~10–15% extra forward time. Use only when SP+CP is not enough.

### 5. The activation memory curve

For each (N, CP, recompute) combination you tried, record peak GPU memory. Plot:

```
y = peak memory (GB)
x = sequence length
lines = (CP=1 no recompute, CP=1 recompute, CP=4, CP=4 recompute)
```

You should see the no-CP line going vertical first (OOMs at smaller N), CP=4 staying flat much longer, and recompute pushing both lines down at a fixed N.

---

## Use it in the real stack

- **NeMo Megatron Bridge** wraps all of this — its long-context skill page gives recipes for `seq-length × CP × recompute × precision` combinations per model size.
- **Megatron-LM** is the underlying library with the actual `--context-parallel-size` flag and ring-attention implementation.
- **DeepSpeed-Ulysses** is an alternative sequence-parallel scheme that splits along the **head** dimension instead of the sequence dimension; lower comm volume for some shapes, different trade-offs at very long context.

The cacheon-sglang-miner project we worked on uses FlashInfer for *inference*-side long context. The training-side version (Megatron CP) is structurally similar but tuned for backward + bigger micro-batches.

---

## Measure it

For each configuration in your sweep:

- Per-iter wall-clock (median over the last 10 iterations after 10 warmup).
- Per-GPU peak HBM allocated.
- Achieved per-GPU TFLOPS (`model_FLOPs / wall_clock / GPUs`).
- Communication time as fraction of step time (Megatron logs report this when `--log-throughput` is on).

A healthy long-context run with CP=4 should hold per-GPU TFLOPS within 20% of the CP=1 baseline at the same N. If communication exceeds 30% of step time, your CP chunk size is too small or `NCCL_P2P_LEVEL` / `NCCL_NVLS_ENABLE` is misconfigured.

---

## Ship it

Drop into `lcm-course/`:

1. `attention_bench.sh` — your three commands and the resulting timing/memory log.
2. `activation_memory.csv` and `activation_memory.png` — the curve described above.
3. `notes_attention.md` — one paragraph each on FlashAttention, SP, CP, selective recompute, plus the OOM table from your runs.

You now have, in concrete numbers, the cost of each long-context lever on your hardware.

---

## Related pages

- [Module 01 — Why long context is hard](01-Long-Context-Bottlenecks.md)
- [Module 03 — Position encoding](03-Position-Encoding.md)
- [Module 09 — Distributed training infrastructure](09-Distributed-Training-Infrastructure.md)
- [FlashAttention Course — Lecture 3](../../../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/Lecture%2003%20-%20FlashAttention-1%20Algorithm.md)
- Megatron-LM: <https://github.com/NVIDIA/Megatron-LM>
- Ring Attention paper: <https://arxiv.org/abs/2310.01889>
