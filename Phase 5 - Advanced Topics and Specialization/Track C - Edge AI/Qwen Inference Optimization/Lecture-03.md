# Lecture 3: Decode Optimization for Qwen3-4B on Jetson Orin Nano

## Overview

You now have a Qwen3-4B-Q4_K_M GGUF and a runtime that loads it. The previous JLLM log showed it running at **0.2 tok/s**. The Orin Nano roofline says you should be able to hit **~14–20 tok/s** for this model. This lecture closes that 70–100× gap by walking the optimizations in order of impact:

1. Fix platform configuration (`nvpmodel`, `jetson_clocks`).
2. Verify CUDA path is actually in use.
3. Fuse QKV and gate/up to halve kernel launches.
4. Apply CUDA Graphs to collapse per-token launch overhead.
5. Quantize the KV cache to INT8 once context grows.
6. Add speculative decoding for the last 1.5×.

Every step is concrete: shape arithmetic, kernel-shape changes, and what to measure.

By the end you should be able to:

* Take a JLLM-style trace and identify the single highest-impact fix.
* Estimate per-token bandwidth and predict tok/s before running.
* Configure a fused QKV path and verify the kernel count drops in `nsys`.
* Apply CUDA Graphs to the decode hot path and quantify the win.

---

## 1. Platform Configuration — The First 50× Win

Before anything else, **lock down the platform**. The Orin Nano 8 GB has three power modes; the difference between mode 0 (15 W) and mode 1 (7 W) is roughly **3× in tok/s** for memory-bandwidth-bound workloads.

```bash
# Step 1: maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Step 2: confirm
sudo jetson_clocks --show
# Expect:
#   GPU MinFreq=624000000 MaxFreq=624000000   ← pinned (Nano max) or 1020 MHz on AGX
#   EMC MinFreq=2133000000 MaxFreq=2133000000 ← max memory clock
#   CPU Cluster 0: MinFreq=N MaxFreq=N         ← pinned

# Step 3: watch during inference
tegrastats --interval 500
# Expect during decode:
#   GR3D_FREQ 80–100% (GPU 3D engine busy)
#   EMC_FREQ  near 100% (you're bandwidth-bound)
```

If `GR3D_FREQ` stays at 0% during inference, the GPU isn't being used. That points to a CUDA build issue, not a configuration issue — go back to step 0.

If `EMC_FREQ` stays well below 100% while `GR3D_FREQ` is high, your kernels are compute-bound on something other than weight reads — unusual for a 4B Q4 model and usually indicates large activation working sets (long context, big intermediate buffers).

**Result you should see immediately after this step alone:** 0.2 tok/s → ~10 tok/s on Qwen3-4B-Q4_K_M.

---

## 2. The Decode Hot Path

A single decoded token on Qwen3-4B with naive (unfused) GEMV dispatch:

```
Per layer (× 36):
  1.  fused_rmsnorm_residual   (attn input norm)
  2.  gemv_q4k  Q              M=4096, K=2560, ~2.6 MB weight read
  3.  gemv_q4k  K              M=1024, K=2560, ~0.7 MB
  4.  gemv_q6k  V              M=1024, K=2560, ~0.9 MB
  5.  rope_kernel              ~0
  6.  kv append                ~0
  7.  flash_attention_decode   reads KV cache slice
  8.  gemv_q4k  O              M=2560, K=4096, ~2.6 MB
  9.  fused_rmsnorm_residual   (ffn norm)
  10. gemv_q4k  gate           M=6912, K=2560, ~4.4 MB
  11. gemv_q4k  up             M=6912, K=2560, ~4.4 MB
  12. swiglu_kernel            ~0
  13. gemv_q6k  down           M=2560, K=6912, ~6.0 MB

Then once:
  - output_norm
  - gemv (LM head, tied = 2560 × 151936 Q6_K) ~310 MB
  - softmax
  - sample
```

Per-layer kernel count: **13**. Times 36 layers + ~3 final = **471 kernel launches per token**. At ~10 µs per launch on Orin, that's ~4.7 ms of pure launch overhead per token — even if kernel work itself were free, you'd cap at ~210 tok/s. In practice on Orin Nano you'd cap much lower because some of those kernels are tiny relative to launch cost.

Per-layer bytes read from DRAM:

```
weights:   2.6 + 0.7 + 0.9 + 2.6 + 4.4 + 4.4 + 6.0 = ~21.6 MB/layer
× 36 layers + LM head 310 MB                       = ~1.09 GB/token

Plus KV cache reads — at 4 k context filled:
   2 × 36 layers × 8 KV heads × 128 dim × 4096 ctx × 2 B = 576 MB
   read once per token through flash-attention      = ~576 MB
```

Total: **~1.66 GB/token of DRAM traffic**. At Orin Nano's measured ~50 GB/s effective bandwidth: **~30 tok/s ceiling for short context, ~16 tok/s for full 4 k context**. That's the number you're racing toward.

---

## 3. Fusion #1 — QKV Concatenation

Three GEMVs read `x` from DRAM three times. Concatenate the matrices:

```
W_QKV = [W_Q | W_K | W_V]    # shape: K × (M_Q + M_K + M_V) = 2560 × (4096+1024+1024)
                              #      = 2560 × 6144
```

One GEMV: `M=6144, K=2560`. Output slices to Q, K, V.

```cuda
// Before: three kernels, three reads of x
gemv_q4k_kernel<<<...>>>(q_out, w_q, x, 4096, 2560);
gemv_q4k_kernel<<<...>>>(k_out, w_k, x, 1024, 2560);
gemv_q6k_kernel<<<...>>>(v_out, w_v, x, 1024, 2560);

// After: one kernel, one read of x
gemv_q4k_q6k_mixed_kernel<<<...>>>(qkv_out, w_qkv, x, 6144, 2560);
// then in subsequent kernels treat qkv_out[0:4096], [4096:5120], [5120:6144]
```

Wait — Qwen has different quant types for Q (Q4_K) and V (Q6_K). The straightforward fusion stores them all as the same type. Options:

1. **Quantize everything to Q4_K and live with the quality drop.** ~0.1 perplexity penalty.
2. **Keep separate matrices but launch them on the same stream and overlap.** Saves launch overhead, doesn't save the input-x re-read.
3. **Write a kernel that handles mixed quant inside one launch.** Most production runtimes do this.

For Qwen3-4B-Q4_K_M the JLLM/llama.cpp default is option 2; vLLM/TRT-LLM (working on AWQ-int4) get option 3 for free because AWQ uses uniform quantization across the whole matrix.

**Expected win:** ~15–25% tok/s improvement on Orin Nano.

---

## 4. Fusion #2 — Gate and Up

Same trick on the FFN:

```
W_gu = [W_gate | W_up]    # shape: 2560 × (6912 + 6912) = 2560 × 13824
```

One GEMV produces both, then SwiGLU is applied as:

```
out = silu(gu_out[0:6912]) * gu_out[6912:13824]
```

This is a clean win because both matrices have the same quant type (both Q4_K in the standard Q4_K_M recipe). No mixed-type concerns.

**Expected win:** ~10–15% additional tok/s.

---

## 5. CUDA Graphs — Collapse the Launches

After fusion the per-token kernel count drops from ~470 to ~250. Still a lot. Each launch is ~5–10 µs on Orin.

CUDA Graphs let you **capture the entire decode-one-token computation once** and re-launch as a single graph node:

```c++
cudaGraph_t graph;
cudaGraphExec_t graph_exec;

// Capture mode: first iteration
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
decode_one_token(stream, /* ... */);   // launches all 250 kernels
cudaStreamEndCapture(stream, &graph);

cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

// Steady state: launch the whole graph as one operation
for (int t = 0; t < n_tokens; t++) {
    update_kv_indices_and_token_id(stream);   // tiny CPU-side prep
    cudaGraphLaunch(graph_exec, stream);
    sample_token(stream);
}
```

Constraints to know:
- **Shapes must be fixed at capture time.** This is fine for decode (batch=1, seq_len=1).
- **Pointers must be stable.** Use a persistent scratch arena, don't malloc inside the captured region.
- **KV-cache indices change per token.** Either use a graph with `cudaGraphExecKernelNodeSetParams` to update indices, or precompute index arrays for the whole generation.

**Expected win:** ~30–50% on Orin Nano specifically — the iGPU's launch overhead is a larger fraction of total kernel time than on discrete GPUs.

---

## 6. FlashAttention-Decode for the Attention Block

The `flash_attention_decode_kernel` you saw in the build log is the **canonical optimization** for the attention step. The idea: instead of materializing the `[seq_len × seq_len]` attention-score matrix, **tile across the KV cache** and accumulate `softmax · V` on-chip.

For decode specifically the operation is:

```
q:    [n_heads, head_dim]                = [32, 128]
K:    [seq_len, n_kv_heads, head_dim]    = [ctx, 8, 128]
V:    [seq_len, n_kv_heads, head_dim]    = [ctx, 8, 128]

output[h] = softmax(q[h] · K[:, h//4, :]ᵀ / √128) · V[:, h//4, :]
```

The optimized kernel:
1. Each thread block handles one Q head.
2. Tiles the K cache in shared memory in chunks of ~64 tokens.
3. Streams through Q · Kᵀ, applies softmax incrementally (online softmax trick from FlashAttention-2).
4. Accumulates against V tiles.
5. Writes one `head_dim`-sized output vector.

The win over a naive implementation is huge — naive can easily be 5× slower for long contexts on Orin because of repeated KV-cache passes.

If your runtime doesn't have a proper FlashAttention-decode kernel, that's where you should focus before touching anything else. The MLC-LLM and llama.cpp implementations are usable references.

---

## 7. INT8 KV Cache — When Context Matters

Once you push past ~4 k context, the KV cache becomes a meaningful fraction of decode bandwidth. Quantize it:

```
FP16 KV: 4096 bytes/token/layer  → 576 MB at 4k ctx
INT8 KV: 2048 bytes/token/layer  → 288 MB at 4k ctx (-50%)
INT4 KV: 1024 bytes/token/layer  → 144 MB at 4k ctx (-75%)
```

The quantization layout that works empirically:

* Per-head scale (FP16) computed every N tokens during prefill (e.g., per 64-token chunk).
* INT8 K and V stored separately (different statistical properties — V has more outliers).
* On-the-fly dequant in shared memory inside flash-attention-decode.

Quality impact for Qwen3-4B: INT8 KV is essentially free (< 0.1 perplexity drop). INT4 KV starts showing degradation past ~16 k context. INT4 KV with per-channel calibration during prefill is competitive with INT8 KV up to ~64 k context.

Most production runtimes (vLLM, SGLang, TRT-LLM) ship INT8 KV as a one-flag option. llama.cpp has it as `--cache-type-k q8_0 --cache-type-v q8_0`.

---

## 8. Speculative Decoding — The Last 1.5×

The decoded sequence is **autoregressive**: every token depends on the previous. Speculative decoding breaks this by:

1. Running a **small draft model** (say, Qwen3-0.5B at Q4) for the next K tokens.
2. Running the **target model** (Qwen3-4B-Q4_K_M) on those K candidates **in parallel** (one forward pass with seq_len = K).
3. Accepting the prefix of candidates that the target would have produced, plus one extra free token.

On Orin Nano specifically, the math:

```
Naive: 36-layer Qwen3-4B forward = 1.66 GB DRAM / token → ~12 tok/s

Spec dec (K=4):
  - Draft model 0.5B at Q4 → ~400 MB DRAM / token, fast
  - Target model forward pass with seq_len=4
    → still ~1.66 GB DRAM (weights dominate, batch dimension is free for bandwidth)
    → so the target step costs ~83 ms wall
  - With 60% acceptance rate, you get 2.4 tokens out per target step
  - Effective rate: 2.4 / 83ms = 29 tok/s
```

Catches:
- The draft model needs to **agree with the target most of the time** for spec dec to win. Qwen3-0.5B as a draft for Qwen3-4B works reasonably (~50-65% acceptance). Random tiny models do not work.
- You pay extra DRAM for two models. On Orin Nano 8 GB this is tight — typically you'd run both models at Q4 and accept the smaller cache budget.

Most edge runtimes don't ship speculative decoding yet. As of 2026 it's standard in vLLM/SGLang, optional in MLC-LLM, missing in llama.cpp and most embedded paths.

---

## 9. Putting It Together — A Budget for Qwen3-4B on Orin Nano

| Step | Action | Cumulative tok/s |
|---|---|---|
| Baseline (from JLLM log) | unmodified, default DVFS | 0.2 |
| + `nvpmodel -m 0 && jetson_clocks` | lock max power and clocks | 8–10 |
| + Confirm CUDA path active | not on CPU fallback | 10–12 |
| + Fused QKV | one GEMV instead of three | 12–14 |
| + Fused gate+up | one GEMV instead of two | 14–16 |
| + CUDA Graphs over the per-token decode | collapse 250 launches into one | 18–22 |
| + FlashAttention-decode (proper impl) | if not already there | 20–24 |
| + INT8 KV (>4k context only) | preserves perf as ctx grows | 20–24 (longer ctx) |
| + Speculative decoding (Qwen3-0.5B draft) | trade extra DRAM for accept rate | 28–35 |

A well-engineered runtime should be in the 25–35 tok/s band on Orin Nano for Qwen3-4B-Q4_K_M at short context. That's the target. If you're shipping 8–12, you're missing fusion or graphs. If you're shipping 1–3, you're still on a config issue or CPU fallback.

---

## 10. Diagnosing Your Specific Trace

From the JLLM log:

```
Power: 0W mode, GPU @ 0 MHz
[engine] Prefill: 18 tokens in 110400 ms (0.2 tok/s)
[engine] Decode:  16 tokens in 100064 ms (0.2 tok/s)
```

Walkthrough:
1. **0.2 tok/s, GPU @ 0 MHz** → DVFS-parked. Step 1 of this lecture fixes that.
2. After `jetson_clocks`, expect ~10 tok/s. If you get that, the runtime itself is functional and you continue down the optimization list.
3. **Three GEMVs visible** (`#0 #1 #2` for Q, K, V) → no QKV fusion. Step 3.
4. **Prefill same speed as decode** → JLLM is running per-token GEMV during prefill instead of batched GEMM. Big prefill win available by switching to GEMM (TTFT improves an order of magnitude, doesn't affect decode rate).

The JLLM runtime is structurally fine — it's missing the standard optimizations. Apply them in order from §9.

---

## Hands-On Exercises

1. **The before-and-after table.** On the same Orin Nano, run Qwen3-4B-Q4_K_M through llama.cpp with:
   (a) default config,
   (b) after `jetson_clocks`,
   (c) with `--mlock` (pin weights),
   (d) with CUDA Graphs enabled (recent llama.cpp builds expose this).
   Record tok/s and `tegrastats` snapshots for each. Produce the four-row table.

2. **Roofline plot.** For Qwen3-4B-Q4_K_M, plot tok/s vs context length from 256 to 4096. Overlay the bandwidth-bound theoretical curve. Identify where you deviate and why (KV cost rising, attention compute taking over, etc.).

3. **FlashAttention check.** Determine whether your runtime is using a fused attention kernel by inspecting the kernel names in `nsys` traces. If it isn't, switch to a build/branch that has it and re-measure §1's roofline plot.

4. **INT8 KV at long context.** Generate a 16 k-token prompt (chunked code completion is a good source). Decode 256 new tokens with FP16 KV, then with INT8 KV. Compare tok/s and the actual generated text. Quantify the bandwidth saving and the perceived quality difference.

5. **Speculative decoding with Qwen3-0.5B.** Download Qwen3-0.5B (or 1.7B), quantize to Q4_K_M, and set up speculative decoding in vLLM (or MLC-LLM, where supported). Measure acceptance rate on chat prompts and on code prompts. Report the actual end-to-end tok/s win on Orin Nano.

6. **The "is it even using CUDA?" sanity check.** Take a runtime where you suspect CPU fallback. Run a 32-token decode. Read `tegrastats`. If `GR3D_FREQ` is 0% and CPU load is 100%, your runtime is on CPU. Fix the build (rebuild with `LLAMA_CUDA=1` or equivalent) and re-test.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| `nvpmodel` + `jetson_clocks` is the single biggest knob | 50× delta between worst and best config — and they're free |
| 471 kernel launches per token is the unfused baseline | Launch overhead alone can dominate decode |
| Fused QKV and fused gate+up halve the launch count | First two optimizations to ship after configuration |
| CUDA Graphs are uniquely valuable on Orin | Higher launch-overhead fraction than discrete GPUs |
| FlashAttention-decode is non-negotiable past ~1k context | Naive attention can dominate KV-cache bandwidth |
| INT8 KV is essentially free quality-wise on Qwen3-4B | Use it whenever context > 4 k |
| Speculative decoding needs a draft that agrees with the target | Qwen3-0.5B for Qwen3-4B is a reasonable pairing |

---

## Resources

* **[NVIDIA Jetson Linux Developer Guide — Power Modes](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/):** Canonical `nvpmodel` and `jetson_clocks` documentation.
* **[CUDA Graphs — Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs):** Capture mechanics, parameter updates.
* **[FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):** The online-softmax + tiling primitive used in the decode kernel.
* **[FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html):** Decode-specific variant that parallelizes across KV blocks.
* **[Speculative Decoding paper (Leviathan et al.)](https://arxiv.org/abs/2211.17192):** The original spec-dec analysis.
* **[Medusa: Multiple decoding heads](https://arxiv.org/abs/2401.10774):** Inline-spec-dec variant; relevant if you want to avoid two-model overhead.
* **[MLC-LLM Qwen example](https://llm.mlc.ai/docs/):** Reference for fused kernels on Jetson.
* **[llama.cpp Qwen3 support](https://github.com/ggerganov/llama.cpp):** Default reference runtime.
* **[Phase 5 — Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md):** Prereq with the roofline math.
