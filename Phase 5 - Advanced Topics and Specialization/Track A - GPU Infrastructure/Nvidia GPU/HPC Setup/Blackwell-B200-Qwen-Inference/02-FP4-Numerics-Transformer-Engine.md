# Chapter 2: FP4 Numerics and Transformer Engine 2 for Qwen Inference

## Overview

Blackwell's 5th-gen tensor cores natively execute **FP4**, **FP6**, and **FP8** with shared block scales — the **MX (microscaling)** family of formats from the OCP standard. That sounds esoteric. In practice it means: a Qwen2.5-72B model that was 145 GB at FP16 is **36 GB at MX-FP4**, with quality degradation small enough to ship in production.

The second-generation **Transformer Engine** is the software layer that makes this work without you hand-tuning per-tensor scales. It dynamically chooses scale factors per block during inference, picks the right format per tensor (FP4 for FFN-up, FP6 or FP8 for FFN-down and V, etc.), and emits the right kernel variants.

This chapter covers the formats, the engine, and the per-tensor decisions you make when deploying Qwen on B200.

By the end you should be able to:

* Distinguish MX-FP4 / MX-FP6 / MX-FP8 by bit layout and block scale.
* Predict Qwen2.5-72B memory and bandwidth for each format.
* Pick a mixed-precision recipe similar to the Q4_K_M asymmetry (V and FFN-down at higher precision).
* Read a Transformer Engine 2 calibration log and identify which tensors fell back to a higher precision.

---

## 1. The OCP Microscaling (MX) Format Family

A microscaled format stores a **block of values** with a single shared scale factor. The scale is FP8/UE8M0 (an unsigned 8-bit exponent), and the block size is 32 elements. Within the block, each element is a small int or float using the format's element type.

```
MX block layout (32 elements):

  [ E ]  ← 1 byte shared scale (UE8M0, an 8-bit power of two)
  [ x0 | x1 | x2 | ... | x31 ]   ← 32 element values

The reconstructed value at slot i is:
    value_i = scale × element_i
```

The element type determines the format:

| Format | Element type | Element bits | Block size | Effective bits/value | Range |
|---|---|---|---|---|---|
| MX-FP8 (E5M2) | float 5e2m | 8 | 32 | ~8.25 | ±5.7×10⁴ |
| MX-FP8 (E4M3) | float 4e3m | 8 | 32 | ~8.25 | ±448 |
| MX-FP6 (E3M2) | float 3e2m | 6 | 32 | ~6.25 | ±28 |
| MX-FP6 (E2M3) | float 2e3m | 6 | 32 | ~6.25 | ±7.5 |
| MX-FP4 (E2M1) | float 2e1m | 4 | 32 | ~4.25 | ±6 |
| MX-INT8 | signed int | 8 | 32 | ~8.25 | ±127 |

The "effective bits/value" is the element bits plus the per-block scale overhead amortized across 32 elements: `bits + 8/32 = bits + 0.25`.

### 1.1 Why microscaling at all?

Plain INT4 or FP4 has a fundamental problem: weights and activations in trained LLMs have **wide dynamic ranges** but **small block-local ranges**. Globally, a tensor might span ±50; locally, a block of 32 consecutive weights might all be in ±0.05 or all in ±5. With a single global scale, you waste bits.

Microscaling gives each block of 32 its own scale, so each block gets nearly full FP4 precision relative to its local range. The cost: 0.25 bits/element of scale overhead. The gain: dramatically better quality at the same average bit-rate vs uniform-quant baselines.

This is the same insight as ggml's K-quants (Q4_K, Q6_K) — block-wise scales beat global scales — but standardized at the hardware level and natively executed by tensor cores. No on-the-fly dequant in registers; the tensor core consumes the MX-format bits directly.

---

## 2. What Each Format Costs for Qwen2.5-72B

Per-tensor breakdown for a Qwen2.5-72B-Instruct (FP16 baseline: ~145 GB):

| Tensor | FP16 GB | MX-FP8 GB | MX-FP6 GB | MX-FP4 GB |
|---|---|---|---|---|
| `attn_q` (× 80) | 10.7 | 5.5 | 4.2 | 2.8 |
| `attn_k` (× 80) | 1.3 | 0.7 | 0.5 | 0.35 |
| `attn_v` (× 80) | 1.3 | 0.7 | 0.5 | 0.35 |
| `attn_o` (× 80) | 10.7 | 5.5 | 4.2 | 2.8 |
| `ffn_gate` (× 80) | 38.8 | 20.0 | 15.2 | 10.3 |
| `ffn_up` (× 80) | 38.8 | 20.0 | 15.2 | 10.3 |
| `ffn_down` (× 80) | 38.8 | 20.0 | 15.2 | 10.3 |
| Embeddings (2 ×) | 4.98 | 2.57 | 1.95 | 1.32 |
| Norms, biases | ~0.05 | ~0.05 | ~0.05 | ~0.05 |
| **Total** | **~145 GB** | **~75 GB** | **~57 GB** | **~38.5 GB** |
| Roofline tok/s @ 8 TB/s | ~55 | ~106 | ~140 | **~210** |

The bandwidth × format multiplier directly gives you the decode tok/s ceiling. **MX-FP4 hits ~210 tok/s on a single B200** for Qwen2.5-72B at short context — competitive with what an 8×H100 cluster delivers at FP8.

### 2.1 Mixed precision — the asymmetric recipe

As with K-quants (Lecture 2 of the Edge AI Qwen series), not every tensor deserves the same precision. The empirical rule transfers cleanly to MX formats:

| Tensor group | Recommended format | Reasoning |
|---|---|---|
| `attn_q`, `attn_k` | MX-FP4 | Followed by softmax — small Q/K perturbations get smoothed |
| `attn_v` | MX-FP6 or MX-FP8 | Direct content path; quality-sensitive |
| `attn_o` | MX-FP4 | Acceptable degradation; large parameter count |
| `ffn_gate`, `ffn_up` | MX-FP4 | Largest tensors; biggest bandwidth win |
| `ffn_down` | MX-FP6 or MX-FP8 | Most quality-sensitive; folds into residual |
| Embeddings | MX-FP6 | Vocabulary-wide; affects every token |
| Norms, biases | FP32 | Tiny; no reason not to keep |
| KV cache | MX-FP8 (or FP16 for safety) | Read on every decoded token; INT8 also viable |

This "FP4 everywhere except V, FFN-down, and embeddings" recipe is the Blackwell counterpart of `Q4_K_M`. Memory cost lands around **42–45 GB** instead of a pure-FP4 38.5 GB, with notably better quality. Expected MMLU drop vs FP16 baseline: <0.5 pts; vs pure MX-FP4: ~0.4 pts recovery.

---

## 3. The Transformer Engine 2 Pipeline

Transformer Engine (TE) is the NVIDIA library that owns format selection, scale-factor management, and kernel dispatch for FP8/FP4-mixed inference. Version 2 (Blackwell-aware) handles MX formats and per-block scaling automatically.

### 3.1 What TE2 does at load time

```
GGUF/safetensors (FP16) ─►  TE2 quantizer
                              │
                              ├─ Determine per-tensor format from recipe
                              ├─ Run calibration (a few prompts through the model)
                              ├─ Compute initial per-block scales
                              ├─ Pack into MX layout
                              └─ Write Blackwell-resident weights
```

The calibration pass collects per-block max-abs values across a few hundred prompts, which it uses to set initial scale factors. For pure weight-only quantization (the default for inference) the calibration is **fast** — a minute or two.

### 3.2 What TE2 does at runtime

```
Per layer per token:
  1. Read MX-FP4/FP6/FP8 weights from HBM (5th-gen tensor cores handle dequant internally)
  2. Compute activations in FP16 or BF16 (the high-precision "accumulator" path)
  3. Optionally quantize activations on-the-fly using a learned scale (for KV cache writes)
  4. Detect overflow/underflow and trigger fallback if a block saturates
```

The fallback case is real: a block whose values blow up at runtime gets transparently promoted to FP8 or FP16 for that step. The TE2 log shows promotion events; if you see lots of them on a specific tensor, your recipe is wrong (V or FFN-down at FP4 is the usual culprit).

### 3.3 A typical TE2 calibration log

```
[TE2] Calibrating Qwen2.5-72B with recipe='qwen-mx-mixed'
[TE2] Layer 0: q=MX-FP4, k=MX-FP4, v=MX-FP6, o=MX-FP4
[TE2] Layer 0: gate=MX-FP4, up=MX-FP4, down=MX-FP6
[TE2] Layer 12: WARNING — v block 7/128 saturated, promoting to MX-FP8
[TE2] Layer 47: WARNING — down block 23/231 saturated, promoting to MX-FP8
[TE2] Calibration complete: 78 of 800 weight blocks promoted (0.0098%)
[TE2] Final memory: 42.7 GB (vs 36.5 GB pure FP4, vs 145 GB FP16)
[TE2] Bandwidth/token estimate: 42.7 GB → tok/s ceiling 187
```

The 0.01% promotion rate is healthy. If you see promotion rates above 1%, the recipe is too aggressive for this model.

---

## 4. KV Cache in MX Formats

The KV cache is read once per token, *every* token. Quantizing it pays off proportionally to how often you read it.

| KV format | Bytes per token per layer (Qwen2.5-72B GQA) | KV @ 32k ctx |
|---|---|---|
| FP16 | 4096 | 10.0 GB |
| MX-FP8 | ~2080 | 5.1 GB |
| MX-FP6 | ~1568 | 3.8 GB |
| MX-FP4 | ~1056 | 2.6 GB |
| INT4 with per-channel calibration | ~1024 | 2.5 GB |

Quality-wise: MX-FP8 KV is essentially free (< 0.05 perplexity drop on Qwen2.5-72B). MX-FP4 KV is usable past 8k context but shows degradation in retrieval tasks past ~32k. The safe production default is **MX-FP8 KV** — cuts bandwidth in half vs FP16 with no measurable quality cost.

Implementation detail: KV is quantized **after** RoPE, not before. The rotated K is what gets stored, so the per-block scale must accommodate the rotation's magnitude. TE2 handles this automatically; hand-rolled KV-quant runtimes often get this wrong and produce subtle long-generation degradation.

---

## 5. Comparison vs Ggml K-Quants and AWQ

How do MX formats stack up against the formats you already know from Edge?

| Format | Bits/weight (effective) | Hardware-native? | Per-block scale | Production support |
|---|---|---|---|---|
| Q4_K_M (ggml) | ~4.5 | No — dequant in registers | Yes (256-element superblock) | llama.cpp |
| AWQ-INT4 g128 | ~4.25 | Marlin kernel | Per-128 channel | vLLM, TRT-LLM |
| GPTQ-INT4 g128 | ~4.25 | Marlin kernel | Per-128 channel | vLLM, exllamav2 |
| **MX-FP4** | **~4.25** | **Yes — 5th-gen tensor core native** | **Per-32 element** | **TRT-LLM 0.20+, TE2** |
| **MX-FP8** | **~8.25** | **Yes — tensor core native** | **Per-32 element** | **TRT-LLM 0.20+, TE2** |

The decisive differentiator for MX is **hardware-native execution**. K-quants and AWQ require a custom CUDA kernel that dequantizes blocks into registers before feeding the matmul. MX formats are consumed **directly** by the 5th-gen tensor core's MMA instruction — there's no software dequant step.

This matters for two reasons:

1. **Bandwidth efficiency** — the tensor core reads MX-FP4 bytes from HBM, and that's the full DRAM cost. No second pass.
2. **Compute throughput** — peak MX-FP4 throughput on B200 is ~2.25 PFLOPS dense. AWQ-INT4 via Marlin maxes out at ~half that because the kernel has to do dequant work outside the tensor core path.

For Qwen workloads on Blackwell, the recipe is: **MX-FP4 with TE2 mixed-precision auto-recipe → MX-FP8 KV cache**. Use Q4_K_M only on platforms (Jetson, AMD) where MX isn't supported.

---

## 6. The Quality Story — What Actually Drops?

From early-2026 published benchmarks on Qwen2.5-72B-Instruct:

| Config | MMLU | IFEval | GSM8K | HumanEval | MT-Bench |
|---|---|---|---|---|---|
| BF16 baseline | 84.2 | 87.1 | 91.5 | 75.6 | 9.04 |
| MX-FP8 (all) | 84.1 | 86.9 | 91.4 | 75.5 | 9.02 |
| MX-FP6 mixed | 83.9 | 86.5 | 91.0 | 75.1 | 8.98 |
| **MX-FP4 mixed (V/down=FP8)** | **83.8** | **86.2** | **90.8** | **74.7** | **8.94** |
| MX-FP4 pure (no mixing) | 83.0 | 84.1 | 89.5 | 73.1 | 8.71 |

The mixed-precision MX-FP4 recipe is the sweet spot — under 0.5 pts off MMLU, under 1 pt off any benchmark, less than 0.1 off MT-Bench. **The single most important parameter is keeping V and FFN-down at FP6 or FP8.** Pure FP4 drops noticeably; the mixed recipe is production-ready.

For comparison, Q4_K_M on the same model drops ~0.4 MMLU and ~1.2 IFEval — slightly worse than MX-FP4 mixed at the same effective bit-rate, with no hardware-native execution path.

---

## 7. Deployment — From `transformers` to TRT-LLM

A typical conversion pipeline for Qwen2.5-72B → B200 production:

```bash
# Step 1: download FP16/BF16 weights
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir ./qwen72b

# Step 2: convert via TRT-LLM's quantizer (uses TE2 internally)
python -m tensorrt_llm.quantization.quantize \
    --model_dir ./qwen72b \
    --output_dir ./qwen72b-mx-fp4 \
    --dtype bf16 \
    --qformat mx_fp4_mixed \
    --calib_dataset openassistant-en-zh \
    --calib_size 256

# Step 3: build the TensorRT engine for B200
trtllm-build --checkpoint_dir ./qwen72b-mx-fp4 \
             --output_dir ./qwen72b-mx-fp4-engine \
             --gemm_plugin mx_fp4 \
             --gpt_attention_plugin auto \
             --max_batch_size 64 \
             --max_input_len 32768 \
             --max_seq_len 65536 \
             --kv_cache_type mx_fp8 \
             --use_paged_context_fmha
```

The build produces a TensorRT engine targeting `sm_100` (Blackwell). It ships ~42 GB and loads to one B200 die in ~3 seconds.

---

## 8. When MX-FP4 Doesn't Work

Real failure modes worth knowing:

* **Long-context retrieval** — Needle-in-haystack at >50k context starts showing FP4 degradation. Promote V and FFN-down to FP8 for retrieval-heavy workloads.
* **Code generation** — HumanEval pass@1 drops ~2 pts at pure FP4, ~0.5 pts at mixed. Acceptable for assistants; might not be for high-stakes code review pipelines.
* **Multilingual** — minority-language quality drops more than English. Calibrate on a multilingual mix if you serve global traffic.
* **Long structured output** (JSON, tool calls) — schema adherence degrades subtly. Some production deployments keep `attn_o` at FP6 for this reason.
* **MoE active expert routing** — for MoE Qwen variants, the router weights are tiny and quality-sensitive. Always keep router at FP16/BF16.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| MX-FP4 is hardware-native on B200 — no software dequant | Bandwidth-efficient and compute-efficient at the same time |
| Per-32-element block scales beat global scales | Same insight as Q4_K_M, standardized at hardware level |
| Mixed-precision recipe (V and FFN-down at FP6/FP8) is essential | Pure FP4 loses ~1 pt across benchmarks; mixed loses <0.5 |
| TE2 manages scales and dispatches kernels automatically | Inference engineers rarely write per-tensor scale code |
| MX-FP8 KV cache is essentially free quality-wise | Use it by default for context >4k |
| AWQ/GPTQ remain the right choice off-Blackwell | MX requires 5th-gen tensor cores |
| Quality regressions concentrate in long-context retrieval and code | Validate on your actual workload, not just MMLU |

---

## Resources

* **[OCP Microscaling Formats v1.0 Spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):** The authoritative MX format reference.
* **[NVIDIA Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/):** The TE2 API and recipe system.
* **[TensorRT-LLM Quantization Guide](https://nvidia.github.io/TensorRT-LLM/architecture/quantization.html):** End-to-end MX-FP4 workflow.
* **["Microscaling Data Formats for Deep Learning" (2023)](https://arxiv.org/abs/2310.10537):** The research backing the OCP standard.
* **[Chapter 1 — Blackwell Architecture](01-Blackwell-Architecture.md):** The hardware foundation.
* **[Chapter 3 — Single-B200 Qwen Inference](03-Single-B200-Qwen-Inference.md):** Applying these numerics in deployment.
* **[Qwen Inference Optimization — Lecture 2 (Q4_K_M etc.)](../../../../Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-02.md):** The edge-quantization counterpart.
