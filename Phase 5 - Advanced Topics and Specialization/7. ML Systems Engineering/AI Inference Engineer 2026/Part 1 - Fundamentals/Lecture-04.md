# Part 1 · Lecture 04 — The Precision Stack: FP16 → FP8 → FP4 → INT4

## Overview

If the roofline (Lecture 03) is the *static* ceiling of a GPU, **precision is the lever that moves where on the roofline a kernel sits.** Cutting weight precision in half cuts the bytes read from HBM in half, doubles arithmetic intensity, and — if the kernel was bandwidth-bound — roughly doubles throughput.

The catch: every precision drop is a potential parity drop. Quantization is the engineering discipline of taking a precision floor down without taking accuracy with it. The engineer who quantizes without a parity gate is shipping incidents.

This lecture covers:

1. The 2026 precision landscape — what each format actually stores and where it ships natively.
2. The three quantization axes — weights, activations, KV cache — and why they're three different decisions.
3. The major weight-only methods — AWQ, GPTQ, QuaRot, SpinQuant — and how to pick.
4. The known anomalies — including the Llama-3-70B W8A8 sensitivity (arXiv:2408.15301) that Part 2 will revisit.
5. The parity validation methodology — what to measure, what budget to set, what to never trade away.

By the end you should be able to look at a model + workload + hardware target and write down a precision recipe (e.g., "AWQ-INT4 weights, FP16 activations, FP8 KV") with a defended parity budget for each.

---

## 1. The 2026 precision landscape

The precision floor has dropped twice in three years. The current stack:

| Format | Bits | Range / mantissa | Where it lives |
|--------|------|------------------|----------------|
| FP32 | 32 | full IEEE 754 | training reference, never deployed |
| TF32 | 19 | 8-bit exp, 10-bit mantissa | Ampere+ training, rarely inference |
| BF16 | 16 | 8-bit exp, 7-bit mantissa | default training precision, common inference |
| FP16 | 16 | 5-bit exp, 10-bit mantissa | inference workhorse pre-FP8 |
| **FP8 E4M3** | 8 | 4-bit exp, 3-bit mantissa | Hopper + Blackwell native, primary FP8 weight/activation |
| **FP8 E5M2** | 8 | 5-bit exp, 2-bit mantissa | Hopper + Blackwell, often used for gradients / KV |
| **FP6 (E3M2, E2M3)** | 6 | various | Blackwell support, less common in practice |
| **FP4 (E2M1, MX-FP4)** | 4 | 2-bit exp, 1-bit mantissa | Blackwell native, microscaled |
| INT8 | 8 | signed integer | universal, calibrated; legacy quantization |
| INT4 | 4 | signed integer | weight-only mainstream (AWQ / GPTQ / GGUF) |

The key 2025–2026 shifts:

* **FP8 is the new default activation precision** on Hopper-and-newer hardware where the kernels are mature. Most flagship inference deployments (TensorRT-LLM, vLLM 0.22+) ship FP8 first.
* **FP4 native arithmetic on Blackwell** (Transformer Engine 2) doubles FP8 throughput. Microscaling format (MX-FP4) attaches per-block scale factors so the dynamic range is preserved.
* **INT4 weight-only remains the dominant quantization for cost-sensitive serving** — AWQ, GPTQ, and GGUF-IQ-quants all live here. INT4 weights at FP16 (or FP8) activations is the practical recipe for 70B-class on 1–4 GPUs.
* **KV cache quantization** is increasingly common at FP8 (and sometimes INT4 with calibration) because the KV cache dominates HBM at long context.

### 1.1 The hardware support gate

A precision format is only useful if the GPU has native tensor cores for it. The matrix:

| Format | Ampere (A100, RTX 3000) | Hopper (H100/H200) | Ada (L40S, RTX 4000) | Blackwell (B200, RTX 5000) |
|--------|--------------------------|--------------------|----------------------|----------------------------|
| FP16/BF16 | ✓ | ✓ | ✓ | ✓ |
| FP8 | ✗ (emulated) | ✓ TE | ✓ TE | ✓ TE2 |
| FP6 | ✗ | ✗ | ✗ | ✓ |
| FP4 | ✗ | ✗ | ✗ | ✓ TE2 |
| INT8 (DP4A) | ✓ | ✓ | ✓ | ✓ |
| INT4 weight-only | weights stored low-precision, computed at higher | same | same | same |

**INT4 weight-only is a software pattern**, not a hardware operation: the weights are stored at 4 bits per element, dequantized to FP16 (or FP8) in shared memory, and computed at that higher precision. There is no native INT4 tensor core. This is why AWQ-INT4 + FP16 activations works on any modern GPU back to A100.

**FP8 and FP4 *are* hardware operations** — Hopper / Blackwell tensor cores execute matmul natively at those precisions. The performance win is real silicon, not a software trick.

---

## 2. The three quantization axes — weights, activations, KV cache

These are three separate engineering decisions with three separate parity costs.

### 2.1 Weights

Quantizing weights cuts:

* HBM capacity by the ratio (FP16 → INT4: 4× smaller).
* HBM bandwidth on decode read (the dominant cost) by the same ratio.

Sensitivity is **per-layer and per-channel** — some attention heads or MLP rows tolerate quantization poorly. Method choice (AWQ / GPTQ / QuaRot) is mostly about handling the sensitive rows gracefully.

### 2.2 Activations

Quantizing activations cuts:

* Intermediate HBM traffic between layers (small win — activations are not the dominant traffic).
* Tensor core throughput if the format has hardware support (FP8 doubles vs FP16 on Hopper).

Sensitivity is **per-tensor or per-token** — outliers in the activations (a few large values) cause clipping that propagates. Methods (SmoothQuant, OmniQuant, QuaRot) work by *redistributing* activation outliers into the weights or by applying invariant rotations.

**Weight-only INT4 + FP16 activations is the safe default.** FP8 weights + FP8 activations is the higher-performance recipe on Hopper / Blackwell once parity is validated.

### 2.3 KV cache

Quantizing KV cuts:

* HBM capacity of the KV cache (allows longer context or larger batches).
* HBM bandwidth on every decode step (proportional to context length).

Sensitivity is **per-head and per-position** — KV cache quantization at INT4 typically requires per-block calibration. FP8 KV is usually safe with per-tensor or per-head scaling.

Note the bandwidth cost of KV grows linearly with context. **At 128K context the KV cache read is often a larger HBM cost than the weight read.** This is why FP8 KV is increasingly standard for long-context serving even when weights stay at INT4.

### 2.4 The three-axis recipe table

| Workload | Weights | Activations | KV cache | Notes |
|----------|---------|-------------|----------|-------|
| Chat, 70B, 16K context, H200 | INT4 (AWQ) | FP16 | FP16 | safe default |
| Chat, 70B, 128K context, H200 | INT4 (AWQ) | FP16 | FP8 | KV pressure forces precision |
| Chat, 70B, 128K context, B200 | FP4 (MX-FP4) | FP4 | FP8 | full Blackwell stack |
| Batch / offline, 70B, H100 | FP8 | FP8 | FP8 | throughput optimization |
| Edge, 4B, Jetson Orin Nano | INT4 (AWQ or Q4_K_M) | FP16 | FP16 | memory tight, KV small |
| Agent (BFCL-gated), 7B–70B | INT4 (AWQ) | FP16 | FP16 → FP8 once parity OK | tool-call accuracy is the parity bar |

The recipe is a starting point. Every cell must be parity-verified per §5 before shipping.

---

## 3. The major weight-only quantization methods

### 3.1 GPTQ — Post-training INT4 via OBS

[GPTQ](https://arxiv.org/abs/2210.17323) (2023) quantizes one layer at a time using a second-order error metric (Optimal Brain Surgeon). It is the canonical "first practical INT4" method.

Strengths:

* Mature; ships in `auto-gptq`, `vLLM`, `TensorRT-LLM`, `llama.cpp`.
* Predictable parity on dense models that don't have severe outlier behavior.

Weaknesses:

* No explicit handling of activation outliers; quality degrades on models where activations have large outlier channels.
* Can be sensitive to calibration set choice.

### 3.2 AWQ — Activation-aware Weight Quantization

[AWQ](https://arxiv.org/abs/2306.00978) (2023) observes that a small set of weight channels (~1%) corresponds to channels with large activation magnitudes, and protects them by per-channel scaling. Quantizes the rest at INT4 group-128.

Strengths:

* Generally better parity than GPTQ on agent/tool-use workloads (the BFCL discussion in [Phase 5 → Edge AI → BFCL Lecture](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) shows this).
* Fast inference: the trick is a software trick on top of standard INT4 matmul kernels.

Weaknesses:

* Requires calibration data that matches the deployment distribution.
* Doesn't help with activation quantization (it's a weight-only method).

**The current default for weight-only INT4 on dense LLMs.** Part 2 Lecture 03 walks the AWQ pipeline on Llama 3.3 70B and Qwen 2.5 72B step by step.

### 3.3 QuaRot — Rotation-invariant quantization

[QuaRot](https://arxiv.org/abs/2404.00456) (2024) applies an invariant rotation (Hadamard matrices) to the model so that activation outliers spread across all channels evenly. After rotation, even W8A8 (or W4A8) quantization becomes tractable on previously-difficult models.

Strengths:

* Handles models like Llama-3-70B that GPTQ and AWQ alone struggle with under W8A8 (see the anomaly in §4).
* Enables low-precision activations, not just weights — important on Hopper FP8 / Blackwell FP4.

Weaknesses:

* More invasive — requires modifying the model graph to insert the rotation matrices.
* Newer, less ecosystem coverage as of mid-2026.

### 3.4 SpinQuant — Learnable rotations

[SpinQuant](https://arxiv.org/abs/2405.16406) (2024) extends QuaRot by *learning* the rotation matrix on a calibration set rather than using a fixed Hadamard. Often achieves better parity than QuaRot at the same precision.

Strengths:

* Best-in-class parity for W4A4 on models with severe outliers.

Weaknesses:

* Requires a learning step; not pure post-training.
* Ecosystem support thinner than AWQ/GPTQ.

### 3.5 SmoothQuant — Activation-to-weight migration

[SmoothQuant](https://arxiv.org/abs/2211.10438) (2022) "smooths" activation outliers by migrating them to weights, enabling W8A8. Predecessor to QuaRot in concept.

Strengths:

* Well-supported in TensorRT-LLM and DeepSpeed.
* Good for INT8/INT8.

Weaknesses:

* Less effective than QuaRot/SpinQuant at lower precisions.

### 3.6 GGUF / K-quants / IQ-quants

The llama.cpp ecosystem ships its own family of quantizations stored in the GGUF format. The widely-used ones:

* **Q4_K_M** — 4.5 bits/weight effective; the de facto edge default.
* **Q5_K_M** — 5.5 bits/weight; better parity, larger files.
* **Q3_K_S / Q3_K_M** — 3-bit, aggressive; usually too lossy for production.
* **IQ4_XS / IQ3_M** — newer "Improved Quants" with importance-matrix calibration; better parity at the same bit depth.

GGUF/K-quants are weight-only and don't fundamentally differ from AWQ/GPTQ at the math level — they differ in tooling, format, and integration with the llama.cpp runtime. Use them when shipping into llama.cpp deployments.

### 3.7 Method-picking guide

| Constraint | Method |
|------------|--------|
| Server, dense 7B–70B, INT4, validated parity | **AWQ** (start here) |
| Server, dense 70B+, INT4, agent workload | **AWQ** with BFCL calibration |
| Server, low-precision activations (W4A8 / W4A4) | **QuaRot** or **SpinQuant** |
| Server, FP8 weights and activations | **FP8 native** (NVIDIA TE) |
| Edge (llama.cpp / Jetson) | **Q4_K_M** GGUF or **AWQ-INT4** via TRT-LLM |
| Blackwell, full FP4 stack | **TE2 native FP4** + per-block scaling |
| Aggressive: W3 / W2 | **SpinQuant + rotation** if you must; usually re-think the model size |

---

## 4. The known anomalies

Two specific anomalies worth knowing:

### 4.1 Llama-3-70B W8A8 sensitivity

The paper *"The Uniqueness of LLaMA3-70B Series with Per-Channel Quantization"* ([arXiv:2408.15301](https://arxiv.org/abs/2408.15301)) showed that Llama-3-70B (and by extension 3.3-70B, which is architecturally identical) has unusually severe activation outliers in specific MLP channels. Standard SmoothQuant W8A8 produces a 2–4 pp drop on MMLU; W4A8 is worse.

The fix: use QuaRot or SpinQuant (rotation-based) — they handle the outliers correctly. Or stay at W4A16 (AWQ-INT4 weights + FP16 activations), which dodges the activation-quantization problem entirely.

Part 2 Lecture 03 walks through this anomaly with concrete numbers on Llama 3.3 70B.

### 4.2 KV cache rolloff at INT4

KV cache quantization to INT4 with per-tensor scaling typically loses several percentage points on long-context tasks (RULER, needle-in-haystack). Per-head or per-block scaling recovers most of it. **Always validate KV quantization at the actual context length the product will use** — short-context parity does not imply long-context parity.

### 4.3 FP8 KV on heads with rare-token specialization

A small number of attention heads in some models specialize in rare-token positions (BOS, code-tokens). FP8 per-tensor scaling clips these. Per-head scaling fixes it. This is a known issue in mid-2025–2026 vLLM and SGLang FP8 KV implementations.

---

## 5. The parity validation methodology

The non-negotiable discipline. Every precision drop needs a gate.

### 5.1 The parity contract

Before any precision drop:

1. **Pin the reference.** The FP16/BF16 deployment, exact tokenizer, exact prompts, exact seed list. Hash the weights. This is the contract.
2. **Pin the eval set.** A fixed, public eval that matches your workload class:
   * Chat: MMLU subset (200 questions), GSM8K subset, HumanEval if code.
   * Agent: [BFCL](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) at the categories you serve.
   * Long context: RULER (4K → 128K), needle-in-haystack.
   * Embedding: MTEB subset.
3. **Run reference twice with different seeds** — the variance between runs is your noise floor.
4. **Set a parity budget** ≤ 3× the noise floor, intersected with the product-acceptable drop.

### 5.2 Run the candidate

1. Apply the precision drop in isolation (one knob at a time).
2. Re-run the same eval set.
3. Compute Δ per category. Categories that exceed budget block the recipe.
4. Inspect failures: which questions / requests now fail that used to pass? Sample 20 and read them. Often you can predict which precision axis is responsible (weights vs activations vs KV) by the failure pattern.

### 5.3 The four common parity failure modes

| Symptom | Suspect axis | Fix |
|---------|--------------|-----|
| Tool-call accuracy drops, MMLU stable | Weight quantization touched rare-vocabulary projections | Per-channel weight scaling, calibration data with tool-use examples |
| Long-context recall drops, short-context fine | KV cache precision | Per-head KV scaling, or stay at FP8 KV |
| Code generation regresses, chat fine | Activation quantization in MLP | Switch to weight-only, or use QuaRot |
| Random spikes on specific prompts | Outlier channels | QuaRot / SpinQuant |

### 5.4 The cardinal rule

**Never trade a parity drop in the workload-defining metric for a throughput gain in a different metric.**

If your product is an agent and BFCL drops 4 pp for a 2× throughput gain, the recipe ships incidents. Pick a different recipe. The discipline of the [VLA action-parity harness](../../../4.%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/Lecture-02.md) applies to LLMs too — every optimization is hypothetical until parity is verified.

---

## Lab — quantize Qwen3-4B and validate parity

Goal: extend the benchmark harness from Lectures 01–03 with a quantization parity gate.

1. **Reference run** — Qwen3-4B Instruct at BF16 (the published precision). Run a small eval set (MMLU-200, BFCL-50 if you have it). Record numbers + variance across two seeds.
2. **Candidate A — AWQ-INT4** weights, FP16 activations, FP16 KV. Quantize using `auto-awq` with 256 calibration prompts that match your deployment distribution. Re-run eval. Compute Δ per category. Decide: within budget?
3. **Candidate B — FP8** weights and activations (TensorRT-LLM or vLLM 0.22+ FP8 path, if you have Hopper). Re-run eval.
4. **Candidate C — AWQ-INT4 weights, FP8 KV cache.** Re-run eval, especially at long context if applicable.
5. **Produce a parity report** — one CSV per candidate, one summary markdown.

Pass criterion: at least one candidate passes the parity budget on your defined eval set, and you have written down which one would ship under what product constraints.

---

## Self-check

1. You are deploying a 7B model on a single L40S (no FP8 tensor cores fast enough to matter). What is the safest precision recipe to start with? What does it not buy you that FP8 would on H100?
2. A teammate shows you a benchmark where W8A8 quantized Llama 3 70B is 1.6× faster than FP16 but loses 3 pp on MMLU. You insist on QuaRot W4A8 instead. Defend the choice in two sentences.
3. You quantize a model to FP4 on Blackwell and tool-call accuracy (BFCL) drops 5 pp while MMLU is stable. Which of the three quantization axes is the likely culprit, and what is the first experiment to confirm?
4. You measure parity twice on the FP16 reference and get MMLU = 78.2 and 78.6. A candidate scores 76.8. Is this within or outside a reasonable parity budget? Show the math.
5. On Hopper, you are choosing between (a) AWQ-INT4 weights + FP16 activations and (b) FP8 weights + FP8 activations for a chat workload at concurrency 32. Which has the higher decode throughput ceiling, and why?

---

## References

* GPTQ — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
* AWQ — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
* SmoothQuant — [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
* QuaRot — [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
* SpinQuant — [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
* OmniQuant — [arXiv:2308.13137](https://arxiv.org/abs/2308.13137)
* "The Uniqueness of LLaMA3-70B Series with Per-Channel Quantization" — [arXiv:2408.15301](https://arxiv.org/abs/2408.15301)
* NVIDIA Transformer Engine documentation — [docs.nvidia.com/deeplearning/transformer-engine/](https://docs.nvidia.com/deeplearning/transformer-engine/)
* MX (microscaling) FP4 / FP6 specification — Open Compute Project, 2023+ — [opencompute.org](https://www.opencompute.org/)
* llama.cpp GGUF / IQ-quants reference — [github.com/ggml-org/llama.cpp/wiki](https://github.com/ggml-org/llama.cpp/wiki)

Cross-references:

* [Phase 5 → Edge AI → Qwen Inference Optimization → Lecture 02 — Quantizing Qwen3-4B to Q4](../../../3.%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-02.md)
* [Phase 5 → Edge AI → Agent Tool-Dispatch Evaluation with BFCL](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) — parity gating
* [Phase 4 → Track C → Quantization](../../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md) — compiler-side foundation

---

## Current as of 2026-06

Methods pinned as of mid-2026: AWQ, GPTQ, QuaRot, SpinQuant for INT4/W4A8. FP8 (E4M3, E5M2) on Hopper/Blackwell. FP4 (MX-FP4) on Blackwell. Refresh when a new quant method ships with better-than-AWQ parity on agent workloads, or when FP6 / FP3 lands in hardware.

---

## Next

* Next: [Lecture 05 — The runtime landscape](Lecture-05.md)
* Previous: [Lecture 03 — Roofline, bandwidth, and the memory hierarchy](Lecture-03.md)
* Up: [Part 1 — Fundamentals](README.md)
