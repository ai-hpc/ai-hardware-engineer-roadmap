# Qwen Inference Optimization — 5-Lecture Series

A hands-on, hardware-first series on getting real throughput out of two specific Qwen models that bracket the deployment spectrum:

* **Qwen3-4B-Instruct (Q4_K_M)** — edge target, runs on Jetson Orin Nano 8 GB.
* **Qwen2.5-72B-Instruct (FP16)** — datacenter target, needs multi-GPU.

Same architecture family, completely different hardware engineering. The series teaches both endpoints, then unifies them through cross-model strategies (speculative decoding, edge↔cloud routing).

**Scope:** inference only. Training and fine-tuning are out of scope.

| Lecture | Title | Focus |
|---|---|---|
| 01 | Qwen Architecture Deep Dive | Config, shapes, GQA, RoPE, SwiGLU, tokenizer |
| 02 | Quantizing Qwen3-4B to Q4 | AWQ, GPTQ, K-quants, calibration, GGUF layout |
| 03 | Decode Optimization on Jetson | GEMV chain, KV cache, fusion, CUDA Graphs, INT8 KV |
| 04 | Qwen2.5-72B Multi-GPU FP16 | TP/PP, NCCL hot path, paged attention, YaRN |
| 05 | Cross-Model & Production Serving | Speculative decoding, vLLM/TRT-LLM, observability, hybrid |
| 06 | Batched GEMM vs Normal GEMM | cuBLAS API forms, layout, tensor cores, bit-exact reproducibility |

**Prerequisites:**

* Phase 5 — Edge AI — [Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md) (GEMV vs GEMM, roofline)
* Phase 4 Track B — [Jetson Real-Time Inference](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md)
* Phase 4 Track C — [Quantization](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md)

## Practice and assessment

**Entry diagnostic (prerequisite check):** Trace prefill, decode, and cache memory for the chosen Qwen config. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Profile one supported Qwen deployment and compare a single kernel/runtime change.
2. **Guided variation:** Repeat on a different context/batch shape and verify output quality.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Quality, TTFT/TPOT, memory, kernel traces, and exact checkpoint/runtime versions. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
