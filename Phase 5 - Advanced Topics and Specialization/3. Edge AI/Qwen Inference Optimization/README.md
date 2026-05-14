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
* Phase 4 Track C — [Quantization](../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md)
