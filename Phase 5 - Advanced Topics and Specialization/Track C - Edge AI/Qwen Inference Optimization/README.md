# Qwen Inference Optimization — 6-Lecture Series

A hands-on, hardware-first series on getting real throughput out of two specific Qwen models that bracket the deployment spectrum:

* **Qwen3-4B-Instruct-2507 (Q4_K_M)** — edge target, runs on Jetson Orin Nano 8 GB.
* **Qwen2.5-72B-Instruct (FP16)** — datacenter target, is typically deployed across multiple GPUs at FP16; exact fit depends on memory and context.

Both are Qwen-family causal transformers; their exact configurations and deployment scale differ. The [Qwen3-4B-Instruct-2507 config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json) and [Qwen2.5-72B config](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json) are the source of shape calculations. The series teaches both endpoints, then unifies them through cross-model strategies (speculative decoding, edge↔cloud routing).

**Scope:** inference only. Training and fine-tuning are out of scope.

| Lecture | Title | Focus |
|---|---|---|
| [01](Lecture-01.md) | Qwen Architecture Deep Dive | Config, shapes, GQA, RoPE, SwiGLU, tokenizer |
| [02](Lecture-02.md) | Quantizing Qwen3-4B to Q4 | AWQ, GPTQ, K-quants, calibration, GGUF layout |
| [03](Lecture-03.md) | Decode Optimization on Jetson | GEMV chain, KV cache, fusion, CUDA Graphs, INT8 KV |
| [04](Lecture-04.md) | Qwen2.5-72B Multi-GPU FP16 | TP/PP, NCCL hot path, paged attention, YaRN |
| [05](Lecture-05.md) | Cross-Model & Production Serving | Speculative decoding, vLLM/TRT-LLM, observability, hybrid |
| [06](Lecture-06.md) | Batched GEMM vs Normal GEMM | cuBLAS API forms, layout, tensor cores, bit-exact reproducibility |

**Prerequisites:**

* Phase 5 — Edge AI — [Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md) (GEMV vs GEMM, roofline)
* Phase 4 Track B — [Jetson Real-Time Inference](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md)
* Phase 4 Track C — [Quantization](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md)


## Integration lab: Do the tensor shapes agree with the exact Qwen checkpoint?

Extract the pinned configuration before tracing kernels; recompute projection and KV sizes. Separate the Jetson small-model experiment from the multi-GPU 72B experiment instead of pooling their metrics.

**Acceptance evidence:** The report must reconcile predicted and stored parameter counts, isolate prefill/decode, and state runtime/quantization support. Pass the bottleneck shape distribution to kernel engineering.

**Continue:** take this artifact to [02 - Kernel Engineering](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/Guide.md). Explain which assumption from this course the next engineer must preserve.
