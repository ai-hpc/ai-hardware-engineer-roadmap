# Long-Context MoE Foundation Model Training

**Parent:** [HPC Setup](../Guide.md)

**Reviewed:** October 18, 2026. Start with the [CPU training and dispatch lab](CPU-Training-Lab.md), which includes actual optimization, held-out results, gradient checks and two-process communication. The cluster experiments are a separate completion level.

**Format:** 10 modules. Theory → systems mechanics → lab artifact. Each module ships a measurable output (config diff, profiler trace, training run log, benchmark table, or eval CSV).

**Why this course exists.** Most "train a big model" tutorials stop at single-node fine-tuning. Most "long context" tutorials stop at the attention kernel. This course covers what is actually required to train a **long-context, Mixture-of-Experts foundation model** at scale — the joint system of attention math, position encoding, expert routing, distributed parallelism, adaptive data, and honest evaluation. By the end you can read NVIDIA Megatron / NeMo Bridge code, design a context-parallel + expert-parallel mesh that fits your hardware, debug a router collapse, run a long-context evaluation that is not gamed by perplexity, and explain why your model is or is not useful outside the training rig.

---

## What you will be able to do at the end

- Derive the memory + compute scaling of attention vs MLP at long sequence lengths and predict where each becomes the dominant cost.
- Design and evaluate a length-extension curriculum, separating configured maximum length from measured usable context.
- Pick a positional encoding strategy (RoPE base scaling, YaRN, position interpolation) for a target context length with concrete trade-offs.
- Set up a sparse MoE FFN block with top-k routing, capacity-factor tuning, load-balancing loss, and router z-loss; debug router collapse.
- Lay out a training mesh combining data, tensor, pipeline, sequence, context, and expert parallelism without treating every term as an independent device-count multiplier for an 8x or 64x H200 cluster.
- Build an adaptive data pipeline that turns evaluation failure modes into next-iteration training examples.
- Run real long-context evaluations (RULER, LongBench, needle-in-haystack with distractors, multi-hop, codebase reasoning) and report results honestly.
- Argue rigorously about whether your model is good in general or good only on a narrow test distribution.

---

## Prerequisites

- Phase 4 Track B (Jetson, CUDA basics, TensorRT).
- Phase 4 Track C Units 01–02 (graph optimization, kernel engineering).
- [FlashAttention systems / kernel course](../../../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/Guide.md) (this course depends on the IO/roofline + online-softmax mental model).
- HPC Setup [NCCL Deep Dive](../NCCL-Deep-Dive/README.md) and [8× H200 Training/Inference](../8x-H200-Training-Inference/README.md) modules.
- Python 3.12 and matrix/gradient fundamentals for the CPU route. No cluster is required for the ledger, small MoE training, dispatch correctness or evaluation-design work. Multi-GPU training and performance experiments require appropriate supported hardware, software and a compute budget.

---

## Syllabus

| # | Module | Lab artifact |
|---|--------|--------------|
| 01 | [Why long context is hard](01-Long-Context-Bottlenecks.md) | Memory + compute scaling table across `N ∈ {4K, 32K, 128K, 1M}`; identify the crossover where attention dominates MLP |
| 02 | [Long-context attention mechanics](02-Long-Context-Attention.md) | FlashAttention + context-parallel benchmark on one node; activation-memory plot vs sequence length |
| 03 | [Positional encoding for long context](03-Position-Encoding.md) | Per-position retrieval-accuracy plot comparing RoPE base scaling vs YaRN vs position interpolation at 32K → 128K |
| 04 | [MoE fundamentals](04-MoE-Fundamentals.md) | Top-k MoE FFN layer in PyTorch; ablation showing aux-loss-weight effect on expert utilization |
| 05 | [MoE systems and infrastructure](05-MoE-Systems-Infrastructure.md) | All-to-all dispatch micro-benchmark; capacity-factor sweep with dropped-token rate |
| 06 | [Adaptive data pipelines](06-Adaptive-Data-Pipelines.md) | One closed-loop iteration: failure analysis → targeted data → retrain → re-eval |
| 07 | [Long-context evaluation](07-Long-Context-Evaluation.md) | RULER + LongBench harness producing per-length, per-task breakdown |
| 08 | [Combining long-context + MoE](08-Combining-LongContext-and-MoE.md) | Mesh-layout decision table; communication-cost breakdown for one realistic config |
| 09 | [Distributed training infrastructure](09-Distributed-Training-Infrastructure.md) | Working multi-node training run with checkpoint, resume, and fault-tolerance test |
| 10 | [From experiment to general-purpose model](10-General-Purpose-Model.md) | Comparison harness against an open-source baseline on tasks **outside** your training distribution |

---

## How to use this course

- Do the modules in order. Each one assumes the systems primitives from the previous one.
- Read NVIDIA's Megatron / NeMo Bridge MoE + long-context docs alongside the course. The course is a tour guide; the upstream code is the ground truth.
- Keep all training configs, eval CSVs, and profiler traces in a single `lcm-course/` working directory so the capstone (Module 10) can diff against earlier runs.
- Evaluation is a first-class artifact. Every change to the model or data must be paired with an eval delta — not just a loss curve.

---

## Core sources

- NeMo Megatron Bridge — long-context + MoE training recipes: <https://docs.nvidia.com/nemo/megatron-bridge/nightly/>
- Megatron-LM (core distributed training primitives): <https://github.com/NVIDIA/Megatron-LM>
- DeepSpeed MoE: <https://www.deepspeed.ai/tutorials/mixture-of-experts/>
- FlashAttention: <https://github.com/Dao-AILab/flash-attention>
- Effective Long-Context Scaling of Foundation Models (Meta): <https://arxiv.org/abs/2309.16039>
- YaRN: <https://arxiv.org/abs/2309.00071>
- RULER long-context benchmark: <https://github.com/hsiehjackson/RULER>
- LongBench: <https://github.com/THUDM/LongBench>
- ICML 2025 Long Context Foundation Models workshop: <https://longcontextfm.github.io/>

---

## Role mapping

- **MTS Distributed Training / Training Infrastructure Engineer** — direct skill match. Capstone artifact is in the form a hiring manager can read.
- **MTS Kernels / DL Inference Optimization** — depends on the FlashAttention course; this course extends it with the training-side concerns (recomputation, ZeRO, context parallel).
- **ML Research Engineer (long-context, agents, codebase models)** — Modules 03, 06, 07, 10 map directly to the research-engineering loop.
- **AI Infrastructure Architect** — Modules 05, 08, 09 cover the mesh-design and multi-node decisions you will own.

---

## What this course is not

- Not a research course on novel long-context architectures. We use proven, in-production designs (rotary scaling, top-k MoE, ring/context parallel) and focus on how to ship them.
- The CPU route explains the underlying contracts; the cluster route assumes transformer-training basics and a working pinned recipe. A small regression MoE is not a foundation model.
- Not exhaustive on MoE variants. We focus on token-choice top-k routing; we mention expert-choice, hash routing, and soft MoE only to point at where to read next.

**Study method:** use the [shared lesson cycle and completion rubric](../../../../../Learning-Guide.md) with this course's existing labs. Predict first, test one variation, retain a failure case, and revisit the self-check after two days and one week.


## Two completion levels and a shared artifact

The multi-node labs require suitable hardware; analytical or small-model work does not establish 64-GPU scaling. Without a cluster, complete memory/communication calculations, a tiny top-k router experiment, and the evaluation-design modules, labeling scaling results as modeled. With hardware, compare predictions with traces and checkpoint/resume behavior.

Keep total parameters, active parameters per token, and resident expert memory separate. Sparsely activating experts reduces expert arithmetic but does not automatically remove their weights from memory. Retain the [pinned Megatron source record](labs/sources.json) and [executed CPU evidence](labs/validation.json). Long-context support also requires evaluation beyond retrieval of one needle.

**Handoff to VLA hardware:** extract one operator/shape trace from the trained or reference model and explain which state grows with history, action horizon, or expert count. The chip-design course consumes that trace; training a frontier foundation model is not a prerequisite for prototyping its inference subgraph.
