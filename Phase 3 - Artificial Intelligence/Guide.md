# Phase 3: Artificial Intelligence — The Workloads Your Hardware Must Run

<div class="course-identity ai-workloads" markdown="1">
<div class="course-identity__icon">AI</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 3 · AI Workloads</p>
<p class="course-identity__title">Learn the models, frameworks, and application patterns your hardware must actually run.</p>
<p class="course-identity__meta">Artifact: workload profile · Measure: compute, memory, precision, latency</p>
</div>
</div>


> *Before you design hardware, you must deeply understand the software it accelerates.*

**Layer mapping:** **L1** (Application & Framework) — this entire phase teaches you what AI chips compute.

**Role targets:** ML Inference Engineer · Edge AI Engineer · Agentic AI Engineer · ML Engineer · AI Compiler Engineer

**Prerequisites:** Phase 1 (Digital Foundations), Phase 2 (Embedded Systems).

**What comes after:** Phase 4 Track A (FPGA), Track B (Jetson), Track C (ML Compiler).

---

## Why This Phase Exists

Every decision in the 8-layer stack is driven by workload requirements. If you skip this phase, you'll design hardware without knowing what it needs to run. Phase 3 gives you the workload intuition that informs every hardware decision downstream.

---

## Structure: Core + Two Tracks

**Modules 1–2** are mandatory for everyone. Then you choose **Track A**, **Track B**, or both.

```
Module 1: Neural Networks          ← mandatory (what accelerators compute)
Module 2: Deep Learning Frameworks ← mandatory (micrograd, PyTorch, tinygrad)
        ↓                    ↓
   Track A                Track B
   Hardware &             Agentic AI &
   Edge AI                ML Engineering
        ↓                    ↓
   Phase 4               Phase 4 or
   (FPGA/Jetson/          Phase 5
    Compiler)             (HPC/GenAI)
```

---

## Core Modules (Mandatory)

| # | Module | What you learn | Why it matters for hardware |
|---|--------|---------------|---------------------------|
| **1** | [Neural Networks](1.%20Neural%20Networks/Guide.md) | MLPs, CNNs, training, backpropagation, loss functions | What accelerators compute — tensors, matmul, activations |
| **2** | [Deep Learning Frameworks](2.%20Deep%20Learning%20Frameworks/Guide.md) | [micrograd](2.%20Deep%20Learning%20Frameworks/micrograd/Guide.md) → [PyTorch](2.%20Deep%20Learning%20Frameworks/PyTorch/Guide.md) → [tinygrad](2.%20Deep%20Learning%20Frameworks/tinygrad/Guide.md): autograd, ops, compiler pipeline | How software generates workloads — the interface between models and hardware |

---

## Track A — Hardware & Edge AI

> *For engineers heading to Phase 4 (FPGA, Jetson, ML Compiler) and Phase 5 (Autonomous Vehicles, AI Chip Design).*

This track teaches the **perception and deployment workloads** that drive edge inference hardware.

| # | Module | What you learn | Leads to |
|---|--------|---------------|----------|
| **3** | [Computer Vision](Track%20A%20-%20Hardware%20and%20Edge%20AI/3.%20Computer%20Vision/Guide.md) | Image processing, detection, segmentation, 3D vision, OpenCV | Phase 4A (FPGA vision), Phase 5E (AV perception) |
| **4** | [Sensor Fusion](Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Guide.md) | Camera/LiDAR/IMU, Kalman filtering, BEVFusion, MOT | Phase 4B (Jetson + ROS2), Phase 5E (AV) |
| **5** | [Voice AI](Track%20A%20-%20Hardware%20and%20Edge%20AI/5.%20Voice%20AI/Guide.md) | STT (Whisper), TTS (VITS/Piper), VAD, keyword spotting, noise suppression | Phase 4A (FPGA audio DSP), Phase 4B (Jetson voice pipeline) |
| **6** | [Edge AI & Model Optimization](Track%20A%20-%20Hardware%20and%20Edge%20AI/6.%20Edge%20AI%20and%20Model%20Optimization/Guide.md) | Quantization, pruning, knowledge distillation, deployment pipeline | Phase 4 (bridge to all hardware tracks) |

**Build:** OpenCV detection, sensor calibration, INT8 quantization, tinygrad on-device inference, Whisper on Jetson, edge voice pipeline (VAD→STT→TTS).

---

## Track B — Agentic AI & ML Engineering

> *For engineers heading to Phase 5 (HPC, GPU Infrastructure) or building AI applications that generate the inference demand your hardware serves.*

This track teaches the **AI application, infrastructure, and safety workloads** that sit on top of modern inference systems — the demand side of the chip market.

| # | Module | What you learn | Leads to |
|---|--------|---------------|----------|
| **3** | [Agentic AI & GenAI](Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) | LLM agents, RAG pipelines, tool use, multi-step reasoning, prompt-injection basics, coding agents, and GenAI products | Phase 5A/B (GPU Infrastructure, HPC) |
| **4** | [ML Engineering & MLOps](Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Guide.md) | Training pipelines, experiment tracking, model serving, observability, CI/CD for models | Phase 5A/B (HPC, distributed training) |
| **5** | [LLM Application Development](Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/5.%20LLM%20Application%20Development/Guide.md) | Prompt engineering, [Qwen3.5-4B Unsloth fine-tuning](Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/5.%20LLM%20Application%20Development/Qwen3.5-4B%20Unsloth%20Fine-Tuning/Guide.md), RAG architecture, evaluation, guardrails, moderation, PII controls, production deployment | L1d/L1e roles (highest job volume) |

**Build:** RAG pipeline with vector search, agent with tool calling, coding agent in terminal or CI, fine-tune `Qwen/Qwen3.5-4B-Base` with Unsloth, deploy model behind Triton/vLLM, secure prompt gateway with moderation and redaction.

---

## Relevant Structure: Safe LLM Systems in Track B

For Track B, safety is not a side note. It is part of the application architecture.

- **Module 3B** teaches the threat model: prompt injection, tool misuse, document attacks, and unsafe retrieval context.
- **Module 4B** teaches the operating model: policy gateways, logs, tracing, rollout controls, and production monitoring.
- **Module 5B** teaches the enforcement model: content moderation, jailbreak detection, denied topics, PII redaction, output validation, and fallback behavior.

Use a defense-in-depth pipeline:

```text
User input
  -> Unicode normalization / sanitization
  -> blocklists or denied-topic checks
  -> moderation and jailbreak / prompt-injection classifiers
  -> model with hardened system prompt and tool constraints
  -> output moderation, validation, and PII redaction
  -> logging, review, and threshold tuning
```

The exact policy depends on the product. Some systems block harmful or illegal requests. Others also block application-disallowed topics such as political persuasion, regulated advice, or competitor-sensitive content. The important point is to make policy enforcement explicit and measurable.

---

## Why Two Tracks?

| | Track A (Hardware & Edge AI) | Track B (Agentic AI & ML Eng) |
|---|---|---|
| **Goal** | Understand workloads that run on your chip | Understand workloads that create inference demand |
| **Focus** | Perception, sensors, optimization, deployment | LLMs, agents, training pipelines, serving |
| **Hardware connection** | Direct — you deploy on FPGA/Jetson/NPU | Indirect — you generate the traffic the chip serves |
| **Job market** | L1a, L1b, L1c roles (~4,500/month) | L1d, L1e roles (~15,000/month) |
| **Remote %** | 10–15% (hardware access needed) | 20–25% (cloud/API-based) |
| **Phase 4 path** | Track A → B → C (all hardware) | Track C (compiler) or Phase 5 directly |

**Do both?** If you have time, Track A → Track B gives you full L1 coverage. Most hardware-focused engineers do Track A first, then add Track B topics as needed.

---

## How This Phase Connects to the Stack

| What you learn | How it informs hardware design |
|---------------|-------------------------------|
| Matrix multiply in neural networks | L5: systolic array dimensions, dataflow strategy |
| Conv2D, attention, pooling ops | L2: what the compiler must fuse and tile |
| Quantization (INT8, FP8) | L6: precision support in PE design |
| LLM inference (KV-cache, batching) | L5: memory hierarchy, HBM bandwidth requirements |
| Model computational graphs | L2: graph IR representation, fusion opportunities |
| Training at scale (distributed) | L3: NCCL, multi-GPU runtime |
| Moderation, guardrails, and prompt shielding | L3/L4: serving topology, latency budget, sidecar services, and policy gateways |

---

## What You Should Produce

By the end of this phase, you should have at least:

- one model-implementation or workload-intuition artifact from the core modules
- one deployment- or optimization-oriented artifact from Track A or Track B
- one short write-up connecting workload behavior to hardware constraints such as memory, latency, throughput, or precision

Examples include a quantization comparison, a profiling note, a tinygrad trace, a RAG latency analysis, or a sensor-pipeline benchmark.

---

## Exit Criteria

You are ready for Phase 4 when you can:

- explain what your target workloads actually compute
- identify whether a workload is dominated by compute, memory, batching, context length, or deployment constraints
- connect model structure to compiler, runtime, or hardware implications
- show at least one artifact that proves you did more than read about the workload

---

## Additional Resources

- [CMU AI Courses Reference](CMU-AI-Courses.md)
- [Maxime Labonne's LLM Course](https://github.com/mlabonne/llm-course) — free LLM roadmap with Colab notebooks covering LLM fundamentals, fine-tuning, quantization, RAG, evaluation, and deployment. Use it as a hands-on companion for Track B.
- [LLM Visualization (bbycroft.net)](https://bbycroft.net/llm) — interactive 3D walk through a GPT-style model: every tensor, every matmul, attention, layer norm, MLP, softmax, output projection. The fastest way to convert "I read about transformers" into "I can see what each tensor is for." Pair with [Neural Networks → Transformer Fundamentals](1.%20Neural%20Networks/Transformer%20Fundamentals/Lecture-01.md).

---

## Next

→ [**Phase 4 Track A — Xilinx FPGA**](../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md) · [**Phase 4 Track B — Jetson**](../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md) · [**Phase 4 Track C — ML Compiler**](../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/Guide.md)
