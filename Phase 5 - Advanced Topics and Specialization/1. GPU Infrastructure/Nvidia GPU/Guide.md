# HPC with Nvidia GPU

<div class="course-identity auto-course" style="--course-accent: #db2777; --course-accent-rgb: 219, 39, 119;" markdown="1">
<div class="course-identity__icon">HWNG</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Specialization</p>
<p class="course-identity__title">Specialized course identity for HPC with Nvidia GPU.</p>
<p class="course-identity__meta">Artifact: specialization case study · Measure: performance, reliability, role fit</p>
</div>
</div>


**Parent:** [High Performance Computing](../Guide.md)

**Timeline:** 12–24 months (fundamentals and deep dives); 24–48 months for advanced phase.

---

## Basic concepts: what "HPC with Nvidia GPU" means


**HPC** = high-performance computing: solving problems that need massive compute and memory by using many machines (or many GPUs) working together. In AI, "HPC" usually means **large-scale training** and **high-throughput inference** on GPU clusters, not single workstations.

**Why Nvidia GPUs?** Nvidia GPUs are the dominant hardware for training and deploying large models. They offer the best-supported software stack: **CUDA** (programming model and runtime), **cuDNN** and **CUTLASS** (high-performance kernels — cuDNN for conv/RNN/attention; CUTLASS for customizable GEMM, matrix multiply, and epilogue fusion used by frameworks and custom kernels), **TensorRT** (inference optimization and deployment), and **NCCL** (multi-GPU collectives). Add the fastest inter-GPU links (NVLink, NVSwitch) and the architectures (Hopper, Blackwell) that ML frameworks target first, and you see why AI infrastructure and kernel-level optimization work almost always involves Nvidia GPUs in data centers.

**What this track covers:**

* **Single GPU → many GPUs** — From one GPU (e.g. Jetson, which you saw in Phase 4 Track B) to **multi-GPU nodes** and **multi-node clusters**. You need to understand how jobs are placed, how data and gradients move, and how to avoid communication becoming the bottleneck.
* **Two main workloads:**
    * **Training** — One big model, one big dataset; you split work across GPUs (data parallelism, model parallelism, pipeline parallelism). Performance is about throughput (samples/sec) and scaling to hundreds or thousands of GPUs.
    * **Inference** — Many requests, one (or many) deployed model; you care about latency and throughput under load. At scale this means batching, KV-cache, and often multi-GPU or multi-node serving (e.g. TensorRT-LLM, vLLM).
* **The stack you must understand:**
    * **Hardware:** GPUs (A100, H100/H200, L40S, etc.), NVLink/NVSwitch inside a node, InfiniBand or Ethernet across nodes.
    * **Software:** CUDA, drivers, containers (NGC), orchestrators (Slurm, Kubernetes), and collective libraries (NCCL) for multi-GPU communication.
    * **Storage and I/O:** Getting data to GPUs fast (dataloaders, GPUDirect Storage, high-throughput disks) so the GPU is not waiting.

### Key terms (used in this track)

*Order: basic ops → attention → distributed.*

| Term | Meaning |
|------|--------|
| **Matrix multiply** | Compute A·B for matrices A, B (often plus bias or activation). The core operation in linear layers and most heavy compute. "GEMM" is the standard name in libraries. |
| **GEMM** | **G**eneral **E**lement-wise **M**atrix **M**ultiply: C = α(A·B) + βC. The BLAS/cuBLAS/CUTLASS interface for matrix multiply. GPU GEMM kernels (tiling, tensor cores) dominate training and inference time. |
| **Epilogue fusion** | In a GEMM kernel, the **epilogue** is what you do with the result (add bias, ReLU/GELU, write to memory). **Fusion** = doing that in the *same* kernel as the multiply. Saves memory bandwidth and launch overhead; CUTLASS and similar libraries support it. |
| **Attention** | In transformers: each token has **Q, K, V** (from linear layers); *attention* = softmax(Q·K^T/√d)·V. Lets the model focus on relevant tokens. The matmuls are heavy for long sequences; optimized attention kernels and KV-cache are critical. |
| **FlashAttention** | A family of **attention kernels** that reduce memory traffic by tiling and keeping Q,K,V in SRAM, and avoid materializing the full Q·K^T matrix. Faster and more memory-efficient than naive attention; standard in LLM training and inference (e.g. FlashAttention-2, -3). |
| **KV-cache** | In transformer attention: keys and values for previous tokens are cached so you don't recompute them. **KV-cache** = that cache. Long context → huge cache → memory and bandwidth become the bottleneck; paging/sharding and efficient kernels matter. |
| **Data / model / pipeline parallelism** | **Data:** same model on every GPU, different data; sync gradients. **Model:** split the model across GPUs. **Pipeline:** different layers on different GPUs, pass activations in a pipeline. |
| **Collectives** | Multi-GPU operations: **AllReduce** (everyone gets the same sum), **AllGather** (everyone gets all pieces), **ReduceScatter**. Used to sync gradients (data parallel) or exchange activations (model/pipeline parallel). |
| **NCCL** | **N**vidia **C**ollective **C**ommunications **L**ibrary. Implements collectives (AllReduce, AllGather, etc.) for multi-GPU training and inference. Often the bottleneck at scale. |
| **NVLink / NVSwitch** | **NVLink:** high-bandwidth GPU↔GPU (and GPU↔CPU) link inside a node. **NVSwitch:** switch connecting many GPUs in one node. Much faster than PCIe for GPU↔GPU traffic. |

---

## What this track covers (after reorganization)

This track now focuses on **HPC infrastructure and multi-GPU operations**. The **DL Inference Optimization** content (graph optimization, kernel engineering, compiler stack, quantization, inference runtimes, tinygrad deep dive) has moved to **[Phase 4 Track C — ML Compiler & Graph Optimization](../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md)** as Part 2, since those skills are foundational for all hardware tracks, not just HPC.

| Part | Description | Guide |
|------|--------------|-------|
| **HPC Setup** | Fundamentals, virtualization, interconnects, advanced CUDA/distributed training/performance — plus hardware-specific deep dives (8x H200, L40S, NCCL, CUDA Advanced, GPUDirect Storage) | [HPC Setup →](./HPC%20Setup/Guide.md) |
| ~~DL Inference Optimization~~ | **Moved to [Phase 4 Track C Part 2](../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/Guide.md)** | [Track C →](../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md) |

---

## How to use this track

1. **Complete [Phase 4 Track C](../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md)** first — covers compiler fundamentals and DL inference optimization (graph ops, kernels, compiler stack, quantization, runtimes).
2. **Then [HPC Setup](./HPC%20Setup/Guide.md)** — Covers Nvidia GPU HPC fundamentals, virtualization (vGPU, KVM), interconnects and storage (InfiniBand, GDS, Slurm, Kubernetes), and Phase 2 advanced topics (advanced CUDA, distributed training, performance modeling). Use the deep dives (8x H200, L40S, NCCL, CUDA Advanced, GDS) for your target hardware and stack.

**Prerequisite:** Phase 4 Track B (Jetson, TensorRT, CUDA) and Phase 4 Track C (compiler + inference optimization).
