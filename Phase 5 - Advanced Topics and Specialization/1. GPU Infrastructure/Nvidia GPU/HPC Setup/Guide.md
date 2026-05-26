# HPC Setup

**Part of:** [High Performance Computing — Nvidia GPU](../Guide.md) (Phase 5)

This guide combines **HPC fundamentals, virtualization, interconnects, and advanced topics** with **hardware-specific deep dives** for real-world GPU cluster setup and optimization. Study the fundamentals first, then use the deep dives for your target hardware (8x H200, L40S, NCCL, CUDA, GDS).

---

## Hardware & Stack Deep Dives

Detailed guides for specific GPU cluster configurations and subsystems:

| Setup | Use Case | Guide |
|-------|----------|-------|
| **8x H200 SXM5** | Large model training & inference (1.1 TB HBM3e, NVLink 4.0) | [8x-H200-Training-Inference/](./8x-H200-Training-Inference/README.md) |
| **L40S x12** | Cost-efficient inference deployment (576 GB GDDR6, PCIe) | [L40S-x12-Inference/](./L40S-x12-Inference/README.md) |
| **NCCL Deep Dive** | GPU-to-GPU communication: algorithms, tuning, debugging, 1T-scale | [NCCL-Deep-Dive/](./NCCL-Deep-Dive/README.md) |
| **CUDA Advanced Optimization** | CUDA Graphs, Cooperative Groups, Persistent Kernels, Fusion, Warp Specialization | [CUDA-Advanced-Optimization/](./CUDA-Advanced-Optimization/README.md) |
| **GPUDirect Storage (GDS)** | Direct NVMe→GPU DMA, NVMe-oF, WD OpenFlex + RapidFlex, libcufile API | [GPUDirect-Storage/](./GPUDirect-Storage/README.md) |
| **Long-Context MoE Foundation Training** | End-to-end course: long-context attention, MoE routing, mesh design, adaptive data, honest eval | [Long-Context-MoE-Foundation-Training/](./Long-Context-MoE-Foundation-Training/README.md) |

### 8x H200 — Topics
- [01 Hardware Architecture](./8x-H200-Training-Inference/01-Hardware-Architecture.md) — GH100 die, HBM3e, NVLink 4.0, NVSwitch topology
- [02 Training Setup](./8x-H200-Training-Inference/02-Training-Setup.md) — FSDP, DeepSpeed ZeRO-3, Megatron-LM, 3D parallelism
- [03 Inference Setup](./8x-H200-Training-Inference/03-Inference-Setup.md) — vLLM, TensorRT-LLM, FP8, speculative decoding
- [04 Memory Management](./8x-H200-Training-Inference/04-Memory-Management.md) — KV cache, PagedAttention, GQA, profiling
- [05 Performance Optimization](./8x-H200-Training-Inference/05-Performance-Optimization.md) — Roofline, CUDA Graphs, kernel fusion, NCCL tuning
- [06 Benchmarks & Validation](./8x-H200-Training-Inference/06-Benchmarks-and-Validation.md) — MFU, MBU, latency/throughput targets

### L40S x12 — Topics
- [01 Hardware Architecture](./L40S-x12-Inference/01-Hardware-Architecture.md) — AD102 die, GDDR6, PCIe topology, no-NVLink constraints
- [02 Inference Optimization](./L40S-x12-Inference/02-Inference-Optimization.md) — GPTQ, AWQ, FP8 quantization, vLLM, continuous batching
- [03 Multi-GPU Strategy](./L40S-x12-Inference/03-Multi-GPU-Strategy.md) — PCIe parallelism, pipeline vs tensor parallel, InfiniBand
- [04 Deployment Guide](./L40S-x12-Inference/04-Deployment-Guide.md) — systemd, Docker, Kubernetes, NGINX load balancing, monitoring
- [05 Benchmarks](./L40S-x12-Inference/05-Benchmarks.md) — throughput tables, L40S vs H200 comparison, load testing

### NCCL Deep Dive — Topics
- [01 Fundamentals](./NCCL-Deep-Dive/01-Fundamentals.md) — AllReduce, Broadcast, AllGather, ReduceScatter, AllToAll explained with diagrams
- [02 Algorithms & Bandwidth](./NCCL-Deep-Dive/02-Algorithms-and-Bandwidth.md) — Ring vs Tree vs Double Binary Tree, how 900 GB/s is achieved, bandwidth math
- [03 Framework Integration](./NCCL-Deep-Dive/03-Framework-Integration.md) — how PyTorch DDP, FSDP, DeepSpeed ZeRO, Megatron call NCCL internally
- [04 Configuration & Tuning](./NCCL-Deep-Dive/04-Configuration-and-Tuning.md) — every important env var, per-topology recipes (H200 / L40S / multi-node)
- [05 Multi-Node Clusters](./NCCL-Deep-Dive/05-Multi-Node-Clusters.md) — hierarchical AllReduce, InfiniBand, GPUDirect RDMA, SHARP in-network compute
- [06 Debugging](./NCCL-Deep-Dive/06-Debugging.md) — hangs, errors, XID codes, fault tolerance, recovery patterns
- [07 Trillion-Parameter Scale](./NCCL-Deep-Dive/07-Trillion-Parameter-Scale.md) — 3D parallelism NCCL patterns, MoE AllToAll, communication budgets at scale

### CUDA Advanced Optimization — Topics
- [01 CUDA Graphs](./CUDA-Advanced-Optimization/01-CUDA-Graphs.md) — capture/replay pipelines, PyTorch patterns, bucketing for dynamic shapes, profiling
- [02 Cooperative Groups](./CUDA-Advanced-Optimization/02-Cooperative-Groups.md) — thread block, warp, tiled partition, coalesced, grid-wide sync with examples
- [03 Persistent Kernels](./CUDA-Advanced-Optimization/03-Persistent-Kernels.md) — always-resident GPU workers, GPU-side work queues, zero-overhead dispatch
- [04 Kernel Fusion](./CUDA-Advanced-Optimization/04-Kernel-Fusion.md) — HBM round-trip elimination, Triton, torch.compile, FlashAttention as fusion example
- [05 Warp Specialization](./CUDA-Advanced-Optimization/05-Warp-Specialization.md) — producer/consumer warpgroups, TMA, WGMMA, software pipelining, CUTLASS 3.x

### GPUDirect Storage (GDS) — Topics
- [01 Architecture & Data Path](./GPUDirect-Storage/01-Architecture-and-Data-Path.md) — CPU vs GDS data paths, PCIe topology, NUMA pinning, 3 transport modes
- [02 Hardware Setup](./GPUDirect-Storage/02-Hardware-Setup.md) — WD OpenFlex reference config (A100 + CX-7 + SN3700), PCIe layout, version matrix
- [03 Software Stack](./GPUDirect-Storage/03-Software-Stack.md) — OFED 5.8, GDS 2.17.3, libcufile install, gdscheck verification, cufile.json config
- [04 libcufile API](./GPUDirect-Storage/04-libcufile-API.md) — cuFileRead/Write, buffer registration, batch I/O, PyTorch DataLoader integration
- [05 Performance Tuning](./GPUDirect-Storage/05-Performance-Tuning.md) — 512-byte alignment, optimal transfer size, queue depth, buffer pool, benchmarks
- [06 Disaggregated Storage](./GPUDirect-Storage/06-Disaggregated-Storage.md) — NVMe-oF over RoCEv2, WD OpenFlex + RapidFlex, 75 GB/s scale-out, lossless config

---

## 1. Nvidia GPU HPC Fundamentals

* **GPU Architecture for HPC:**
    * **CUDA and Tensor Cores:** Master CUDA programming for HPC workloads. Understand Tensor Core utilization for mixed-precision compute (FP16, BF16, TF32) in scientific and AI applications.
    * **NVLink and NVSwitch:** Learn about high-bandwidth GPU interconnect technologies for multi-GPU systems. Understand NVLink topology, NVSwitch for scalable GPU clusters, and bandwidth optimization.
    * **GPU Memory Hierarchy:** Deep dive into GPU memory—global memory, shared memory, L1/L2 cache, and unified memory. Optimize memory access patterns for HPC workloads.

* **Multi-GPU Programming:**
    * **NCCL (Nvidia Collective Communications Library):** Master NCCL for efficient multi-GPU and multi-node collective operations—all-reduce, broadcast, all-gather. Understand NCCL topology detection and tuning for optimal performance.
    * **CUDA Multi-Process Service (MPS):** Learn MPS for sharing GPUs across multiple processes, improving utilization in HPC and inference workloads.
    * **MPI + CUDA:** Combine MPI for distributed computing with CUDA for GPU acceleration. Implement hybrid MPI-CUDA applications for large-scale HPC clusters.

**Resources:** Nvidia NCCL Documentation · Nvidia vGPU Documentation · "Professional CUDA C Programming" by Cheng et al.

**Projects:** Implement a Multi-GPU Training Pipeline with NCCL all-reduce; benchmark NVLink vs. PCIe.

---

## 2. Virtualization and Cloud HPC (vGPU, KVM)

* **Nvidia vGPU (Virtual GPU):**
    * **vGPU Architecture:** Understand vGPU technology for sharing physical GPUs across multiple virtual machines. Learn vGPU types (e.g., vComputeServer, vPC, vApp) and licensing.
    * **vGPU Deployment:** Deploy and configure vGPU on hypervisors. Understand GPU partitioning, time-slicing, and MIG (Multi-Instance GPU) for fine-grained sharing.
    * **vGPU for HPC and AI:** Configure vGPU environments for HPC workloads, ML training, and inference in virtualized data centers.

* **KVM and GPU Passthrough:**
    * **GPU Passthrough (VFIO):** Learn PCIe passthrough for dedicating physical GPUs to VMs. Understand IOMMU groups, VFIO drivers, and SR-IOV for GPU virtualization.
    * **KVM with Nvidia GPUs:** Configure KVM-based virtualization with Nvidia GPUs. Explore nested virtualization and GPU resource management.
    * **Orchestration:** Integrate GPU VMs with Kubernetes, Slurm, or other HPC job schedulers for resource allocation.

* **Containerization for HPC:**
    * **Nvidia Container Toolkit:** Use the Nvidia Container Toolkit to run GPU workloads in Docker and Podman containers.
    * **Singularity/Apptainer:** Deploy HPC applications with Singularity/Apptainer for GPU-accelerated containerized workloads in shared clusters.

**Resources:** Nvidia vGPU Software Documentation · Linux VFIO and IOMMU Documentation · Nvidia Container Toolkit.

**Projects:** Deploy a vGPU environment; configure GPU passthrough with KVM (VFIO).

---

## 3. HPC Interconnects and Storage

* **High-Speed Interconnects:**
    * **InfiniBand:** Master InfiniBand for low-latency, high-bandwidth HPC networking. Understand RDMA (Remote Direct Memory Access), GPUDirect RDMA, and topology design.
    * **RoCE (RDMA over Converged Ethernet):** Explore RoCE for Ethernet-based RDMA in HPC and cloud environments.
    * **GPUDirect Storage:** Learn GPUDirect Storage (GDS) for direct GPU-to-NVMe data access, bypassing CPU for I/O-bound workloads. *See [GPUDirect-Storage](./GPUDirect-Storage/README.md) deep dive above.*

* **Parallel File Systems and I/O:**
    * **Lustre and GPFS:** Understand parallel file systems for HPC storage. Optimize I/O patterns for large-scale scientific applications.
    * **DAOS (Distributed Asynchronous Object Storage):** Explore DAOS for next-generation HPC storage with native GPU support.

* **Job Scheduling and Orchestration:**
    * **Slurm with GPU Support:** Configure Slurm for GPU resource management, GRES (Generic Resources), and multi-node GPU jobs.
    * **Kubernetes for HPC/AI:** Use Kubernetes with Nvidia GPU operator for orchestrating GPU workloads in hybrid HPC/cloud environments.

**Resources:** Nvidia GPUDirect Documentation · Slurm GPU Configuration · TOP500 and Green500.

**Projects:** Build a multi-node GPU cluster with InfiniBand or high-speed Ethernet, NCCL, and Slurm; optimize I/O with GPUDirect Storage.

---

## Phase 2: Advanced HPC (24–48 months)

### 1. Advanced CUDA Programming for HPC

* **CUDA Memory Optimization:** Memory access coalescing, shared memory and L1 cache, pinned and unified memory. Analyze with Nsight Compute; restructure layouts (AoS → SoA).
* **Warp-Level and Thread-Level Optimization:** Warp divergence, warp shuffle intrinsics (`__shfl_sync`, `__ballot_sync`, `__reduce_sync`), Tensor Cores (WMMA/CUTLASS).
* **CUDA Graphs and Streams:** Overlap compute and transfer with streams; capture/replay with CUDA Graphs; cooperative groups. *See [CUDA-Advanced-Optimization](./CUDA-Advanced-Optimization/README.md) deep dive.*

**Resources:** Nsight Compute and Nsight Systems · CUTLASS · "CUDA Programming" by Shane Cook.

**Projects:** Optimize a GEMM kernel (tiling, shared memory, Tensor Cores) vs cuBLAS; overlapped pipeline with streams; CUDA Graph for inference.

---

### 2. Distributed Training and Large-Scale AI

* **Parallel Training Strategies:** Data parallelism (NCCL all-reduce), model parallelism (tensor + pipeline), 3D parallelism (DeepSpeed + Megatron).
* **Frameworks and Infrastructure:** PyTorch DDP and FSDP, DeepSpeed ZeRO (1/2/3), Megatron-LM. *See [8x-H200 Training Setup](./8x-H200-Training-Inference/02-Training-Setup.md) and [NCCL](./NCCL-Deep-Dive/03-Framework-Integration.md).*
* **Monitoring and Fault Tolerance:** Weights & Biases / TensorBoard; distributed checkpointing; PyTorch Elastic for node failure and resizing.

**Resources:** DeepSpeed Documentation · Megatron-LM GitHub · "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (paper).

**Projects:** 3D-parallel training run (>1B params); ZeRO-3 memory analysis; fault-tolerant training with Elastic.

---

### 3. HPC Performance Modeling and Application Optimization

* **Roofline Model and Performance Analysis:** Compute-bound vs memory-bound; arithmetic intensity (FLOPs/Byte); hierarchical memory rooflines (L1, L2, HBM). *See [8x-H200 Performance Optimization](./8x-H200-Training-Inference/05-Performance-Optimization.md).*
* **Scientific Computing Applications:** Molecular dynamics (GROMACS, AMBER), CFD (sparse solvers, CG, GMRES), cuFFT for FFT-based applications.
* **Compiler and Auto-Tuning:** TVM for GPU, Triton for custom kernels, cuBLAS/cuDNN tuning (workspace, algorithm selection, math modes).

**Resources:** Nsight Compute Roofline Analysis · "Programming Massively Parallel Processors" (Kirk & Hwu) · OpenAI Triton Documentation.

**Projects:** Roofline analysis of a scientific kernel; custom Triton kernel (e.g. layer norm + dropout); auto-tuned cuFFT pipeline.
