# NVIDIA CUDA-X Libraries

<div class="course-identity auto-course" style="--course-accent: #475569; --course-accent-rgb: 71, 85, 105;" markdown="1">
<div class="course-identity__icon">NCL</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Specialization</p>
<p class="course-identity__title">Specialized course identity for NVIDIA CUDA-X Libraries.</p>
<p class="course-identity__meta">Artifact: specialization case study · Measure: performance, reliability, role fit</p>
</div>
</div>


**Parent:** [Phase 5 — High Performance Computing](../Guide.md)

**Timeline:** Ongoing reference — study each category as your projects demand it.

**Prerequisites:** Phase 1 §4 (C++/CUDA), Phase 4 Track B (Jetson, CUDA runtime), Phase 4 Track C Part 2 (DL inference optimization).

---

## What CUDA-X is

**CUDA-X** is NVIDIA's full suite of GPU-accelerated libraries built on top of the CUDA runtime. While CUDA gives you the programming model (kernels, streams, memory), CUDA-X gives you **production-grade implementations** of the algorithms you'd otherwise spend months writing: linear algebra, FFTs, neural network primitives, graph analytics, video codecs, multi-GPU communication, and more.

Every layer of the AI chip stack consumes CUDA-X libraries:
- **L1 (Application):** cuDNN, TensorRT, DALI, CV-CUDA, DeepStream
- **L2 (Compiler):** CUTLASS, FlashInfer, cuDNN primitives as codegen targets
- **L3 (Runtime):** cuBLAS, cuFFT, NCCL, NVSHMEM, GPUDirect Storage
- **L1–L3 (Data):** cuDF, cuML, cuGraph, cuVS, NeMo Curator

Understanding what each library does — and when to use it vs. writing a custom kernel — is essential for any GPU infrastructure role.

---

## CUDA Math Libraries

The foundation for compute-intensive workloads: molecular dynamics, CFD, medical imaging, seismic exploration, and the matrix math inside every neural network.

| Library | What it does | Key APIs / Concepts | When to use it |
|---------|-------------|--------------------|-----------------------|
| **cuBLAS** | GPU-accelerated BLAS (Basic Linear Algebra Subprograms) | GEMM (`cublasSgemm`, `cublasGemmEx`), batched GEMM, mixed-precision (FP16/BF16/FP8 accumulate to FP32) | Any dense matrix multiply — the single most important library for AI inference and training |
| **cuFFT** | Fast Fourier Transform on GPU | 1D/2D/3D FFTs, real-to-complex, batched FFTs, multi-GPU FFT | Signal processing, audio, spectral analysis, conv via FFT |
| **cuRAND** | Random number generation on GPU | Pseudorandom (XORWOW, MRG32k3a, Philox) and quasirandom (Sobol) generators; host and device API | Monte Carlo simulations, dropout in training, data augmentation |
| **cuSOLVER** | Dense and sparse direct solvers | LU, Cholesky, QR, SVD, eigenvalue decomposition; sparse Cholesky | Linear systems, least squares, PCA, matrix factorization |
| **cuSPARSE** | Sparse matrix operations | SpMV, SpMM, sparse triangular solve; CSR, CSC, COO formats | GNNs, sparse attention, scientific computing with sparse matrices |
| **cuTENSOR** | Tensor contractions and operations | Tensor contraction, reduction, element-wise ops on multi-dimensional arrays | Quantum chemistry, tensor networks, high-dimensional contractions |
| **cuDSS** | Direct sparse solver | Sparse LU/Cholesky with GPU acceleration | Large sparse systems in structural analysis, circuit simulation |
| **CUDA Math API** | Standard math functions | `sin`, `cos`, `exp`, `log`, `sqrt` — GPU-optimized, single and double precision | Any kernel that needs standard math — these are intrinsics, not library calls |
| **AmgX** | Algebraic multigrid solver | AMG preconditioner + Krylov solvers for implicit unstructured methods | CFD, structural mechanics, any PDE solver using implicit methods |
| **nvmath-python** | Python interface to CUDA math | `nvmath.fft`, `nvmath.linalg` — NumPy-compatible but GPU-accelerated | Rapid prototyping of GPU math in Python without writing C++ |

### What to study first

**cuBLAS** — if you understand GEMM (shapes, leading dimensions, transposition, batching, mixed precision), you understand 80% of AI compute. Start here.

### Projects

1. **cuBLAS GEMM benchmark** — Benchmark `cublasGemmEx` for FP32, FP16, INT8 on your GPU. Plot TFLOPS vs matrix size. Find the crossover where GPU beats CPU BLAS.
2. **cuFFT spectral analysis** — Apply 2D FFT to an image, filter high frequencies, inverse FFT. Compare GPU vs CPU time for 4K images.
3. **cuSOLVER SVD** — Compute SVD of a large matrix on GPU. Use it for image compression (keep top-k singular values). Benchmark vs NumPy.

---

## Scientific Computing Libraries

For applications requiring neural networks that respect mathematical symmetries — molecular structures, proteins, materials science.

| Library | What it does | Domain |
|---------|-------------|--------|
| **cuEquivariance** | Accelerate geometry-aware neural networks (rotation/translation equivariance in 3D) | Molecular dynamics, protein folding, materials discovery |
| **NVIDIA ALCHEMI** | NIM microservices for chemical and materials discovery | Drug discovery, battery materials, catalysts |
| **cuLitho** | Computational lithography acceleration | Semiconductor manufacturing (L7/L8 connection) |
| **cuEST** | Electronic structure calculations on GPU | Quantum chemistry, DFT |

---

## Physics Libraries

GPU-accelerated physics simulation — relevant for robotics simulation, autonomous vehicle testing, and digital twins.

| Library | What it does | Use case |
|---------|-------------|----------|
| **NVIDIA Warp** | Python framework for GPU physics kernels | Simulation AI, robotics, differentiable physics |
| **NVIDIA PhysicsNeMo** | Training and fine-tuning physics AI models | Weather prediction, CFD surrogate models |
| **NVIDIA Earth-2** | Weather and climate AI models | Climate modeling, weather forecasting |

---

## Quantum Computing Libraries

| Library | What it does |
|---------|-------------|
| **cuQuantum** | Accelerate quantum circuit simulation on GPUs |
| **cuPQC** | Post-quantum cryptography acceleration |
| **CUDA-Q QEC** | Quantum error correction simulation |
| **CUDA-Q Solvers** | Hybrid quantum-classical optimization |

---

## Deep Learning Core Libraries

The libraries that **directly implement neural network inference and training**. These are the most critical for AI hardware engineers — they are the primary consumers of GPU compute.

| Library | What it does | Layer | Key concepts |
|---------|-------------|:-----:|-------------|
| **cuDNN** | Neural network primitives: conv, attention, normalization, pooling, activation | L1/L2 | `cudnnConvolutionForward`, `cudnnMultiHeadAttnForward`, graph API for fusion; auto-tuning selects fastest algorithm per hardware |
| **TensorRT** | Inference optimizer + runtime: graph optimization, layer fusion, precision calibration, engine serialization | L2/L3 | Build phase (optimize graph) → serialize → deploy phase (execute). INT8/FP8 calibration, dynamic shapes, DLA offload |
| **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** | LLM-specific inference engine: in-flight batching, paged KV-cache, tensor/pipeline parallelism, speculative decoding, FP8/INT4 | L2/L3 | Compiles LLMs to optimized TRT engines with custom Hopper/Blackwell kernels. Python API for build + deploy. Integrates with Triton for production serving. See [Phase 4C Part 2 Unit 05](../../../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md) for deep dive |
| **CUTLASS** | C++ template library for custom GEMM, conv, and attention kernels targeting Tensor Cores | L2/L3 | CuTe layout DSL, tile-based programming, epilogue fusion, warp-level MMA. The reference for how production GEMM kernels are structured |
| **FlashInfer** | Optimized attention and MoE kernels for LLM inference via Python API | L2 | Paged KV-cache attention, variable-length sequences, speculative decoding support |

### What to study first

1. **cuDNN** — understand what primitives exist and how frameworks (PyTorch, TensorRT) call them
2. **TensorRT** — end-to-end inference optimization for vision/small models (covered in Phase 4 Track C Part 2 and Track B §8)
3. **TensorRT-LLM** — LLM-specific inference: in-flight batching, paged KV-cache, multi-GPU serving (covered in Phase 4 Track C Part 2 Unit 05)
4. **CUTLASS** — when you need to write or modify GEMM/attention kernels (covered in Phase 4 Track C Part 2, unit 02)

### Projects

1. **cuDNN algorithm benchmark** — For a specific conv layer (ResNet-50 first conv), enumerate all cuDNN algorithms (`cudnnFindConvolutionForwardAlgorithm`). Compare time and workspace for each. Understand why auto-tuning matters.
2. **CUTLASS custom GEMM** — Build a CUTLASS GEMM with a fused bias+ReLU epilogue. Benchmark against cuBLAS. Study the CuTe layout to understand tiling.
3. **FlashInfer attention** — Run FlashInfer paged attention on a transformer model. Compare latency vs standard PyTorch attention for 4K, 16K, 64K context lengths.

---

## Parallel Algorithm Libraries (CCCL)

**CUDA Core Compute Libraries** — fundamental building blocks for writing GPU algorithms in C++ and Python.

| Library | What it does | When to use it |
|---------|-------------|---------------|
| **Thrust** | C++ STL-like parallel algorithms: sort, reduce, scan, transform | High-level parallel algorithms without writing raw CUDA kernels |
| **CUB** | Warp-wide, block-wide, and device-wide cooperative primitives | Inside custom CUDA kernels — block reduce, block scan, radix sort |
| **cuda.compute** | Python interface to CCCL device-level algorithms | GPU-accelerated algorithms from Python |
| **cuda.parallel** | Standardized sort, scan, reduction primitives | Distributed and local parallel patterns |

### What to study first

**CUB** — if you write custom CUDA kernels, you'll use CUB's `BlockReduce`, `BlockScan`, and `WarpReduce` constantly. It's the building block layer.

### Projects

1. **Thrust vs custom kernel** — Implement parallel prefix sum (scan) using Thrust, CUB, and a hand-written kernel. Benchmark all three. Understand the abstraction cost.
2. **CUB histogram** — Build a GPU histogram using CUB's `BlockHistogram`. Compare with `atomicAdd`-based approach.

---

## Data Processing Libraries

GPU-accelerated data pipelines — critical for feeding models fast enough that the GPU doesn't starve.

| Library | What it does | Replaces |
|---------|-------------|----------|
| **cuDF** | GPU-accelerated DataFrames | pandas, Polars, Spark (zero code changes) |
| **cuVS** | GPU vector search (nearest neighbors, CAGRA algorithm) | FAISS, Annoy — for RAG, recommendation, semantic search |
| **cuML** | GPU-accelerated ML algorithms | scikit-learn, UMAP, HDBSCAN (zero code changes) |
| **cuOpt** | Decision optimization engine (millions of variables) | OR-Tools, Gurobi for routing/scheduling |
| **cuGraph** | GPU graph analytics | NetworkX — for knowledge graphs, social networks |
| **NeMo Curator** | Data pipeline for training/fine-tuning LLMs | Custom data cleaning scripts — text, image, video at scale |
| **Morpheus** | Cybersecurity AI pipeline | Custom SIEM analysis pipelines |
| **nvComp** | GPU-accelerated compression/decompression | zstd, lz4 — for training data I/O bottlenecks |
| **GPUDirect Storage** | Direct path: NVMe → GPU memory, bypassing CPU | Standard `read()` → `cudaMemcpy()` — eliminates bounce buffers |
| **Dask** | Distributed computing framework with RAPIDS GPU support | Single-node pandas/scikit-learn — scales to clusters |

### Projects

1. **cuDF vs pandas** — Load a 10M-row CSV with pandas and cuDF. Benchmark groupby, join, filter operations. Measure speedup.
2. **GPUDirect Storage benchmark** — Compare model loading time with standard I/O vs GDS direct-to-GPU path. Measure throughput in GB/s.

---

## Image and Video Libraries

Hardware-accelerated encode/decode and vision preprocessing — these feed the inference pipeline.

| Library | What it does | Use case |
|---------|-------------|----------|
| **nvImageCodec** | GPU image encode/decode (JPEG, PNG, WebP) | High-throughput data loading for training |
| **NVIDIA DALI** | GPU-accelerated data loading and preprocessing for DL | Training pipelines — replaces CPU-bound DataLoader |
| **CV-CUDA** | GPU pre/post-processing for vision AI pipelines | Production inference: resize, normalize, color convert on GPU |
| **cuCIM** | Accelerated image processing for biomedical/geospatial | Whole-slide imaging, satellite imagery |
| **NPP (Performance Primitives)** | 2D image and signal processing primitives | Filtering, color conversion, morphological operations |
| **Video Codec SDK** | Hardware NVDEC/NVENC encode and decode | Video analytics, transcoding, DeepStream input |
| **Optical Flow SDK** | Hardware-accelerated optical flow | Motion estimation, video stabilization, frame interpolation |

### Projects

1. **DALI training pipeline** — Replace PyTorch DataLoader with DALI for ImageNet training. Measure throughput (images/sec) improvement.
2. **CV-CUDA inference preprocessing** — Build a preprocessing pipeline (resize + normalize + color convert) using CV-CUDA. Compare latency vs OpenCV CPU.
3. **Video decode pipeline** — Decode a 4K video stream using NVDEC (Video Codec SDK) → feed frames to a detection model via TensorRT. Measure end-to-end FPS.

---

## Communication Libraries

**Multi-GPU and multi-node communication** — the bottleneck at scale. These libraries determine how fast your cluster can train and serve models.

| Library | What it does | Key operations | When to use it |
|---------|-------------|---------------|---------------|
| **NCCL** | Multi-GPU/multi-node collectives optimized for NVIDIA hardware | AllReduce, AllGather, ReduceScatter, Broadcast, Send/Recv | Distributed training (gradient sync), model parallelism, inference sharding |
| **NVSHMEM** | Partitioned global address space across GPU memories | `nvshmem_put`, `nvshmem_get` — one-sided remote memory access | Fine-grained GPU-to-GPU communication without collective overhead |
| **NIXL** | Low-latency inference transfer library | KV-cache migration, tensor transfer between GPUs/memory tiers | LLM inference serving — moving KV-cache for disaggregated serving |

### What to study first

**NCCL** — if you work on distributed training or multi-GPU inference, NCCL is the library you'll interact with most. Understand AllReduce (data parallelism), AllGather (model parallelism), and how to overlap communication with compute.

### Projects

1. **NCCL AllReduce benchmark** — Run `nccl-tests` on 2+ GPUs. Measure bandwidth and latency for AllReduce across different message sizes. Compare NVLink vs PCIe topology.
2. **Communication-compute overlap** — In a simple 2-GPU training loop, overlap gradient AllReduce with the next forward pass. Measure the throughput improvement vs synchronous AllReduce.

---

## Partner Libraries

Community and third-party libraries with GPU acceleration via CUDA.

| Library | What it does |
|---------|-------------|
| **OpenCV** | Computer vision, image processing, ML — GPU-accelerated modules |
| **FFmpeg** | Multimedia framework — NVDEC/NVENC hardware codec integration |
| **ArrayFire** | High-level C++/Python GPU array library |
| **CuPy** | NumPy/SciPy-compatible GPU array library for Python |
| **MAGMA** | GPU linear algebra for heterogeneous architectures |
| **Gunrock** | GPU graph processing library |

---

## Which CUDA-X Libraries to Learn by Role

| Your role | Must know | Should know | Nice to have |
|-----------|----------|------------|-------------|
| **ML Inference Optimization Engineer** | cuDNN, TensorRT, CUTLASS | cuBLAS, NCCL, CV-CUDA, DALI | FlashInfer, nvComp |
| **AI Compiler Engineer** | CUTLASS, cuDNN (as codegen target), cuBLAS | TensorRT, FlashInfer | CUB, Thrust |
| **GPU Runtime Engineer** | cuBLAS, NCCL, NVSHMEM, GPUDirect Storage | CUB, Thrust, CUDA Math API | nvComp, NIXL |
| **GPU Infrastructure / HPC Engineer** | NCCL, NVSHMEM, GPUDirect Storage, Slurm | cuBLAS, cuFFT, nvComp | Dask, cuDF |
| **Edge AI / Jetson Engineer** | TensorRT, cuDNN, VPI, DeepStream | CV-CUDA, Video Codec SDK, NPP | DALI, GStreamer |
| **AI Accelerator Architect** | CUTLASS (understand what hardware must support), cuDNN | cuBLAS, TensorRT | FlashInfer, NCCL |

---

## Resources

| Resource | URL |
|----------|-----|
| CUDA-X Library Overview | https://developer.nvidia.com/gpu-accelerated-libraries |
| CUDA Toolkit Documentation | https://docs.nvidia.com/cuda/ |
| cuBLAS Documentation | https://docs.nvidia.com/cuda/cublas/ |
| cuDNN Documentation | https://docs.nvidia.com/deeplearning/cudnn/ |
| CUTLASS GitHub | https://github.com/NVIDIA/cutlass |
| NCCL Documentation | https://docs.nvidia.com/deeplearning/nccl/ |
| TensorRT Documentation | https://docs.nvidia.com/deeplearning/tensorrt/ |
| FlashInfer Documentation | https://docs.flashinfer.ai/ |
| RAPIDS (cuDF, cuML, cuGraph) | https://rapids.ai/ |
| NVIDIA Developer Program | https://developer.nvidia.com/developer-program |
