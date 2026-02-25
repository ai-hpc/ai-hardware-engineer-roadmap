**5. AI Chip Design (18-36 months)**

**Prerequisite: Master AI/Software Stack First**

* **Why Software First:**
    * **Hardware-Software Co-Design:** AI chip design requires deep understanding of the software stack—training frameworks, inference runtimes, and operator semantics. Design hardware that software actually needs.
    * **Workload Characterization:** Profile real AI workloads (training, inference) to identify bottlenecks. Understand memory bandwidth, compute intensity, and data reuse patterns before designing hardware.
    * **Reference: tinygrad:** Study tinygrad—a minimal, readable deep learning framework. It exposes the essence of tensor operations, autograd, and code generation. Understanding tinygrad helps you see what hardware must accelerate.

* **tinygrad and Minimal ML Frameworks:**
    * **tinygrad Architecture:** Study tinygrad's lazy evaluation, linearized IR (intermediate representation), and code generation for different backends (CPU, GPU, custom).
    * **Operator Semantics:** Understand how tinygrad implements conv2d, matmul, attention, and other ops. Trace the graph and memory access patterns.
    * **Extending tinygrad:** Add a custom backend or new op. This teaches you the interface between software and hardware.
    * **Other Minimal Frameworks:** Explore mlx, JAX (for understanding tracing and compilation), or Triton (for GPU kernel design).

**Resources:**

* **tinygrad GitHub:** https://github.com/tinygrad/tinygrad — Minimal and readable. Study the codebase.
* **tinygrad Learning Materials:** See [4. Autonomous Driving/tinygrad](../4.%20Autonomous%20Driving/tinygrad/) for hands-on guides, ops reference, and Jetson support.
* **"Tinygrad: A Simple Autograd Engine" (blog/videos):** George Hotz's explanations of tinygrad design.
* **"Computer Architecture: A Quantitative Approach" (Hennessy & Patterson):** Foundation for understanding accelerator design.

**Projects:**

* **Implement a Custom tinygrad Backend:** Target a simple accelerator (e.g., FPGA, custom simulator) from tinygrad.
* **Profile and Optimize a Model in tinygrad:** Identify compute-bound vs. memory-bound layers. Propose hardware optimizations.


**2. AI Accelerator Architecture**

* **Compute and Memory:**
    * **Systolic Arrays and TPUs:** Understand systolic array architecture for matrix multiplication. Study Google TPU and similar designs.
    * **Dataflow Architectures:** Learn dataflow (e.g., Eyeriss, NVDLA) vs. control-flow architectures. Understand spatial vs. temporal compute.
    * **Memory Hierarchy:** Design for memory bandwidth—on-chip SRAM, HBM, and data reuse. Roofline model for accelerators.

* **Quantization and Precision:**
    * **INT8/INT4 Inference:** Quantization-aware design for low-precision compute. Understand the trade-offs for different precisions.
    * **Mixed-Precision Training:** FP16/BF16 for training. Study gradient scaling and loss scaling.

* **Compiler and Runtime:**
    * **TVM and MLIR:** Study TVM for compiling high-level models to hardware. Understand MLIR dialects for AI accelerators.
    * **Kernel Fusion and Scheduling:** Learn how compilers fuse ops and schedule kernels for optimal performance.

**Resources:**

* **"Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNN" (paper):** Classic accelerator architecture.
* **"A Full-Stack Accelerator for Deep Learning" (NVDLA):** Open-source Nvidia design.
* **TVM and Apache TVM:** Compiler stack for deep learning.

**Projects:**

* **Design a Simple Matrix Multiply Accelerator:** Specify the architecture (e.g., systolic array) for a small accelerator in RTL or high-level synthesis.
* **Map a tinygrad Model to Your Accelerator:** Define the mapping from tinygrad ops to your hardware.


**3. From Software to Silicon**

* **RTL and Verification:**
    * **Verilog/SystemVerilog for Accelerators:** Implement key datapath and control blocks for an AI accelerator.
    * **High-Level Synthesis (HLS):** Use HLS (Vitis HLS, etc.) to accelerate design for matrix ops and custom kernels.
    * **Formal Verification:** Apply formal methods for critical datapath correctness.

* **FPGA Prototyping:**
    * **FPGA as Accelerator:** Deploy a small AI accelerator on FPGA. Use frameworks like FINN (Xilinx) for quantized neural networks.
    * **Co-Design with CPUs/GPUs:** Integrate FPGA accelerators with host systems. Understand PCIe, DMA, and driver interfaces.

* **ASIC and Tape-Out (Advanced):**
    * **Physical Design:** Overview of place-and-route, timing closure, and power analysis for AI chips.
    * **Industry and Startups:** Study commercial AI chips (Nvidia, AMD, Cerebras, Groq, etc.) and startup approaches.

**Resources:**

* **FINN (Xilinx):** Framework for building fast, flexible FPGA accelerators for neural networks.
* **NVDLA (Nvidia):** Open-source deep learning accelerator.
* **Chip design courses (e.g., Berkeley, Stanford):** Online courses on digital design and computer architecture.

**Projects:**

* **Implement a Small Accelerator on FPGA:** Build a matrix multiply or conv2d accelerator using HLS or RTL.
* **Compare tinygrad on CPU vs. Your FPGA Accelerator:** Benchmark and analyze the speedup and efficiency.


**Phase 2 (Significantly Expanded): AI Chip Design (36-60 months)**

**1. Advanced Accelerator Microarchitecture**

* **Dataflow Optimization:**
    * **Dataflow Architectures (In-Depth):**  Study the "energy-efficient dataflow for CNN" design space from the Eyeriss framework. Understand the seven categories of dataflow (Weight Stationary, Output Stationary, Input Stationary, No Local Reuse, Row Stationary) and their memory bandwidth and energy trade-offs.
    * **Spatial vs. Temporal Compute:**  Design spatial architectures (like TPUs, Groq TSP) where computation is statically mapped to processing elements. Compare with temporal architectures (like GPUs) that share resources across time.
    * **Dataflow Compilers:**  Study dataflow compilers (Halide, TVM schedules, MLIR Linalg dialect) that map tensor operations onto spatial architectures. Understand the affine loop analysis and polyhedral optimization that underpins them.

* **Processing Element (PE) Design:**
    * **Systolic Array Design:**  Implement a parameterized systolic array for matrix multiply in RTL—configure PE count, data precision, and accumulation depth. Analyze throughput, latency, and area via FPGA or synthesis reports.
    * **Mixed-Precision and Approximate Computing:**  Design PEs that support multiple precisions (FP32, FP16, BF16, INT8, INT4) with configurable precision at runtime. Explore approximate computing (stochastic rounding, reduced-precision accumulation) for energy efficiency.
    * **Sparse Accelerators:**  Study architectures that exploit sparsity in weights and activations—Nvidia's A100 structured sparsity (2:4 sparsity), SCNN (sparse CNN accelerator), and Cambricon-X. Implement sparse GEMM on FPGA.

* **On-Chip Network (NoC) and Memory System:**
    * **NoC Design:**  Design on-chip interconnect for multi-PE accelerators—mesh, ring, and hierarchical NoC topologies. Analyze bandwidth, latency, and power for different workload communication patterns.
    * **Scratchpad vs. Cache:**  Understand the trade-offs between scratchpad (programmer-managed SRAM) and cache-based on-chip memory. Design scratchpad controllers with DMA engines for bulk data movement.
    * **HBM Interface Design:**  Study HBM2/HBM3 interface requirements—PHY, memory controller design, and request scheduling. Understand how on-chip bandwidth limits accelerator performance and design memory subsystems accordingly.

**Resources:**

* **"Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNN" (Chen et al., ISCA 2016):**  Foundational paper on CNN accelerator dataflow and energy analysis.
* **"Efficient Processing of Deep Neural Networks" by Sze, Chen, Yang, and Emer:**  Comprehensive book and MIT OCW course on DNN hardware acceleration.
* **Timeloop and Accelergy:**  Architecture-level modeling and evaluation framework for DNN accelerators—maps workloads to architectures and estimates performance, energy, and area.
* **Halide** — [github.com/halide/Halide](https://github.com/halide/Halide): Language for fast, portable data-parallel computation (image processing, GPU). Study the schedule/algorithm separation and how it maps to spatial architectures.

**Projects:**

* **Parameterized Systolic Array:**  Implement a scalable systolic array (e.g., 16×16 PEs) in SystemVerilog with configurable data width (INT8/FP16). Synthesize on FPGA and measure TOPS/W.
* **Dataflow Analysis with Timeloop:**  Use Timeloop to evaluate three different dataflows (Weight Stationary, Output Stationary, Row Stationary) for ResNet-50 on a hypothetical accelerator. Compare energy efficiency and identify optimal dataflow.
* **Sparse GEMM on FPGA:**  Implement a 2:4 structured sparse matrix multiply on FPGA using HLS. Compare performance and resource utilization against dense GEMM.


**2. Chip Architecture and Physical Design**

* **RTL-to-GDS Flow:**
    * **Synthesis and Timing:**  Run logic synthesis with commercial tools (Synopsys Design Compiler, Cadence Genus) or open-source (Yosys + ABC). Understand technology mapping, multi-corner multi-mode (MCMM) timing, and area/power/timing optimization.
    * **Floorplanning and Placement:**  Plan chip floorplan—allocate areas for compute arrays, SRAM, I/O, and control logic. Understand placement density, routing congestion, and power/ground distribution network planning.
    * **Routing and Signoff:**  Complete physical routing (global + detailed), perform signoff checks—DRC (Design Rule Check), LVS (Layout vs. Schematic), and timing signoff with parasitic extraction (RC extraction, STA with SPEF).

* **Open-Source EDA and OpenROAD:**
    * **OpenROAD Flow:**  Run the OpenROAD open-source RTL-to-GDS flow for a small AI accelerator design. Use the SkyWater 130nm or GlobalFoundries 180nm PDK for a manufacturable design.
    * **Magic and KLayout:**  View and edit VLSI layouts using Magic and KLayout. Understand GDSII format, layer mappings, and design rule enforcement.
    * **Efabless Chipignite:**  Explore multi-project wafer (MPW) shuttle programs (Efabless chipIgnite, TinyTapeout) for low-cost physical chip fabrication of student designs.

* **Power Analysis and Optimization:**
    * **Dynamic and Static Power:**  Analyze dynamic power (switching activity × capacitance × voltage²) and static (leakage) power using simulation-based (VCD/SAIF) or statistical estimation tools.
    * **Clock Gating and Power Domains:**  Implement fine-grained clock gating to disable idle logic and multi-voltage power domains (UPF/CPF) to supply inactive blocks at reduced voltage.
    * **IR Drop and Electromigration:**  Analyze power distribution network (PDN) IR drop and electromigration limits. Optimize metal width and via arrays for reliable power delivery under peak current loads.

**Resources:**

* **OpenROAD Project:**  Open-source RTL-to-GDS flow with commercial-quality tools for academia and research.
* **TinyTapeout:**  Educational chip fabrication shuttle—design a small chip and get it fabricated as part of a multi-project wafer.
* **"VLSI Physical Design: From Graph Partitioning to Timing Closure" by Kahng, Lienig, Markov, and Hu:**  Comprehensive physical design textbook.

**Projects:**

* **OpenROAD Accelerator Tape-Out:**  Take a small RTL design (e.g., an 8×8 INT8 systolic array) through the OpenROAD flow on SkyWater 130nm. Generate GDSII and analyze timing, area, and power reports.
* **TinyTapeout Submission:**  Design a minimal AI inference block (e.g., a 4-element dot product unit) and submit it to a TinyTapeout MPW shuttle for physical fabrication.
* **Power Domain Analysis:**  Implement a multi-voltage design (compute core at 0.8V, I/O at 1.2V) in a simulation environment. Analyze leakage power savings and verify UPF correctness.


**3. AI Chip Industry, Business, and Research Frontier**

* **AI Chip Landscape:**
    * **Commercial AI Chip Analysis:**  Study the architectures of commercial AI chips—Nvidia H100 (SXM5), AMD MI300X (CDNA3), Google TPUv4, Cerebras WSE-3, Groq TSP, Graphcore IPU, and SambaNova RDU. Compare performance, memory bandwidth, precision support, and total cost of ownership.
    * **Inference vs. Training Chips:**  Understand the different design points for training (high-precision, large memory, flexible) vs. inference (low-latency, low-power, quantized, fixed function) accelerators.
    * **Edge AI Chips:**  Study edge inference chips—Apple Neural Engine, Qualcomm Hexagon DSP, Hailo-8, Kendryte K210—and their design trade-offs for power-constrained embedded deployment.

* **Chiplets and Advanced Packaging:**
    * **Chiplet Architecture:**  Understand chiplet-based design where a large chip is disaggregated into smaller dies connected via high-bandwidth die-to-die interconnects (UCIe, BoW, AIB, HBI).
    * **2.5D and 3D Integration:**  Study 2.5D packaging (HBM on an interposer, as in AMD Instinct MI series) and 3D stacking (TSMC SoIC, Intel Foveros) for integrating compute and memory dies.
    * **Design for Chiplets:**  Understand PHY design for die-to-die interfaces, power delivery challenges in stacked dies, and thermal management for 3D-integrated AI chips.

* **Research Directions:**
    * **Neuromorphic Computing:**  Study spike-based neuromorphic architectures (Intel Loihi, IBM TrueNorth) and their potential advantages for event-driven, sparse, ultra-low-power AI processing.
    * **In-Memory and Near-Memory Computing:**  Explore compute-in-memory (CIM) architectures that perform computations within SRAM or DRAM arrays, dramatically reducing data movement energy—the dominant cost in AI inference.
    * **Photonic and Analog AI:**  Survey emerging modalities—optical neural networks (Lightelligence, Lightmatter), analog compute-in-memory (IBM PCM, Mythic), and their potential paths to commercialization.

**Resources:**

* **Chip Architects Podcast and SemiAnalysis:**  Industry analysis of AI chip architectures, competitive landscape, and technology trends.
* **Hot Chips and ISSCC Conference Proceedings:**  Annual conferences where leading companies present AI chip architectures—primary source for understanding state-of-the-art designs.
* **"Demystifying AI Chipmakers" (Various analyst reports):**  Business and technology landscape of AI chip startups and incumbents.

**Projects:**

* **AI Chip Architecture Comparison:**  Perform a systematic comparison of three AI chips (e.g., H100, MI300X, TPUv4) across performance, memory bandwidth, precision, and TCO for LLM training. Present findings as a technical report.
* **Chiplet Design Study:**  Design a simple chiplet system with a compute die and an HBM-equivalent memory die. Model die-to-die bandwidth, latency, and power using UCIe specifications.
* **Research Proposal:**  Write a 2-page research proposal for a novel AI accelerator architecture targeting a specific gap (e.g., long-context transformer inference, sparse GNN acceleration). Include workload analysis, proposed architecture, and expected benefits.


---

## 4. Custom ML Framework Engineering: NVIDIA-Native Design

> This section grows directly from a critical engineering question: tinygrad claims it will be 2x faster than PyTorch on a single NVIDIA GPU — but is that true, and if not, what would it actually take to build a framework that genuinely wins on NVIDIA hardware?

---

### 4.1 The Tinygrad 2x Claim: An Honest Assessment

Tinygrad's README states: *"Will leave alpha when it can reproduce common papers 2x faster than PyTorch on 1 NVIDIA GPU."* This is an engineering exit condition for the alpha milestone, not a current performance reality.

**What the benchmarks actually show (as of early 2026):**

| Workload | Tinygrad vs PyTorch | Notes |
|---|---|---|
| **Training (NVIDIA)** | Roughly 2x **slower** | The gap tinygrad is actively trying to close — PyTorch+cuDNN+torch.compile is deeply co-engineered with NVIDIA |
| **Inference (NVIDIA, simple ops)** | Within ~10% of peak bandwidth | Competitive for memory-bound inference where cuDNN advantage is smallest |
| **Non-NVIDIA hardware** | Often **fastest** available | Genuine win on Qualcomm, AMD, Apple Metal where no cuDNN equivalent exists |
| **Openpilot (Snapdragon 845)** | 2x faster than SNPE | The closest thing to a real 2x win — but on Qualcomm, not NVIDIA |

**Why NVIDIA is specifically hard:**

1. **The cuDNN moat**: PyTorch calls `cudnnConvolutionForward()` which selects from hundreds of hand-tuned, architecture-specific kernels (Volta, Ampere, Hopper, Blackwell). These represent 10+ years of NVIDIA engineer-hours and are closed-source. A codegen system cannot reproduce them from generic loop nests.

2. **FlashAttention-3 integration**: PyTorch's `scaled_dot_product_attention()` (SDPA) dispatches to FlashAttention-3 on Hopper — achieving **740 TFLOP/s (75% of peak H100 throughput)** via Hopper-specific hardware features (TMA, WGMMA, warp specialization). Tinygrad cannot derive this kernel from its 12 primitive UOps.

3. **torch.compile + CUDA Graphs**: PyTorch 2.x traces the model, generates Triton kernels for elementwise/reduction ops, calls cuDNN/cuBLAS for heavy ops, and replays the full sequence via `cudaGraphLaunch()` — eliminating all CPU overhead with zero Python dispatch cost per inference step.

4. **NCCL for multi-GPU**: All-reduce, all-gather, and reduce-scatter using NVLink topology-aware algorithms. On NVSwitch systems, multicast addressing reduces the data volume to one message per step regardless of GPU count.

5. **NVIDIA co-engineering**: NVIDIA provides PyTorch with early hardware access and validates cuDNN/cuBLAS compatibility before each GPU generation launches. Tinygrad must reverse-engineer or re-implement those capabilities afterward.

**Tinygrad's realistic path to the 2x milestone**: Most likely on a **transformer inference workload** (LLaMA decode) where CUDA Graphs eliminate CPU overhead, FP8 doubles GEMM throughput, and the lazy scheduler fuses dequantization into adjacent elementwise ops — none of which PyTorch eager does by default. This is a narrow but real win condition, and it is exactly where this NVIDIA-native framework should compete.

---

### 4.2 What NVIDIA Hardware Provides That Generic Backends Miss

Understanding these features is mandatory for both framework design and AI chip design — this is the software side of hardware-software co-design.

**Tensor Cores: Three Generations**

| API Level | Architecture | PTX Instruction | Granularity |
|---|---|---|---|
| WMMA (C++) | Volta (sm_70+) | `mma.sync.aligned.*` | Warp (32 threads) 16×16 fragments |
| MMA (PTX) | Ampere (sm_80+) | `mma.sync.aligned.m16n8k16.*` | Warp-level, finer precision control (TF32, BF16, INT8) |
| WGMMA (PTX) | Hopper (sm_90) | `wgmma.mma_async.sync.aligned.*` | Warpgroup (128 threads), asynchronous — overlaps with TMA data movement |

A generic loop codegen cannot automatically discover that a reduce-of-multiply pattern maps to `wgmma.mma_async`. The framework must explicitly detect this graph pattern and emit the correct PTX.

**Memory System: From cp.async to TMA**

* **Registers → Shared Memory → Global Memory** is the classic GPU memory hierarchy. Each hop has latency and bandwidth constraints.
* **`cp.async` (Ampere, sm_80+):** Copies from global memory directly to shared memory *without routing through registers* — freeing 32K registers per SM for compute. Enables **software-pipelined double buffering**: load tile N+1 into shared memory simultaneously with computing on tile N.
* **TMA — Tensor Memory Accelerator (Hopper, sm_90):** A dedicated hardware unit that accepts a tensor descriptor (base pointer, shape, stride, swizzle pattern) and autonomously moves full tensor tiles between global and shared memory. A single thread issues the TMA instruction while all other threads continue computing. This eliminates all address arithmetic from thread code.

**Warp-Level Primitives**

* **`__shfl_xor_sync()`**: A thread reads another thread's register directly — no shared memory needed. Enables warp-level reductions in `log₂(32) = 5` instructions instead of 32 `atomicAdd` calls. Any framework emitting shared-memory reductions for warp-sized problems is leaving 5× latency on the table.
* **Warp vote functions** (`__ballot_sync`, `__any_sync`, `__all_sync`): Collective predicates across all 32 threads in one instruction — used for conditional execution and sparse pattern detection.

**Persistent Kernels**

In conventional GPU execution, each Cooperative Thread Array (CTA) processes one tile and exits. The scheduler then launches the next CTA. For many small tiles, this re-scheduling overhead is significant.

Persistent kernels keep CTAs alive across multiple tiles: a CTA finishes tile N, atomically fetches tile N+1's index, and continues. This is the execution model required for warp specialization on Hopper — producer warpgroups issue TMA requests while consumer warpgroups run WGMMA continuously. CUTLASS 3.x uses persistent kernels as the default for all Hopper GEMMs.

**Warp Specialization (Hopper)**

Different warpgroups within a CTA take different code paths:
* **Producer warpgroup**: Issues TMA loads for Q, K, V tiles. Signals completion via `mbarrier`.
* **Consumer warpgroup**: Waits on `mbarrier`, runs `wgmma.mma_async` on loaded tiles, accumulates to output.

This producer-consumer overlap is the primary reason FlashAttention-3 reaches 75% of H100 theoretical throughput. It requires structured control flow that is impossible to express in a single-program-multiple-thread (SPMT) model.

**Additional NVIDIA-Specific Capabilities**

* **CUDA Graphs**: Record kernel launches into a DAG (`cudaGraph_t`), replay with `cudaGraphLaunch()` — zero CPU overhead, zero Python dispatch, zero CUDA driver call per step. For LLM inference with static shapes: **1.2–3× speedup** over eager execution on top of an already-fast kernel sequence.
* **2:4 Structured Sparsity (Ampere+)**: In every group of 4 values, exactly 2 must be zero. The hardware stores compressed values + 2-bit indices and executes at **2× Tensor Core throughput** with decompression in-flight. Requires training with 2:4 pruning; compressed weights fed to `cuSPARSELt`.
* **L2 Cache Persistence (Ampere+)**: Mark a buffer with `cudaAccessPropertyPersisting` so it stays in the 40 MB L2 across kernel launches. For transformer inference where the KV-cache is reused every step, this eliminates repeated global memory fetches.

---

### 4.3 Designing an NVIDIA-Native Framework: Architecture

**Primary target: inference. Secondary target: transformer training on NVIDIA.** This focus narrows the competitor landscape and makes the win conditions concrete.

---

**Competitor Landscape: Who You Are Actually Fighting**

| Competitor | Strength | Exploitable Weakness |
|---|---|---|
| **TensorRT** | Fastest conv inference; per-layer plugin fusion | Plugin system cannot fuse *across* plugin boundaries; requires manual graph surgery for custom ops; terrible developer experience for non-standard models |
| **TRT-LLM** | NVIDIA's official LLM inference stack; FP8 on H100 | Opaque C++ internals; slow iteration cycle; not hackable; tightly coupled to TensorRT's graph partitioning |
| **vLLM** | PagedAttention for dynamic KV-cache; continuous batching | PyTorch eager backend — no CUDA Graphs, high CPU dispatch overhead per step; operator fusion is minimal |
| **PyTorch eager + torch.compile** | Fast iteration; good for training | Compile overhead; per-kernel Triton approach misses cross-layer fusion opportunities at inference |
| **ONNX Runtime + TensorRT EP** | Good for fixed-graph ONNX models | Cannot express dynamic control flow; loses fusion opportunities at ONNX import boundaries |

**The structural opening**: A lazy-evaluation whole-graph scheduler that sees the full attention + FFN block as a single fusible unit can eliminate intermediate buffer allocations that TensorRT's plugin-boundary model cannot. Combined with CUDA Graph replay and inference-specific memory management (PagedAttention), this is a real architectural advantage — not a marketing claim.

---

**What to Keep from Tinygrad**

| Component | Why Keep It |
|---|---|
| Lazy evaluation + UOp DAG | Sees the full model graph — fuses across attention + FFN + normalization boundaries that per-kernel compilers cannot |
| `ShapeTracker` (movement ops) | Zero-copy reshape/permute/expand — critical for KV-cache management without buffer copies |
| Scheduler (kernel boundary detection) | The most valuable piece: decides which ops collapse into one kernel launch |
| BEAM search auto-tuner | Finds optimal tile sizes, unroll factors, shared memory layouts per GPU generation |
| `Tensor` API + `nn` module | PyTorch-compatible — load any PyTorch model weight without conversion |
| Python-first, hackable codebase | Entire compiler visible; rapid iteration on new quantization schemes and kernel templates |

---

**The Two-Mode Compiler Pipeline**

```
                    ┌─────────────────────────────────┐
                    │     UOp DAG (tinygrad base)      │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │   Pattern Matcher (NVIDIA Ext.)  │
                    │  matmul-reduce → WGMMA/MMA       │
                    │  softmax(Q@K.T)@V → ATTN_KERNEL  │
                    │  elementwise chain → FUSED_EW    │
                    │  fixed-shape exec → GRAPH_REGION │
                    └────────────┬────────────────────┘
                                 │
              ┌──────────────────┼──────────────────────┐
              │ INFERENCE PATH   │                       │ TRAINING PATH
              ▼                  │                       ▼
  ┌───────────────────────┐      │         ┌────────────────────────────┐
  │ Inference Scheduler   │      │         │ Training Scheduler         │
  │ - Static shape spec.  │      │         │ - Gradient graph extension │
  │ - KV-cache allocation │      │         │ - Gradient checkpointing   │
  │ - Batch slot mgmt     │      │         │ - BF16 loss scaling        │
  │ - FP8 quant pipeline  │      │         │ - NCCL all-reduce hooks    │
  └───────────┬───────────┘      │         └──────────────┬─────────────┘
              │                  │                        │
              ▼                  │                        ▼
  ┌───────────────────────┐      │         ┌────────────────────────────┐
  │ Kernel Template Lib   │      │         │ Kernel Template Lib        │
  │ [Inference]           │      │         │ [Training]                 │
  │ FlashDecoding (FA-3)  │      │         │ FlashAttention-3 (forward) │
  │ PagedAttention KV     │      │         │ FlashAttention-3 (backward)│
  │ FP8 E4M3 GEMM         │      │         │ BF16/TF32 GEMM (CUTLASS)  │
  │ INT8/INT4 GEMM        │      │         │ Fused AdamW (FP8 master)  │
  │ 2:4 sparse weights    │      │         │ Gradient all-reduce        │
  └───────────┬───────────┘      │         └──────────────┬─────────────┘
              │                  │                        │
              └──────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │      PTX Emitter (NVIDIA)        │
                    │  wgmma.mma_async (Hopper)        │
                    │  mma.sync.aligned (Ampere)       │
                    │  cp.async.bulk.tensor (TMA)      │
                    │  __shfl_xor_sync (warp reduce)   │
                    │  mbarrier.* (Hopper sync)        │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │     CUDA Graph Runtime           │
                    │  cudaGraphInstantiate() — once   │
                    │  cudaGraphLaunch() — every step  │
                    │  Shape-keyed graph cache         │
                    │  Dynamic batch: graph pool       │
                    └─────────────────────────────────┘
```

---

**Inference-First: The Critical Components**

**FP8 E4M3 Inference Pipeline (H100 only)**

H100 Tensor Cores support FP8 (E4M3 and E5M2 formats) at **2× the FP16 throughput** — 3.9 PFLOP/s for FP8 vs 1.9 PFLOP/s for FP16 on one H100 SXM5. To exploit this:
- Weights quantized offline to FP8 E4M3 with per-tensor or per-channel scaling factors stored alongside
- Activations quantized online via a fast FP8-cast kernel fused into the preceding elementwise op
- The GEMM kernel uses `wgmma.mma_async` with `.f8f8f32` accumulation
- Output dequantization fused into the post-GEMM activation kernel (no extra memory round-trip)

The whole-graph scheduler is essential here: it can fuse the dequantization → activation → requantization chain as a single elementwise kernel between two GEMMs. Per-kernel compilers (Triton, TensorRT plugins) typically materialize intermediate FP32 results, losing the fusion opportunity.

**PagedAttention KV-Cache**

In production LLM serving, different requests in a batch have different KV-cache lengths (some are on token 10, others on token 8000). PagedAttention (the key innovation in vLLM) solves this by managing KV-cache in fixed-size **pages** (blocks of, e.g., 16 tokens), allocating pages on demand rather than pre-allocating a maximum-length buffer per request.

Integration into the framework:
- A KV-cache allocator maintains a pool of fixed-size page buffers in VRAM
- The scheduler knows which pages belong to which request (a block table per request)
- The attention kernel template takes the block table as an input and handles non-contiguous KV-cache reads internally — this is the FlashDecoding variant with paged memory access

**FlashDecoding for Long-Context Decode**

Standard FlashAttention is optimized for the **prefill** phase (processing the full prompt, compute-bound). During **decode** (generating one token at a time with a long KV-cache), the kernel is memory-bandwidth-bound and needs a different tile scheduling strategy:
- FlashDecoding splits the KV sequence across thread blocks, with each block handling a sub-range of the KV dimension
- Partial softmax results are computed in parallel and then reduced across blocks
- On H100 with a 32K-token KV-cache: FlashDecoding is ~5× faster than naive FlashAttention for decode

**CUDA Graphs with a Shape-Keyed Pool**

Inference serving has variable batch sizes (1 to max_batch). Rather than one graph, maintain a pool of pre-compiled graphs keyed by batch size:
```
graph_pool = {1: cuda_graph_1, 4: cuda_graph_4, 8: cuda_graph_8, ...}
```
For dynamic batch sizes, round up to the nearest compiled size and pad with dummy tokens (a masked attention pass on padding). This is the strategy used in vLLM's CUDA Graph mode and TRT-LLM. The framework should manage this pool automatically.

**L2 KV-Cache Persistence for Decode**

In decode, the KV-cache is read every step and rarely written (only the new token appends). Mark KV-cache buffers with `cudaAccessPropertyPersisting` so the most recently accessed KV pages stay in the H100's 50 MB L2 cache across kernel launches. For short-to-medium contexts (up to ~2048 tokens), this can eliminate repeated global memory fetches entirely.

**2:4 Structured Sparsity for Weight-Bound Layers**

For small batch decode where the GEMM is weight-bandwidth-bound (not compute-bound), 2:4 sparsity halves the weight transfer volume and doubles effective bandwidth, providing a real throughput gain. The framework handles the offline pruning + cuSPARSELt compression as a model export step.

---

**Training on NVIDIA: The Focused Win Condition**

Training is secondary but serious. The target is **transformer training** (LLMs, vision transformers) — not CNNs, not arbitrary models. This is where the structural advantage holds:

* **FlashAttention-3 forward + backward**: The attention layer is the throughput bottleneck for long-sequence training. FA-3 on H100 achieves ~75% of theoretical FLOP/s. The framework substitutes FA-3 when it detects the attention pattern in the UOp graph.
* **BF16 mixed precision**: BF16 Tensor Core GEMMs for forward/backward passes, FP32 master weights for the optimizer. The scheduler manages the cast ops and ensures they fuse with surrounding elementwise ops.
* **FP8 training (experimental, Blackwell-native on B100/B200)**: FP8 forward + FP8 backward with BF16 master weights and gradient scaling. NVIDIA Transformer Engine does this today; the framework should provide the same capability.
* **Gradient checkpointing integration**: For very long sequences, re-compute activations during backward instead of storing them. The scheduler marks checkpointed regions and re-schedules the forward subgraph during backward.
* **NCCL all-reduce hooks**: After each backward pass, all-reduce gradients across GPUs before the optimizer step. Integrated as a scheduled barrier in the UOp graph rather than an ad-hoc synchronization point.

**What training explicitly does NOT target**: ResNet-style CNN training, arbitrary op graphs, models with complex dynamic control flow. The cuDNN conv moat is real and not worth fighting. The transformer stack is the market.

---

**Framework vs. Ecosystem: Where to Position**

| Layer | Tool | This Framework's Role |
|---|---|---|
| Kernel language | Triton, CUDA C++ | Consumes kernel templates; does not replace |
| Inference engine | TensorRT, TRT-LLM | Replaces for transformer workloads — keeps hackability |
| LLM serving | vLLM, SGLang | Provides the execution backend (replaces PyTorch eager in vLLM) |
| Training framework | PyTorch + torch.compile | Competes on transformer training; yields on CNN training |
| Quantization | NVIDIA Modelopt, AutoGPTQ | Provides the FP8/INT8/2:4 quantization export pipeline |

**Honest Performance Ceiling (Inference-First)**

| Workload | Achievable vs. Best-in-Class |
|---|---|
| LLM prefill (H100, BF16, long prompt) | Within 5% of FA-3 throughput — FA-3 template sets the ceiling |
| LLM decode (H100, FP8, large batch) | **Can beat vLLM** — CUDA Graph + L2 persistence + paged FA-3 decode with FP8 weights |
| LLM decode (H100, FP8, small batch, long KV) | **Can beat TRT-LLM** — better cross-layer fusion, FP8 with dequant fusion |
| CNN inference (any model) | Cannot beat TensorRT — do not compete here |
| Transformer training (H100, BF16, long seq) | Within 10% of optimized PyTorch+FA-3 — competitive |
| CNN training | Do not enter this market |

---

### 4.4 Resources

* **CUTLASS 3.x (NVIDIA GitHub):** Production C++ templates for WGMMA, TMA, warp specialization, and persistent kernels on Hopper. The authoritative reference for what NVIDIA-optimal GEMM looks like at the CUDA level.
* **FlashAttention-3 paper and blog (Tri Dao, 2024):** Detailed walkthrough of how TMA + WGMMA + warp specialization achieves 75% of H100 theoretical throughput for attention.
* **Tawa: Automatic Warp Specialization (arXiv 2510.14719):** Adds automatic warp specialization on top of Triton IR, achieving up to 96% of FlashAttention-3 throughput and 1.21× over unspecialized Triton. Shows exactly where Triton's SPMT model falls short.
* **NVIDIA Hopper Architecture In-Depth (developer.nvidia.com):** Official deep-dive on TMA, WGMMA, warp specialization, and the Hopper programming model.
* **CUDA Programming Guide: L2 Cache Control:** How to use `cudaAccessPropertyPersisting` for KV-cache optimization.
* **tinygrad UOp IR and Scheduler:** `tinygrad/codegen/`, `tinygrad/engine/schedule.py` — the components to fork and extend.
* **NCCL Source Code:** How production multi-GPU collective communication handles NVLink topology detection and algorithm selection.
* **"Can tinygrad win?" — geohot's blog (July 2025):** Honest internal assessment of where tinygrad is and what the competitive path looks like.

---

### 4.5 Projects

Projects are ordered by priority: inference first, training second.

**Inference Projects**

* **Decode Throughput Benchmark — Establish the Baseline:** Run LLaMA-2-7B or similar on PyTorch eager, vLLM, and TensorRT-LLM on a single H100. Profile with Nsight Systems: measure time-per-token, GPU utilization, memory bandwidth utilization, and CPU dispatch overhead between kernel launches. This establishes exactly what you need to beat and where the gaps are. Write a report with roofline analysis for each system.

* **CUDA Graph Pool for Variable-Batch Inference:** Implement a shape-keyed CUDA Graph pool in Python (or fork tinygrad's JIT). Pre-compile graphs for batch sizes [1, 2, 4, 8, 16, 32]. For each inference call, select the nearest compiled batch size and pad. Measure the CPU overhead reduction vs. eager tinygrad and vs. eager PyTorch on the same transformer decode step.

* **FlashAttention Pattern Dispatch:** Fork tinygrad. Add a pattern recognizer to the scheduler that detects `softmax(Q @ K.T / sqrt(d)) @ V` in the UOp graph. Substitute a pre-written FlashAttention-2 CUDA kernel (use the reference implementation from Tri Dao's repo). Verify numerical correctness against the naive tinygrad result. Benchmark prefill throughput (tokens/sec) for sequence lengths 512, 2048, 8192.

* **FP8 Quantized GEMM — Dequant Fusion:** Write a CUDA kernel for FP8 E4M3 matrix multiply that fuses output dequantization and a ReLU/SiLU activation into the same kernel (no intermediate FP32 buffer). Compare memory traffic vs. an unfused version using Nsight Compute's memory throughput counters. Measure on a realistic FFN layer size (e.g., 4096×4096 weight matrix).

* **L2 KV-Cache Persistence Experiment:** Implement autoregressive transformer decode in CUDA. Variant A: default memory policy. Variant B: KV-cache buffers marked with `cudaAccessPropertyPersisting`. Measure tokens/sec and L2 hit rate (Nsight Compute) for context lengths 512, 2048, 4096, 8192. Determine the crossover point where the KV-cache exceeds L2 capacity and the optimization stops helping. Report as a graph.

* **PagedAttention Memory Manager:** Implement a KV-cache page allocator in C++/CUDA: fixed-size pages (e.g., 16 tokens × head_dim × num_heads), a free-list allocator, and a block table per request. Write an attention kernel that accepts a block table and handles non-contiguous KV reads. Test with a batch of 8 requests with randomly varying KV-cache lengths.

**Training Projects**

* **Transformer Training Benchmark — Where tinygrad Loses:** Run GPT-2 small training for 100 steps using both tinygrad and PyTorch with `torch.compile`. Profile with Nsight Systems. Identify specifically which ops account for the performance gap: is it the attention kernel? The GEMM? The elementwise chain? The optimizer step? This diagnostic work is the prerequisite for knowing what kernel templates to build first.

* **Tensor Core Lowering Pass:** Fork tinygrad. Add a pattern matcher in the scheduler that detects a matmul-shaped reduce-of-multiply in the UOp graph and annotates it with `TENSOR_CORE`. Write a PTX emitter that for this pattern emits `mma.sync.aligned.m16n8k16.row.col.bf16.bf16.f32.f32` instead of the generic loop. Benchmark the generated kernel vs. the generic tinygrad output on a 4096×4096 BF16 GEMM. Use Nsight Compute to verify tensor core utilization (should read ~90%+, not 0%).

* **FlashAttention-3 Training Integration (Advanced):** Integrate FlashAttention-3's forward and backward CUDA kernels into the framework as a pattern-dispatched template. The backward kernel is the harder part — it requires the log-sum-exp saved from the forward pass. Verify gradient correctness with a finite-difference check. Measure end-to-end training throughput on a 4096-sequence LLaMA attention layer vs. PyTorch SDPA.

* **Framework Architecture Design Document:** Write a 6-page technical design document. Section 1: the inference serving architecture (graph pool, paged KV-cache, FP8 pipeline). Section 2: the training architecture (gradient graph extension, BF16 mixed precision, NCCL hooks). Section 3: the UOp extension taxonomy (which new UOp types and annotations, and their lowering rules). Section 4: build prioritization — which components to build first based on the benchmark findings from the projects above. This document is the roadmap for actually building the framework.

