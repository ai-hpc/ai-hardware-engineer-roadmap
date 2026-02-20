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
