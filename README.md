<div align="center">

# AI Hardware Engineer Roadmap

**A structured, hands-on learning roadmap from digital design fundamentals to AI chip design**

[![GitHub stars](https://img.shields.io/github/stars/ai-hpc/ai-hardware-engineer-roadmap?style=flat-square)](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ai-hpc/ai-hardware-engineer-roadmap?style=flat-square)](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/network/members)

*Build real AI systems on real hardware — from first logic gate to custom accelerator silicon*

</div>

---

## Who Is This For?

- **EE/ECE students** who want to move from digital design into AI hardware
- **Software ML engineers** who want to understand the hardware their models run on
- **Embedded engineers** who want to add AI capabilities to their systems
- **Career changers** targeting roles in AI accelerator design, edge AI, or autonomous systems

You do **not** need prior AI/ML experience. The roadmap teaches AI fundamentals (Phase 4) after you have the hardware foundation to understand why they matter.

---

## What Is This?

A **5-phase self-study curriculum** for engineers who want to understand AI systems **from the hardware up**. You will not just train models — you will build the digital logic, program the FPGAs, optimize the inference engines, and design the accelerator architectures that make AI run in the real world.

Every phase answers one question: ***What does the hardware need to do to make AI work, and how do we build it?***

- **Phase 1:** What are the digital building blocks? *(and why do AI workloads stress them)*
- **Phase 2:** How do we build complete hardware systems? *(SoC platforms that AI runs on)*
- **Phase 3:** How do we make hardware fast? *(acceleration techniques for neural network operations)*
- **Phase 4:** What is AI, and how does it run on hardware? *(neural networks, edge deployment, optimization)*
- **Phase 5:** Where do I specialize? *(autonomous driving, AI chip design, HPC, robotics, security)*

**Estimated timeline:** ~2.5–5 years (flexible based on your pace and goals)

---

## Prerequisites

- **Math:** Comfortable with algebra and basic calculus (derivatives, matrix operations)
- **Programming:** Working knowledge of at least one language (C preferred; Python acceptable)
- **Hardware:** No prior hardware experience required — Phase 1 starts from scratch
- **Equipment:** Access to a computer running Linux (or WSL). FPGA dev boards recommended starting in Phase 2 (Basys 3 or Arty A7 for Phases 1–2; Zynq board for Phases 2–3)

---

## How to Use This Roadmap

1. **Phases 1–3 are sequential.** Each builds on the last. Do not skip.
2. **Phase 4 is the bridge.** This is where hardware meets AI. If you already have hardware experience, you can start here and backfill Phases 1–3 as needed.
3. **Phase 5 tracks are independent.** Choose based on your career goals. You can pursue multiple tracks in parallel.
4. **Each topic links to a detailed guide** with resources, projects, and hands-on exercises.
5. **Estimated pace:** The timeline assumes part-time self-study (~10–15 hours/week). Full-time learners can move significantly faster.

---

## Learning Path Overview

| Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|:--------|:--------|:--------|:--------|:--------|
| **Digital Foundations** | **Hardware Platforms** | **Acceleration** | **AI & Edge Deployment** | **Specialization Tracks** |
| 6–12 mo | 6–12 mo | 6–12 mo | 6–12 mo | Ongoing |
| Logic, Verilog, Embedded C, Linux | Vivado, Zynq SoC, Embedded Linux, Protocols | Timing, HLS, OpenCL, Computer Vision | Neural Networks, Jetson, TensorRT, Sensor Fusion, ROS2 | Autonomous Driving, AI Chips, HPC, Robotics, Security |

---

## Table of Contents

- [Phase 1: Digital Foundations](#-phase-1-digital-foundations-612-months)
- [Phase 2: Hardware Platforms & SoC Design](#-phase-2-hardware-platforms--soc-design-612-months)
- [Phase 3: Hardware Acceleration](#-phase-3-hardware-acceleration-612-months)
- [Phase 4: AI Fundamentals & Edge Deployment](#-phase-4-ai-fundamentals--edge-deployment-612-months)
- [Phase 5: Specialization Tracks](#-phase-5-specialization-tracks-ongoing)
- [Career Paths](#career-paths)
- [Academic References](#academic-references)

---

## ▸ Phase 1: Digital Foundations (6–12 months)

| Topic | Key Skills | AI Connection |
|-------|------------|---------------|
| [**Digital Design Fundamentals**](Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20Fundamentals/Guide.md) | Number systems, Boolean algebra, combinational/sequential logic, memory (SRAM, DRAM, ROM) | *MAC units, memory bandwidth, and data types (INT8, FP16) that power AI inference start here* |
| [**Hardware Description Languages**](Phase%201%20-%20Foundational%20Knowledge/2.%20Hardware%20Description%20Languages%20(HDLs)/Guide.md) | Verilog syntax, behavioral/dataflow/structural modeling, testbenches, synthesis | *The language you will use to design AI accelerator datapaths* |
| [**Embedded Systems Basics**](Phase%201%20-%20Foundational%20Knowledge/3.%20Embedded%20Systems%20Basics/Guide.md) | Microcontroller architecture, C for embedded, RTOS concepts | *TinyML runs on microcontrollers; understanding hardware constraints is essential* |
| [**Linux Fundamentals**](Phase%201%20-%20Foundational%20Knowledge/4.%20Linux%20Fundamentals/Guide.md) | Shell, scripting, permissions, networking | *Every AI development environment and deployment target runs Linux* |

**Projects:** Calculator on breadboard, FPGA digital clock, traffic light controller, UART module, basic RISC-V core

---

## ▸ Phase 2: Hardware Platforms & SoC Design (6–12 months)

| Topic | Key Skills | AI Connection |
|-------|------------|---------------|
| [**Xilinx FPGA Development**](Phase%202%20-%20Xilinx%20and%20Embedded%20Systems/1.%20Xilinx%20FPGA%20Development/Guide.md) | Vivado flow, IP cores, block design, timing closure, ILA/VIO debugging | *FPGAs are the prototyping platform for AI accelerators (FINN, Vitis AI)* |
| [**Zynq UltraScale+ MPSoC**](Phase%202%20-%20Xilinx%20and%20Embedded%20Systems/2.%20Zynq%20UltraScale%2B%20MPSoC/Guide.md) | PS/PL integration, embedded Linux on Zynq, device drivers | *Heterogeneous SoCs like Zynq are the template for AI chips (CPU + accelerator)* |
| [**Embedded Linux**](Phase%202%20-%20Xilinx%20and%20Embedded%20Systems/3.%20Embedded%20Linux/Guide.md) | Yocto, PetaLinux, kernel config, root filesystem | *Jetson, Qualcomm AI, and all edge AI platforms run embedded Linux* |
| [**Communication Protocols**](Phase%202%20-%20Xilinx%20and%20Embedded%20Systems/4.%20Communication%20Protocols%20(SPI%2C%20UART%2C%20I2C%2C%20CAN)/Guide.md) | SPI, UART, I2C, CAN — specs, drivers, hardware implementation | *Sensor interfaces for cameras, LiDAR, IMU — the input pipeline for AI perception* |

**Projects:** High-speed data acquisition, custom protocol, video processing pipeline, motor control with UI, NAS device

---

## ▸ Phase 3: Hardware Acceleration (6–12 months)

| Topic | Key Skills | AI Connection |
|-------|------------|---------------|
| [**Advanced FPGA Design**](Phase%203%20-%20Advanced%20FPGA%20and%20Acceleration/1.%20Advanced%20FPGA%20Design/Guide.md) | CDC, floorplanning, power optimization, partial reconfiguration | *Production FPGA accelerators for AI require timing closure, power budgets, and reconfiguration* |
| [**High-Level Synthesis (HLS)**](Phase%203%20-%20Advanced%20FPGA%20and%20Acceleration/2.%20High-Level%20Synthesis%20(HLS)/Guide.md) | C/C++ to RTL, dataflow, loop unrolling, pipelining | *HLS is how you build CNN accelerators (conv2d, matmul) on FPGAs without writing RTL by hand* |
| [**OpenCL**](Phase%203%20-%20Advanced%20FPGA%20and%20Acceleration/3.%20OpenCL/Guide.md) | Kernels, work-groups, heterogeneous computing (CPU/GPU/FPGA) | *The programming model for deploying AI workloads across different hardware targets* |
| [**Computer Vision**](Phase%203%20-%20Advanced%20FPGA%20and%20Acceleration/4.%20Computer%20Vision/Guide.md) | Image processing, object detection, OpenCV | *The primary AI workload you will deploy on hardware: perception from pixels* |

**Projects:** Matrix multiply accelerator, convolution engine, image processing pipeline, neural network acceleration, CPU vs GPU vs FPGA benchmarking

---

## ▸ Phase 4: AI Fundamentals & Edge Deployment (6–12 months)

> *This is where hardware meets AI. Start with neural network fundamentals to understand what the hardware needs to compute, then deploy and optimize on real edge devices.*

| Topic | Key Skills | Projects |
|-------|------------|----------|
| [**AI Fundamentals**](Phase%204%20-%20Nvidia%20Jetson%20and%20Edge%20AI/2.%20AI%20Fundamentals%20-%20Neural%20Networks%20and%20Edge%20AI/Guide.md) | Neural networks, backpropagation, CNNs, tinygrad, PyTorch | micrograd implementation, CNN from scratch, tinygrad internals |
| [**Nvidia Jetson Platform**](Phase%204%20-%20Nvidia%20Jetson%20and%20Edge%20AI/1.%20Nvidia%20Jetson%20Platform/Guide.md) | Jetson Orin Nano, JetPack, L4T, CUDA, Nsight | Real-time object detection, custom model deployment, autonomous robot |
| [**Edge AI Optimization**](Phase%204%20-%20Nvidia%20Jetson%20and%20Edge%20AI/3.%20Edge%20AI%20Optimization/Guide.md) | Quantization, pruning, TensorRT, CUDA kernels | Optimized model on Orin Nano, video analytics, low-power AI pipeline |
| [**Sensor Fusion**](Phase%204%20-%20Nvidia%20Jetson%20and%20Edge%20AI/4.%20Sensor%20Fusion/Guide.md) | Camera + LiDAR + IMU, Kalman filtering, BEVFusion | Navigation robot, drone flight control, 3D mapping |
| [**ROS2**](Phase%204%20-%20Nvidia%20Jetson%20and%20Edge%20AI/5.%20ROS2/Guide.md) | ROS 2, DDS, nodes, topics, multi-robot systems | Robot navigation, multi-robot coordination, edge deployment |

---

## ▸ Phase 5: Specialization Tracks (Ongoing)

> *Choose one or more tracks based on your career goals. All tracks assume completion of Phases 1–4.*

### Track A: Autonomous Driving

**Prerequisites:** Phase 3 (Computer Vision), Phase 4 (Sensor Fusion, Edge AI Optimization)

[**Detailed Guide →**](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/4.%20Autonomous%20Driving/Guide.md)

Openpilot reference architecture — perception (camerad, modeld), planning, control, ADAS. tinygrad for on-device inference. Camera ISP pipelines, sensor calibration, BEV perception, end-to-end driving models.

### Track B: AI Chip Design

**Prerequisites:** Phase 3 (HLS, Advanced FPGA Design), Phase 4 (AI Fundamentals)

[**Detailed Guide →**](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20AI%20Chip%20Design/Guide.md)

Hardware-software co-design for AI accelerators. tinygrad as a reference ML framework — study how software maps to hardware. Systolic arrays, dataflow architectures, custom operator design. FPGA prototyping of accelerators, ASIC flow overview.

### Track C: HPC & GPU Infrastructure

**Prerequisites:** Phase 4 (CUDA from Jetson Platform)

[**Detailed Guide →**](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/1.%20HPC%20with%20Nvidia%20GPU/Guide.md)

Multi-GPU programming with NCCL, NVLink/NVSwitch interconnects. vGPU and KVM virtualization. InfiniBand, RDMA, GPUDirect for distributed training. GPU cluster architecture and scheduling.

### Track D: Robotics

**Prerequisites:** Phase 4 (ROS2, Sensor Fusion)

[**Detailed Guide →**](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/2.%20Robotics%20Application/Guide.md)

Advanced ROS 2 patterns, Nav2 navigation stack, MoveIt manipulation. Sensor fusion for autonomous robots. Motion planning algorithms. Industrial automation with ROS-Industrial. Builds on Phase 4 Sensor Fusion with robotics-specific applications.

### Track E: Embedded Security

**Prerequisites:** Phases 1–3 (Digital Design, FPGA)

[**Detailed Guide →**](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/3.%20Security%20in%20Embedded%20Systems/Guide.md)

Cryptography fundamentals and hardware implementations. Secure boot mechanisms. Side-channel attack resistance. FPGA bitstream security and IP protection.

---

## Career Paths

| Role | Primary Phases | Specialization Track |
|------|---------------|---------------------|
| AI Accelerator / Chip Design Engineer | 1–3, 4 (AI Fundamentals) | Track B: AI Chip Design |
| Edge AI / Embedded ML Engineer | 1–2, 4 | — |
| ADAS / Autonomous Driving Engineer | 1–2, 4 | Track A: Autonomous Driving |
| GPU / HPC Infrastructure Engineer | 4 | Track C: HPC |
| Robotics Engineer | 2, 4 | Track D: Robotics |
| Hardware Security Engineer | 1–3 | Track E: Embedded Security |

---

## Academic References

For those interested in formal programs or self-study aligned with top university curricula:

[**CMU Robotics & AI Courses**](CMU-Robotics-AI-Courses.md) — 07-280 AI/ML I schedule, B.S. Robotics curriculum, and course catalog. Useful for supplementing Phase 4 (AI Fundamentals) and Track D (Robotics).

---

<div align="center">

**Built for the AI hardware community** · [Star ⭐](https://github.com/ai-hpc/ai-hardware-engineer-roadmap) if you find this useful

</div>
