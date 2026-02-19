<div align="center">

# AI Hardware Engineer Roadmap

**A structured, hands-on learning roadmap from digital design fundamentals to AI chip design**

[![GitHub stars](https://img.shields.io/github/stars/ai-hpc/ai-hardware-engineer-roadmap?style=flat-square)](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ai-hpc/ai-hardware-engineer-roadmap?style=flat-square)](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/network/members)

*Digital Design → Verilog → Xilinx FPGA & Zynq → HLS/OpenCL → Jetson Edge AI → HPC, Robotics, Autonomous Driving, AI Chip Design*

</div>

---

## 📖 What is this?

A **5-phase self-study curriculum** for engineers who want to build expertise in AI hardware—from foundational digital logic to designing custom AI accelerators. Each phase includes detailed guides and hands-on projects to solidify your understanding.

**Estimated timeline:** ~2.5–5 years (flexible based on your pace and goals)

---

## 🗺️ Learning Path Overview

| Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|:--------|:--------|:--------|:--------|:--------|
| **Foundations** | **Xilinx & SoC** | **FPGA & HLS** | **Jetson Edge AI** | **Specialization** |
| 6–12 mo | 6–12 mo | 6–12 mo | 6–12 mo | Ongoing |
| Digital Design, Verilog, Embedded C, Linux | Vivado, Zynq, Linux, SPI/UART/I2C/CAN | Timing, HLS, OpenCL, Computer Vision | TensorRT, CUDA, Sensor Fusion, ROS2 | HPC, Robotics, Autonomous Driving, AI Chip Design, Security |

---

## 📚 Table of Contents

- [Phase 1: Foundational Knowledge](#-phase-1-foundational-knowledge-6-12-months)
- [Phase 2: Xilinx and Embedded Systems](#-phase-2-xilinx-and-embedded-systems-6-12-months)
- [Phase 3: Advanced FPGA and Acceleration](#-phase-3-advanced-fpga-and-acceleration-6-12-months)
- [Phase 4: Nvidia Jetson and Edge AI](#-phase-4-nvidia-jetson-and-edge-ai-6-12-months)
- [Phase 5: Advanced Topics and Specialization](#-phase-5-advanced-topics-and-specialization-ongoing)

---

## ▸ Phase 1: Foundational Knowledge (6-12 months)

> [📄 Detailed Guide →](Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20Fundamentals/Guide.md)

| Topic | Key Skills | Projects |
|-------|------------|----------|
| **Digital Design Fundamentals** | Number systems, Boolean algebra, combinational/sequential logic, memory (SRAM, DRAM, ROM) | Calculator on breadboard, FPGA digital clock, traffic light controller |
| **Hardware Description Languages** | Verilog syntax, behavioral/dataflow/structural modeling, testbenches, synthesis | UART module, basic RISC-V core, Pong/Snake on FPGA |
| **Embedded Systems Basics** | Microcontroller architecture, C for embedded, RTOS concepts | Temperature sensor + LCD, PWM motor control, RTOS scheduler |
| **Linux Fundamentals** | Shell, scripting, permissions, networking | Backup scripts, web server, network troubleshooting |

---

## ▸ Phase 2: Xilinx and Embedded Systems (6-12 months)

> [📄 Detailed Guide →](Phase%202%20-%20Xilinx%20and%20Embedded%20Systems/1.%20Xilinx%20FPGA%20Development/Guide.md)

| Topic | Key Skills | Projects |
|-------|------------|----------|
| **Xilinx FPGA Development** | Vivado flow, IP cores, block design, timing closure, ILA/VIO debugging | High-speed data acquisition, custom protocol, video processing pipeline |
| **Zynq UltraScale+ MPSoC** | PS/PL integration, embedded Linux on Zynq, device drivers | Real-time data logging, motor control + UI, NAS device |
| **Embedded Linux** | Yocto, PetaLinux, kernel config, root filesystem | Custom Linux distro, port application, secure embedded system |
| **Communication Protocols** | SPI, UART, I2C, CAN—specs, drivers, hardware implementation | Accelerometer/gyro (SPI), servo (I2C), CAN bus network |

---

## ▸ Phase 3: Advanced FPGA and Acceleration (6-12 months)

> [📄 Detailed Guide →](Phase%203%20-%20Advanced%20FPGA%20and%20Acceleration/1.%20Advanced%20FPGA%20Design/Guide.md)

| Topic | Key Skills | Projects |
|-------|------------|----------|
| **Advanced FPGA Design** | CDC, floorplanning, power optimization, partial reconfiguration | PCIe/Ethernet interface, DSP algorithm, reconfigurable system |
| **High-Level Synthesis (HLS)** | C/C++ to RTL, dataflow, loop unrolling, pipelining | Image filtering, crypto/compression accelerator, data pipeline |
| **OpenCL** | Kernels, work-groups, heterogeneous computing (CPU/GPU/FPGA) | FPGA acceleration, CPU vs GPU vs FPGA comparison, computer vision |
| **Computer Vision** | Image processing, object detection, OpenCV | FPGA image pipeline, object detection system, facial recognition |

---

## ▸ Phase 4: Nvidia Jetson and Edge AI (6-12 months)

> [📄 Detailed Guide →](Phase%204%20-%20Nvidia%20Jetson%20and%20Edge%20AI/1.%20Nvidia%20Jetson%20Platform/Guide.md)

| Topic | Key Skills | Projects |
|-------|------------|----------|
| **Nvidia Jetson Platform** | Jetson Orin Nano, JetPack, L4T, TensorFlow/PyTorch, CUDA, Nsight | Real-time object detection, custom model deployment, autonomous robot |
| **Edge AI Optimization** | Quantization, pruning, TensorRT, CUDA | Optimized model on Orin Nano, video analytics, low-power AI app |
| **Sensor Fusion** | Camera + LiDAR + IMU, Kalman filtering, ROS | Navigation robot, drone flight control, 3D mapping |
| **ROS2** | ROS 2, DDS, nodes, topics | Robot navigation, multi-robot systems, edge deployment |

---

## ▸ Phase 5: Advanced Topics and Specialization (Ongoing)

> [📄 Detailed Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/1.%20HPC%20with%20Nvidia%20GPU/Guide.md)

| Topic | Focus |
|-------|--------|
| **HPC with Nvidia GPU** | vGPU, KVM, NCCL, NVLink/NVSwitch, InfiniBand, RDMA, MPS, GPU clusters |
| **Robotics Application** | ROS/ROS 2, sensor fusion, motion planning, industrial automation |
| **Security in Embedded Systems** | Cryptography, secure boot, side-channel resistance, FPGA security |
| [**Autonomous Driving**](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/4.%20Autonomous%20Driving/Guide.md) | Openpilot reference—perception, planning, control, ADAS |
| [**AI Chip Design**](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20AI%20Chip%20Design/Guide.md) | tinygrad, ML frameworks, hardware-software co-design for AI accelerators |

---

<div align="center">

**Built for the AI hardware community** · [Star ⭐](https://github.com/ai-hpc/ai-hardware-engineer-roadmap) if you find this useful

</div>
