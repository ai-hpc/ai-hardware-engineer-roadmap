# Hardware sources and claim ledger

[Course guide](README.md) · [Design brief lecture](Lecture-01.md)

**Checked:** October 14, 2026. This ledger records primary sources used for hardware facts and architecture methods. “Verified source” means the cited document supports the stated claim; it does not mean this course independently measured the hardware. Mutable web pages and repository `main` branches require an archived copy or commit reference in a reproducible experiment.

## Commercial platform evidence

| ID | Primary source | What was checked | Evidence class and limitation |
|---|---|---|---|
| H01 | [NVIDIA Jetson Thor specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) | T5000: 2,070 sparse FP4 TFLOPS, 128 GB LPDDR5X, 273 GB/s, 40–130 W. T4000: 1,200 sparse FP4 TFLOPS, 64 GB, 273 GB/s, 40–70 W. | Manufacturer specifications. Peak arithmetic, advertised bandwidth, and module power range are different quantities; none establishes policy latency or complete robot power. |
| H02 | [Qualcomm Dragonwing IQ10 Robotics Reference Design](https://www.qualcomm.com/news/onq/2026/06/dragonwing-iq10-robotics-reference-design) | Up to 700 TOPS; heterogeneous compute and sensor/control integration. | Manufacturer announcement. Precision/sparsity conventions needed for a matched arithmetic comparison are absent from this overview. The page displays May 28, 2026 despite `/2026/06/` in its URL. |
| H03 | [GR00T hardware recommendations](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md) | The N1.7 one-camera, four-denoising-step rates reproduced in Lecture 01; explicit distinction between replanning and action execution. | Vendor-reported model benchmark. No independent reproduction in this course. Do not extrapolate to other camera counts, policies, or thermal conditions. |
| H04 | [GR00T deployment methods and timing table](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md) | Thor full-pipeline TensorRT: 80.4 ms; eager: 112.8 ms. Build defaults: BF16 and batch one. Orin eager: 354.0 ms / 2.8 Hz, conflicting with H03's 2.9 Hz. | Vendor-reported timings and implementation defaults. Defaults are not a complete experimental manifest. The discrepancy is unresolved; retain source identity with each result. |
| H05 | [NVIDIA IGX safety product brief](https://developer.download.nvidia.com/assets/igx/robotics-product-brief-igx-thor-safety-4473375.pdf) | FSI is inside the SoC with independent clocks/power rail; safety MCU is on the carrier board. Safety Extension Package connects these capabilities. | Manufacturer architecture documentation. Application-specific safety evidence and final certification remain separate from component features. |

The [IGX technical overview](https://developer.nvidia.com/blog/nvidia-igx-thor-powers-industrial-medical-and-robotics-edge-ai-applications/) describes configurations with different safety hardware. Use the product brief's explicit physical placement when distinguishing the SoC island, module, carrier, and complete platform. Do not infer that every Jetson configuration exposes the same safety support.

## Reusable architecture and estimation references

| ID | Primary source | Study this mechanism | What it does not prove |
|---|---|---|---|
| H06 | [Gemmini implementation](https://github.com/ucb-bar/gemmini) and [Gemmini paper](https://arxiv.org/abs/1911.09925) | Configurable systolic dataflows, banked scratchpad/accumulator memories, DMA, separate access/execute queues, and a software stack tied to generated parameters. | A generated array does not automatically support a complete VLA operator set, the proposed low-bit formats, or a particular Linux/safety integration. |
| H07 | [PULP Ara](https://github.com/pulp-platform/ara) | A programmable vector coprocessor implementing the RISC-V Vector Extension 1.0 alongside CVA6; use its kernels and RTL to study non-matmul execution. | A general vector ISA does not imply efficient support for every softmax, normalization, rearrangement, or low-precision tensor operation. Profile and map the required operations. |
| H08 | [Timeloop/Accelergy methodology](https://timeloop.csail.mit.edu/) | Joint exploration of hardware constraints and workload mappings; estimates based on working sets and technology-dependent operation costs. | An analytical optimum is not a routed chip or a validated end-to-end latency bound. Calibrate selected points with simulation, synthesis, and physical evidence. |
| H09 | [Accelergy energy/area reference tables](https://timeloop.csail.mit.edu/v4/parsing_and_intermediate_files/energy-and-area-reference-tables) and [original methodology paper](https://accelergy.mit.edu/paper.pdf) | Component action energy and area estimates from explicit component models and estimator plugins. | A tool name alone does not justify SRAM area, access energy, wire cost, leakage, or process assumptions. Record the plugin, version, technology, and coverage. |

These references are examples for mechanism study, not an instruction to combine their interfaces unchanged. Review license terms, generator configuration, software compatibility, and maintained revisions before making a prototype depend on them.

## Evidence rules for the design study

Use one of these labels beside every result:

| Label | Required supporting artifact |
|---|---|
| **Manufacturer specification** | Product/SKU, source, access date, units and stated conditions |
| **Externally reported benchmark** | Source, model/configuration, reported timing boundary, unresolved omissions |
| **Local measurement** | Revisions, commands, logs, inputs, power/clocks, repetitions and uncertainty |
| **Analytical estimate** | Equation, units, assumptions, sensitivity range and known exclusions |
| **Simulation result** | Simulator/model revision, configuration, validation/calibration evidence |
| **Exploration target** | Proposed threshold, motivation, gate that will accept or reject it |

The initial 20–40 W module, 16–32 GB DRAM, 16–64 MiB SRAM, 0.5–4B-policy and 10–30 Hz ranges are **exploration targets inherited from the project brief**. They are not specifications of an existing chip, feasibility results, or simultaneously guaranteed requirements.

## Open questions to carry into the architecture review

1. What sustainable bandwidth remains after camera ingestion, Linux activity, and competing accelerators? Measure traffic and arbitration before assigning the entire interface to the VLA.
2. Which action-head tensors are reusable within an evaluation, and which remain valid across observations? Trace dependencies; capacity alone does not establish cache correctness.
3. How much SRAM is available after banks, ports, ECC, spare rows, control logic and routing? Obtain technology-specific macros and physical estimates.
4. Which operations use the tensor engine, vector engine, or host? Include fallback copies and synchronization in the schedule.
5. Which memory-controller, PHY, interconnect, Linux-driver, and supervision IP is available under the project's integration constraints?
6. Does the predicted energy advantage survive at matched task quality and response time? Keep silicon, algorithm, and scheduling contributions separately attributable.

Before refreshing a number, compare source revisions and benchmark conditions. Replace the associated experiment manifest as well as the headline table; otherwise a newer-looking page can silently describe a different experiment.
