# Foundations, embedded systems, and interview review — 2026-10-14

Scope: **32 inventoried course-entry pages plus the OS demo README** (33 pages total) in Phases 1, 2, and 6, including phase hubs and lecture indexes. The extra page is `Phase 1 - Foundational Knowledge/3. Operating Systems/Lectures/demos/rt-demo/README.md`, excluded from the repository’s course-entry inventory. These are not 33 independent full courses. Every entry received a topic-specific handoff or a substantive correction; generic repeated assessment tails were replaced with concrete work that feeds the physical-AI SoC goal.

**Review depth:** full short hubs/indexes were read; long monolithic guides were reviewed by section inventory plus selected technical/code/lab passages. All OS lecture titles/headings were compared with the index; all 16 short Yocto lectures were read; other long lecture series were sampled at objectives, headings, labs, and implicated technical sections. This is an editorial/technical review with targeted execution, not a line-by-line audit or hardware certification of every lecture.

## Corrections with the greatest learning impact

- Rebuilt the OS hub from its actual 26 lecture titles; its prior index described a different syllabus.
- Fixed unsafe synchronization-free CUDA reduction, a factor-of-1000 roofline unit error, incorrect SIMD byte arithmetic, and misleading fixed-speedup claims.
- Distinguished Vera CPU from Rubin GPU, target-specific HIP wave/bank behavior, source-portability from device support, and oneTBB affinity hints from thread guarantees.
- Corrected FreeRTOS `heap_4`, Cortex-M assumptions, `volatile` synchronization language, coherent DMA cacheability, and persistent DMA ownership semantics.
- Replaced duplicated embedded-Linux expansion prose with a boot → image → driver → service → recovery sequence.
- Rewrote all eight inference interview Q&As: removed unsupported personal production anecdotes; corrected KV accounting, FlashAttention I/O, EAGLE-3 semantics, speculation economics, Q8_0 scale size, and TTFT definitions.
- Repaired the OS C demo’s missing `<sys/mman.h>` include and removed a compiler warning and leaked tool markup; documented that it does not generate a latency histogram.

## Entry-by-entry ledger

### 1. [Phase 1 - Foundational Knowledge/1. Digital Design and Hardware Description Languages/Digital Design Fundamentals/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20and%20Hardware%20Description%20Languages/Digital%20Design%20Fundamentals/Guide.md)

- **Read/reviewed:** All section headings and selected representation, quantization, memory, project, and assessment sections reviewed.
- **Change:** Corrected signed-representation and BF16 conversion claims; added an accumulator-width derivation.
- **Passed onward:** Representable-range proof and golden arithmetic vectors.
- **Source basis:** Primary-source group **logic** below; also the linked local course examples.

### 2. [Phase 1 - Foundational Knowledge/1. Digital Design and Hardware Description Languages/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20and%20Hardware%20Description%20Languages/Guide.md)

- **Read/reviewed:** Full module hub read.
- **Change:** Added digital-to-HDL integration with explicit backpressure semantics.
- **Passed onward:** Clock/reset and ready/valid contract with boundary-case waveforms.
- **Source basis:** Primary-source group **logic** below; also the linked local course examples.

### 3. [Phase 1 - Foundational Knowledge/1. Digital Design and Hardware Description Languages/Hardware Description Languages (HDLs)/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20and%20Hardware%20Description%20Languages/Hardware%20Description%20Languages%20%28HDLs%29/Guide.md)

- **Read/reviewed:** All headings and selected HDL semantics, FIFO/FSM code, simulation, and synthesis sections reviewed.
- **Change:** Removed unsupported language-market claims; constrained FIFO depth, clocking, boundary behavior and memory inference.
- **Passed onward:** FIFO scoreboard, accepted-transaction trace, and arithmetic-block interface.
- **Source basis:** Primary-source group **logic** below; also the linked local course examples.

### 4. [Phase 1 - Foundational Knowledge/2. Computer Architecture and Hardware/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/2.%20Computer%20Architecture%20and%20Hardware/Guide.md)

- **Read/reviewed:** All section headings; memory/coherence, roofline, case-study labels, labs, and exit sections reviewed.
- **Change:** Corrected coherence versus collectives, roofline ceilings, precision assumptions, and quantization ratios.
- **Passed onward:** A shape-specific traffic/capacity worksheet and predicted limit.
- **Source basis:** Primary-source group **cuda** below; also the linked local course examples.

### 5. [Phase 1 - Foundational Knowledge/3. Operating Systems/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md)

- **Read/reviewed:** Full original index compared against titles/headings of all 26 lectures and eight notes; selected DMA/code sections reviewed.
- **Change:** Rebuilt mismatched syllabus; corrected coherent DMA/ownership in L15 and Note 07; removed demo markup
- **Passed onward:** Observation-to-action timeline and buffer ownership contract
- **Source basis:** Primary-source group **os** below; also the linked local course examples.

### 6. [Phase 1 - Foundational Knowledge/3. Operating Systems/Lectures/demos/rt-demo/README.md](../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/demos/rt-demo/README.md)

- **Read/reviewed:** Full demo README read; demo source inspected for CLI and measurement context.
- **Change:** Removed leaked generation/tool markup and clarified privilege and timing evidence.
- **Passed onward:** Baseline/load comparison including requested versus obtained policy.
- **Source basis:** Primary-source group **os** below; also the linked local course examples.

### 7. [Phase 1 - Foundational Knowledge/4. C++ and Parallel Computing/C++ and SIMD/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/C%2B%2B%20and%20SIMD/Guide.md)

- **Read/reviewed:** All headings plus lambda, SIMD, alignment, roofline, and measurement sections reviewed.
- **Change:** Corrected vector-add intensity, device-lambda capture generalization, and guaranteed-performance language.
- **Passed onward:** Scalar/SIMD parity corpus, vectorization report, and byte-counted timing.
- **Source basis:** Primary-source group **sycl** below; also the linked local course examples.

### 8. [Phase 1 - Foundational Knowledge/4. C++ and Parallel Computing/CUDA and SIMT/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/CUDA%20and%20SIMT/Guide.md)

- **Read/reviewed:** All headings; synchronization/reduction, memory tiling, capability tables and roofline examined.
- **Change:** Fixed unsafe final-warp reduction, roofline units, CPU/GPU naming, and unsupported fixed speedups.
- **Passed onward:** Verified kernel plus a profiler-backed explanation of one optimization.
- **Source basis:** Primary-source group **cuda** below; also the linked local course examples.

### 9. [Phase 1 - Foundational Knowledge/4. C++ and Parallel Computing/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md)

- **Read/reviewed:** Full overview read with execution-model and roadmap sections checked.
- **Change:** Removed universal GPU speedup and portability claims; aligned the sequence to reusable experiments.
- **Passed onward:** Shared benchmark inputs and a backend comparison with identical correctness checks.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 10. [Phase 1 - Foundational Knowledge/4. C++ and Parallel Computing/OpenCL and SYCL/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/OpenCL%20and%20SYCL/Guide.md)

- **Read/reviewed:** All headings and host-code/USM/profiling/portability sections reviewed.
- **Change:** Clarified USM support, migration, and asynchronous ordering boundaries.
- **Passed onward:** Dependency graph, capability checks, and a portable operator contract.
- **Source basis:** Primary-source group **sycl** below; also the linked local course examples.

### 11. [Phase 1 - Foundational Knowledge/4. C++ and Parallel Computing/OpenMP and OneTBB/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/OpenMP%20and%20OneTBB/Guide.md)

- **Read/reviewed:** All headings; partitioners, reductions, flow-graph/backpressure, and performance sections sampled.
- **Change:** Corrected partitioner guarantees and added end-to-end concurrency reasoning.
- **Passed onward:** A bounded pipeline timeline and worker-count/latency comparison.
- **Source basis:** Primary-source group **cpu** below; also the linked local course examples.

### 12. [Phase 1 - Foundational Knowledge/4. C++ and Parallel Computing/ROCm and HIP/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/ROCm%20and%20HIP/Guide.md)

- **Read/reviewed:** All headings and CUDA mapping, HIPIFY workflow, memory, and projects sections reviewed.
- **Change:** Removed invented HIPIFY accuracy, unsupported recursive command, universal wave/bank count, container-emulation and memory-speed claims.
- **Passed onward:** Portability matrix and matched CUDA/HIP correctness/performance results.
- **Source basis:** Primary-source group **hip** below; also the linked local course examples.

### 13. [Phase 1 - Foundational Knowledge/Guide.md](../Phase%201%20-%20Foundational%20Knowledge/Guide.md)

- **Read/reviewed:** Full phase hub read.
- **Change:** Added a single operator thread through four modules instead of disconnected artifacts.
- **Passed onward:** Operator specification, golden vectors, transfer contract, and measured baseline.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 14. [Phase 2 - Embedded Systems/1. Schematic Capture and PCB Design/Guide.md](../Phase%202%20-%20Embedded%20Systems/1.%20Schematic%20Capture%20and%20PCB%20Design/Guide.md)

- **Read/reviewed:** Full guide read.
- **Change:** Corrected ODB++ spelling and added schematic-to-driver traceability.
- **Passed onward:** Reviewed schematic/pin map, power budget, and ordered bring-up observations.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 15. [Phase 2 - Embedded Systems/2. Embedded Software/Espressif/Education/Guide.md](../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/Espressif/Education/Guide.md)

- **Read/reviewed:** Full guide; five associated lecture objectives and labs read.
- **Change:** Connected resource selection to measurable device requirements.
- **Passed onward:** A targeted study plan with board-supported examples and acceptance evidence.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 16. [Phase 2 - Embedded Systems/2. Embedded Software/Espressif/Education/Lecture/README.md](../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/Espressif/Education/Lecture/README.md)

- **Read/reviewed:** Full index and all five lecture headings/labs reviewed.
- **Change:** Added a concrete sequence and distinction between study planning and implemented evidence.
- **Passed onward:** Board baseline and a single solution-track plan feeding an external consumer.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 17. [Phase 2 - Embedded Systems/2. Embedded Software/Espressif/Guide.md](../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/Espressif/Guide.md)

- **Read/reviewed:** Full hub read; five lecture introductions/headings/labs reviewed.
- **Change:** Replaced generic identity/assessment with concrete MCU responsibility and framework handoff.
- **Passed onward:** Board/SDK manifest and a versioned sensor or wireless message contract.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 18. [Phase 2 - Embedded Systems/2. Embedded Software/Espressif/Lecture/README.md](../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/Espressif/Lecture/README.md)

- **Read/reviewed:** Full index; all five lecture introductions, headings, and lab sections read.
- **Change:** Upgraded reflection-only lecture sequence to observable staged outcomes.
- **Passed onward:** Five staged evidence items culminating in a rebuildable firmware project.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 19. [Phase 2 - Embedded Systems/2. Embedded Software/Guide.md](../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/Guide.md)

- **Read/reviewed:** Full guide read; niche corrections checked against primary documentation.
- **Change:** Corrected heap_4, Cortex-M generalizations, volatile, A/B update assumptions, and unsupported panda RTOS attribution.
- **Passed onward:** Sensor/control message schema and fault-injection trace.
- **Source basis:** Primary-source group **firmware** below; also the linked local course examples.

### 20. [Phase 2 - Embedded Systems/2. Embedded Software/IoT/Guide.md](../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/IoT/Guide.md)

- **Read/reviewed:** Full guide read.
- **Change:** Connected protocol study to robot telemetry requirements and deadline limits.
- **Passed onward:** Protocol choice with a common data/freshness schema.
- **Source basis:** Primary-source group **network** below; also the linked local course examples.

### 21. [Phase 2 - Embedded Systems/2. Embedded Software/IoT/OpenThread/Guide.md](../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/IoT/OpenThread/Guide.md)

- **Read/reviewed:** All headings; protocol stack, roles, co-processor and projects sections read.
- **Change:** Replaced unlicensed hotlinked topology with original local art; strengthened reachability versus application validation.
- **Passed onward:** Role/route trace plus measured recovery to fresh application data.
- **Source basis:** Primary-source group **network** below; also the linked local course examples.

### 22. [Phase 2 - Embedded Systems/2. Embedded Software/IoT/Zigbee/Guide.md](../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/IoT/Zigbee/Guide.md)

- **Read/reviewed:** Full guide; all five lecture headings/labs and stack analogy section reviewed.
- **Change:** Added a specific data-model handoff beyond generic network lab advice.
- **Passed onward:** An endpoint/cluster-to-host mapping with restart and stale-report evidence.
- **Source basis:** Primary-source group **network** below; also the linked local course examples.

### 23. [Phase 2 - Embedded Systems/2. Embedded Software/IoT/Zigbee/Lecture/README.md](../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/IoT/Zigbee/Lecture/README.md)

- **Read/reviewed:** Full index and all five lecture lab sections read.
- **Change:** Connected descriptive/analogy lectures to protocol evidence.
- **Passed onward:** Decoded transaction and architecture placement for the same device.
- **Source basis:** Primary-source group **network** below; also the linked local course examples.

### 24. [Phase 2 - Embedded Systems/3. Embedded Linux/Guide.md](../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Guide.md)

- **Read/reviewed:** Full original guide read and rewritten.
- **Change:** Removed duplicated generic prose and arbitrary multi-year timelines; clarified Buildroot/Yocto and built a six-stage progression
- **Passed onward:** Boot/image/driver/service/recovery contract
- **Source basis:** Primary-source group **linux** below; also the linked local course examples.

### 25. [Phase 2 - Embedded Systems/3. Embedded Linux/Jetson ESP-Hosted Host Code/Guide.md](../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Jetson%20ESP-Hosted%20Host%20Code/Guide.md)

- **Read/reviewed:** Full guide/index; all lecture introductions/headings/labs; bring-up sections sampled.
- **Change:** Qualified old hardware validation and GPIO/interface-name portability; added accelerator transfer exercise.
- **Passed onward:** Pinned driver path, buffer ownership table, and reset/recovery evidence.
- **Source basis:** Primary-source group **linux** below; also the linked local course examples.

### 26. [Phase 2 - Embedded Systems/3. Embedded Linux/Jetson ESP-Hosted Host Code/Lecture/README.md](../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Jetson%20ESP-Hosted%20Host%20Code/Lecture/README.md)

- **Read/reviewed:** Full index and all five lecture labs read.
- **Change:** Turned shared-transport description into staged, testable subsystem outcomes.
- **Passed onward:** A board-to-subsystem evidence chain with separate Wi-Fi/Bluetooth results.
- **Source basis:** Primary-source group **linux** below; also the linked local course examples.

### 27. [Phase 2 - Embedded Systems/3. Embedded Linux/Yocto/Guide.md](../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Yocto/Guide.md)

- **Read/reviewed:** Full guide/index; all 16 short lectures read; worked-camera example sampled.
- **Change:** Strengthened reproducibility claims and connected the image to the inference ABI.
- **Passed onward:** Pinned product layer, image/SDK manifest, and runtime smoke test.
- **Source basis:** Primary-source group **linux** below; also the linked local course examples.

### 28. [Phase 2 - Embedded Systems/3. Embedded Linux/Yocto/Lecture/README.md](../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Yocto/Lecture/README.md)

- **Read/reviewed:** Full index and all 16 short lectures read.
- **Change:** Converted independent small labs into one cumulative image project.
- **Passed onward:** A continuous product-layer history ending in an image plus matching SDK.
- **Source basis:** Primary-source group **linux** below; also the linked local course examples.

### 29. [Phase 2 - Embedded Systems/4. Product Design/Guide.md](../Phase%202%20-%20Embedded%20Systems/4.%20Product%20Design/Guide.md)

- **Read/reviewed:** Full guide; all four lecture objectives/labs and HomePod analysis sections sampled.
- **Change:** Extended speaker-centered design into the user’s robotics/SoC goal without replacing the case study.
- **Passed onward:** A measurable product brief feeding SoC requirements and system tests.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 30. [Phase 2 - Embedded Systems/4. Product Design/Lecture/README.md](../Phase%202%20-%20Embedded%20Systems/4.%20Product%20Design/Lecture/README.md)

- **Read/reviewed:** Full index plus all four lecture lab sections read.
- **Change:** Clarified interpretation versus evidence and added robotics transfer.
- **Passed onward:** Case-study hypotheses translated into testable product and hardware requirements.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 31. [Phase 2 - Embedded Systems/Guide.md](../Phase%202%20-%20Embedded%20Systems/Guide.md)

- **Read/reviewed:** Full phase hub read.
- **Change:** Added a coherent sensor-to-action phase project and original explanatory figure.
- **Passed onward:** One end-to-end interface and freshness contract shared by all modules.
- **Source basis:** Local course content and original engineering exercise; no new version-specific product claim introduced.

### 32. [Phase 6 - Interview Preparation/1. MLSys Engineer/README.md](../Phase%206%20-%20Interview%20Preparation/1.%20MLSys%20Engineer/README.md)

- **Read/reviewed:** Full hub and all eight Q&As reviewed and rewritten.
- **Change:** Rebuilt inference explanations with primary sources, correct equations, and no invented first-person wins.
- **Passed onward:** An equation sheet, measured counterexample, and honest experience inventory.
- **Source basis:** Primary-source group **interview** below; also the linked local course examples.

### 33. [Phase 6 - Interview Preparation/README.md](../Phase%206%20-%20Interview%20Preparation/README.md)

- **Read/reviewed:** Full phase hub read.
- **Change:** Removed fabricated benchmark prompts and speculative company-interview format assertions.
- **Passed onward:** A traceable design defense grounded in the learner’s own evidence.
- **Source basis:** Primary-source group **interview** below; also the linked local course examples.

## Primary sources checked

- **logic:** [Arm Cortex-M0 ISA](https://support.arm.com/documentation/ddi0432/c/programmers-model/instruction-set-summary?lang=en); [NVIDIA BF16 math](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html); [AMD synchronous FIFO semantics](https://docs.amd.com/r/en-US/ug573-ultrascale-memory-resources/Common-Clock/Single-Clock-FIFO).
- **os:** [Linux DMA API](https://docs.kernel.org/core-api/dma-api-howto.html); [EEVDF](https://docs.kernel.org/scheduler/sched-eevdf.html).
- **cuda:** [NVIDIA independent thread scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html); [Vera Rubin components](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/).
- **cpu:** [oneTBB partitioner specification](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.1-rev-1/elements/onetbb/source/algorithms/partitioners/static_partitioner).
- **hip:** [HIP portability workflow](https://rocm.blogs.amd.com/software-tools-optimization/hipify/README.html); [HIPIFY supported APIs](https://rocmdocs.amd.com/projects/HIPIFY/en/latest/reference/hip_roc_supported_apis.html); [AMD LDS bank mapping](https://rocm.docs.amd.com/projects/composable_kernel/en/7.13.0-preview/conceptual/ck_tile/hardware/lds_bank_conflicts.html).
- **sycl:** [Khronos USM concepts](https://github.khronos.org/SYCL_Reference/iface/usm_basic_concept.html).
- **firmware:** [FreeRTOS allocation](https://www.freertos.org/Documentation/02-Kernel/02-Kernel-features/09-Memory-management/01-Memory-management); [MCUboot design](https://docs.mcuboot.com/design.html).
- **linux:** [Buildroot manual](https://buildroot.org/downloads/manual/manual.html); [Yocto build history](https://docs.yoctoproject.org/current/dev-manual/build-quality.html).
- **network:** [OpenThread roles](https://openthread.io/guides/thread-primer/node-roles-and-types); existing vendor references retained; no new hardware compatibility claims added.
- **interview:** [PagedAttention](https://arxiv.org/abs/2309.06180); [FlashAttention](https://arxiv.org/abs/2205.14135); [FlashAttention-2](https://arxiv.org/abs/2307.08691); [speculative sampling](https://arxiv.org/abs/2211.17192); [EAGLE-3](https://arxiv.org/abs/2503.01840); [ggml layouts](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h); [CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html).

## Original visual material

Two original diagrams have editable SVGs, PNG exports, a generator, and provenance/license notes under [foundations-embedded illustrations](../Assets/illustrations/foundations-embedded/README.md):

1. Observation data path versus runtime/control/wireless responsibilities; used in the Phase 2 hub.
2. Thread roles and application freshness; replaces an externally hotlinked image with unverified reuse terms.

The diagrams were rendered with CairoSVG and visually inspected. They represent teaching architectures rather than vendor-certified implementations.

## Validation and remaining limits

- Entry inventory checked: all 32 assigned inventoried course entries plus the extra OS demo README appear exactly once in this ledger.
- OS demo compiled with `gcc -Wall -Wextra -O2 -pthread`; default non-RT run completed with `Final counter = 10000`. Elevated real-time mode was not exercised.
- Editorial arithmetic independently recalculated: INT8 accumulation bound, 384 MiB KV example, 128 MiB attention-intermediate traffic, and speculation examples.
- `git diff --check` passed after the edits.
- Local link check resolved every link in these phases and figures; its remaining concurrent-work target was the root review index. The root integration pass owns the final all-repository build/link result.
- CUDA, FPGA/RTL, ESP32, radio, Jetson, Yocto images, and physical robots were not executed on hardware in this review. No CUDA or HDL compiler is installed in the review environment. Hardware timing and safety claims require the documented target experiments.
- Existing product specification tables, long code listings, and historical claims outside the sampled sections still need focused follow-up when a learner relies on them. This ledger deliberately does not label every statement verified.
