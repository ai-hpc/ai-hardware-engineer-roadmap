# RISC-V AI systems: architecture, software and an evidence-based product decision

**Reviewed 18 October 2026.** **Prerequisites:** basic C, arrays, memory bandwidth and the [SIMD lesson](../../../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/C%2B%2B%20and%20SIMD/Guide.md). **First deliverable:** run one vector binary at three emulated vector lengths, then explain what that says about a proposed AI platform. No board purchase is needed.

[Parent: AI Chip Design](../Guide.md) · [Executable RVV lab](labs/README.md)

## 1. Find the processor that does the arithmetic

RISC-V describes instruction semantics. A product also needs an implementation, memory system, firmware, operating system, drivers, compiler and runtime. An open ISA does not make every core implementation, accelerator, peripheral or tool open source or free to license.

| Design | RISC-V's role | What to inspect first |
|---|---|---|
| Vector CPU | Executes scalar and vector kernels | Instructions, VLEN, physical execution resources, cache and compiler |
| CPU plus NPU | Schedules a distinct compute engine | Supported operators, buffer ownership, driver/runtime and fallback |
| Controllers inside an accelerator | Runs DMA, scheduling and firmware | Where tensor arithmetic actually executes and how work is synchronized |
| Host for a GPU | Runs the OS and submits GPU work | Platform/driver/toolchain support and the attached GPU |
| Custom instruction design | Adds workload-specific operations | Exact semantics, toolchain, context state and compatibility |

Draw weights, activations, accumulators, command queues and completion state on the block diagram. A CPU issuing a descriptor to a tensor engine is not necessarily executing the model's matrix instructions itself.

## 2. Separate VLEN from lanes per cycle

**VLEN** is the number of bits in an architectural vector register. **SEW** is the selected element width. **LMUL** groups registers. For a supported configuration:

```text
VLMAX = LMUL * VLEN / SEW
```

With VLEN=256, SEW=32 and LMUL=1, the maximum is eight elements. That says how many elements fit in the register group; it does not say eight additions retire each cycle. The implementation may process the register over several cycles, and dependencies or memory can dominate.

The program asks `vsetvl` how many elements to handle for the remaining work and advances by the returned value. It must not assume one constant width. Tail and mask policies also matter: values outside the active region may be unspecified, so do not store them as valid output.

Use the ratified [Vector 1.0 specification](https://docs.riscv.org/reference/isa/unpriv/v-st-ext.html) for the exact rules. A vector-length-agnostic loop still needs the required ISA, ABI and operating-system support. It is not a promise that every RISC-V binary runs on every RISC-V core, or that live vector state can migrate freely between different VLEN implementations.

The [RVV lab](labs/README.md) compiles real vector instructions and executes the same binary under QEMU at VLEN 128, 256 and 512. A deliberately fixed-eight-element increment passes the matching emulator configuration but fails the others. Emulator wall time does not measure a real core's throughput or power.

## 3. Understand profiles and extension status

A profile groups architectural requirements to make a software target more predictable. It does not promise a camera driver, a model runtime, memory bandwidth or whole-system reliability. Consult the [ratified specifications library](https://docs.riscv.org/) and the specific profile version.

Treat these as separate evidence:

- A ratified standard with a named version.
- A draft in an official working group's repository.
- A vendor instruction extension supported by its compiler.
- A paper proposing an instruction or implementation.
- A shipped core on which a particular binary actually runs.

The [Vector Matrix Extension repository](https://github.com/riscv/riscv-vme) and [Integrated Matrix Extension snapshots](https://riscv.github.io/integrated-matrix-extension/) are useful current research inputs. Their presence does not make every instruction named “matrix” part of base RVV 1.0. Pin the specification and implementation revision rather than importing an extension's marketing name into a compatibility claim.

## 4. Read a current chip document carefully: SpacemiT K3

The vendor's K3 datasheet V1.9, dated 24 September 2026, distinguishes **eight X100 application cores with VLEN 256** from **eight A100 AI cores with VLEN 1024**. Here A100 is a SpacemiT core name, not NVIDIA's A100 GPU. It also identifies two RT24 management cores. The AI core section qualifies its profile claim as RVA23* without the hypervisor extension. The advertised **60 TOPS is INT4 sparse**, not a generic dense FP16 rate. [Pinned vendor datasheet](https://github.com/spacemit-com/docs-chip/blob/b43b4f7190e540e2e4394fc58c86741de904e4e3/en/key_stone/k3/k3_docs/k3_ds.md).

This is vendor specification evidence, not a course benchmark. Before running a vector workload, determine which core executes it, which extensions are exposed, and what scheduling/affinity policy the BSP uses. Do not collapse the two different core types into one “eight-core 1024-bit CPU.”

The same datasheet lists a 64-bit LPDDR5 interface at 6,400 Mbit/s per pin and up to 32 GB capacity. The ideal interface calculation is:

```text
6.4 billion transfers/s * 64 bits / 8 = 51.2 GB/s
```

This is a peak payload calculation before utilization losses, not measured application bandwidth. Do not multiply by two again for DDR: the transfer-rate figure already includes both edges.

Now assume a **dense** 30-billion-parameter model stores exactly four bits per weight and reads all weights once per unbatched decode step. Its ideal weight payload is 15 GB before scales or metadata. At 51.2 GB/s, weight traffic alone takes about 293 ms, giving a ceiling near 3.41 steps/s under those assumptions.

If a vendor reports a faster “30B” result, investigate active versus total parameters, MoE routing, quantization format, batching, caching and the timing boundary. The right response is to obtain the workload details, not to assume either the calculation or the headline describes the same experiment.

## 5. Check software availability at the board level

Canonical's February announcement distinguished Ubuntu 24.04 enablement for K1 products from Ubuntu 26.04 preview work on K3. That historical announcement cannot establish which current image enables every peripheral on a particular board. Record the image date/hash, kernel, firmware and board revision, then test the required devices. [Canonical announcement](https://canonical.com/blog/spacemit-announces-availability-of-ubuntu-on-k3-k1-series).

Build a small enablement table:

| Requirement | Evidence to obtain |
|---|---|
| Boots the intended board | Exact image, boot log and device tree |
| Uses the intended vector/core path | Compiler flags, generated instructions and CPU affinity |
| Runs the required model | Pinned converter/runtime, operator coverage and output parity |
| Captures a sensor stream | Real format, timestamps, dropped frames and recovery |
| Meets latency and energy budgets | Full workload with sustained clocks, temperature and power boundaries |
| Can be maintained | Update/rollback procedure, source availability and supported lifecycle |

An upstream compiler patch, vendor BSP and released distribution package are three different delivery stages. A model that silently falls back to scalar CPU work may be correct but miss the product's timing budget.

## 6. Update the host-versus-accelerator picture

The course should not describe RISC-V GPU hosting as only hypothetical. SiFive's August 2026 BigSky announcement reports CUDA running with the RISC-V system acting as a host for NVIDIA GPUs. Its September announcement describes a **demonstration-only** ROCm 10.0 setup using P870-D host CPUs and Radeon AI PRO R9700 GPUs for inference. These are vendor-reported development milestones, not proof of general availability on arbitrary RISC-V boards. [BigSky announcement](https://www.sifive.com/press/risc-v-in-the-datacenter-bigsky-development-server), [ROCm demonstration](https://www.sifive.com/press/sifive-amd-rocm-riscv-datacenter-servers).

The distinction matters: GPU arithmetic still executes on the GPU. This is different from compiling a neural-network kernel into RVV instructions or a vendor matrix extension. Similarly, licensable CPU/vector/matrix IP is not the same product as a board with a supported software image. [SiFive's Intelligence IP announcement](https://www.sifive.com/press/new-x100-series-second-gen-intelligence-family) is an example of that separate stage.

## 7. Turn market claims into engineering hypotheses

Avoid unsupported “high/medium/low threat” scores. Pick a buyer and a task, then define a comparison the buyer can reproduce.

For an illustrative warehouse camera, requirements might include one named detection model, a quality threshold, capture-to-result latency, a sustained power envelope, two camera interfaces and a defined update lifetime. A new CPU architecture matters only if its complete platform meets those requirements at an acceptable integration cost.

Compare candidates on the same input set and quality gate. Record unsupported operators, conversion effort, debugging access, thermal behavior, recovery and maintenance. A lower silicon price can be outweighed by a longer integration effort; an ISA license saving does not remove verification, physical design, packaging, test or software costs.

A bounded workload can make custom hardware easier to evaluate because the required operations are known. Treat that as a design hypothesis, not a forecast that one ISA must win an entire market. A convincing result may favor an existing Arm/x86/GPU/NPU platform for the selected product.

## 8. A three-session project

| Session | Work | Evidence |
|---|---|---|
| 1: instruction contract | Run the RVV lab; predict VLMAX; inspect assembly and two broken variants | Same binary identity, three emulator configurations, detected failures |
| 2: platform contract | Choose one chip/board and pin its specifications, BSP and runtime | Block diagram and enablement table with unresolved items |
| 3: product decision | Define one workload and compare two feasible platforms | Quality/latency/energy/cost assumptions and a reproducible next experiment |

Without hardware, submit a measurement plan and analytical bounds. Mark device throughput, power and peripheral operation **not run**. Do not convert QEMU timings or a vendor headline into a measured product comparison.

## Check your understanding

1. Does VLEN=1024 imply 32 FP32 additions per cycle? **No.** It gives a register-capacity relationship at SEW=32/LMUL=1; issue rate and execution width are microarchitectural.
2. Can code compiled for a vendor matrix extension run because a board supports RVV 1.0? **Not necessarily.** The required extension and toolchain contract must also match.
3. Does 60 sparse INT4 TOPS predict dense FP16 model throughput? **No.** Arithmetic type, sparsity, memory traffic and the actual instruction path differ.
4. Does CUDA on a RISC-V host mean the CPU executes NVIDIA GPU instructions? **No.** Host and device code execute on their respective processors.
5. What would disprove a claim that a new ISA lowers deployment cost? **A controlled comparison showing larger integration/maintenance effort or failure to meet the required workload constraints.**

Complete the [shared rubric](../../../Learning-Guide.md#completion-rubric), then carry the tested instruction/runtime contract and platform evidence into the [SoC capstone](../Guide.md#physical-ai-soc-capstone).
