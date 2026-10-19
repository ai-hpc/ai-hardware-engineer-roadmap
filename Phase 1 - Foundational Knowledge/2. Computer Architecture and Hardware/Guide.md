# Computer Architecture for AI Hardware Engineers

<div class="course-identity computer-architecture" markdown="1">
<div class="course-identity__icon">CPU</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 2 · Computer Architecture</p>
<p class="course-identity__title">Understand how CPUs, GPUs, caches, memory, and accelerators actually execute work.</p>
<p class="course-identity__meta">Artifact: architecture analysis note · Measure: latency, bandwidth, locality, throughput</p>
</div>
</div>


Phase 1 · Section 2 — How processors and memory systems work, and why that shapes every AI accelerator ever built.

> **Goal:** By the end of this section you can read a GPU or NPU architecture paper, identify when an AI workload is limited by computation, data movement, or scheduling, and reason about the trade-offs in any custom accelerator design.

**Layers:** L5 architecture, with L3 runtime connections. **Prerequisites:** arrays, multiplication, basic digital logic, and the [first numerical lab](../Labs/Numbers-Shapes-and-Bytes.md). **Roles:** AI systems, accelerator, embedded, and robotics engineers. **Next:** [operating systems](../3.%20Operating%20Systems/Guide.md) and [parallel computing](../4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md).

**Learning route:** first read §§1–4 and complete the numerical lab; then study §§5–9 and one profiling exercise; finish with a workload case study in §10. Assembly snippets are explanatory unless a test status is stated. The important artifact is a correct workload estimate plus measurements that can confirm or reject it, not a memorized product specification table.

---

## 1. The Two Computing Paradigms

Processors balance latency, throughput, power, area, and cost. CPU and GPU designs emphasize different kinds of work; neither is universally faster.

| Question | Typical high-performance CPU | Typical GPU |
|---|---|---|
| How is work organized? | Independently scheduled cores and software threads | Many threads grouped for SIMT execution |
| How are stalls hidden? | Caches, speculation, out-of-order execution, and sometimes hardware threads | Caches and scheduling other ready warps |
| What work fits well? | Irregular control flow, small tasks, OS services | Large regular workloads with enough parallel work |
| What can dominate a small robot pipeline? | Sensor handling, synchronization, control, preprocessing | Kernel dispatch, transfer, occupancy, and numerical work |

A CPU core, GPU arithmetic lane, and Tensor Core are different resources. Comparing their counts as if each were an interchangeable processor is misleading. Compare the same workload, precision, correctness criterion, and timing boundary. The [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) explains the host/device and thread hierarchy.

**The transistor budget question:** more execution units, larger caches, wider interfaces, and richer control logic all consume silicon and power. A useful architecture sketch labels these resources; it does not assign unsupported area percentages to named products.

**AI connection:** dense attention over a sequence of length `n` contains quadratic pairwise work, while projection layers contain matrix multiplications. Small-batch decoding, large-batch prefill, sparse operators, and tiny robot control networks can have different bottlenecks. Predict bytes and operations for the actual shape before deciding where to run it.

**First runnable exercise:** [Numbers, shapes, and bytes](../Labs/Numbers-Shapes-and-Bytes.md) walks through a tiny quantized operator and a batch-dependent memory budget, with a CPU reference and answer guidance.

---

## 2. Instruction Set Architecture (ISA)

The ISA is the **contract between software and hardware**. Hardware can be redesigned completely (new microarchitecture) as long as the ISA stays compatible — software keeps running.

### RISC vs CISC

| Property | AArch64 / RISC-V | x86-64 |
|---|---|---|
| Instruction length | AArch64: 32 bits; RISC-V base: 32 bits, with optional compressed encodings | Variable length, up to 15 bytes |
| Memory operands | Arithmetic generally operates on registers; loads/stores move data | Many instructions allow a memory operand; not every instruction does |
| Implementation | May be in-order or out-of-order, narrow or wide | May be in-order or out-of-order, narrow or wide |
| Code density and efficiency | Depend on extensions, compiler, implementation, and workload | Depend on extensions, compiler, implementation, and workload |

The ISA is a software-visible contract, not a power-efficiency score. Modern implementations may translate architectural instructions into internal operations. RISC-V's [compressed extension](https://docs.riscv.org/reference/isa/unpriv/c-st-ext.html) permits 16-bit instructions mixed with 32-bit instructions; “RISC means every instruction is 32 bits” is incorrect.

### The Three ISAs You Will Encounter

**ARM64 (AArch64)**

```
31 general-purpose 64-bit registers (X0–X30) + zero register (XZR)
128-bit SIMD/FP registers V0–V31 (also addressable as S/D/Q)

ADD  X0, X1, X2        // X0 = X1 + X2
LDR  X0, [X1, #8]      // X0 = mem[X1 + 8]
FMLA V0.4S, V1.4S, V2.4S  // 4-wide FP32 fused multiply-add (NEON)
```

Used in: Apple Silicon (M-series), NVIDIA Jetson, Qualcomm Snapdragon, AWS Graviton.

**x86-64**

```
16 general-purpose 64-bit registers (RAX–R15)
SIMD: XMM (128b), YMM (256b/AVX2), ZMM (512b/AVX-512)

mov  rax, [rbx + 8]    // load from memory
vmulps ymm0, ymm1, ymm2 // 8-wide FP32 multiply (AVX2)
vfmadd231ps zmm0, zmm1, zmm2  // 16-wide FP32 FMA (AVX-512)
```

Used in: Intel/AMD data-centre servers — the dominant platform for model training.

**RISC-V**

```
Modular open ISA: RV64I base + extensions (M, F, D, A, V, ...)
32 architectural integer registers in RV64I; x0 always reads as zero

add  x1, x2, x3        // x1 = x2 + x3
vsetvli t0, a0, e32,m4 // V extension: set vector length, FP32, 4x LMUL
vfmacc.vv v0, v4, v8   // vector FMA (scalable width)
```

Used in: Emerging edge AI chips (Tenstorrent, SiFive), custom accelerator control cores.

### ISA comparison: check the target, not just its family

| Feature | AArch64 | x86-64 baseline | RV64I with optional extensions |
|---|---|---|---|
| Writable integer registers | 31 general-purpose registers | 16 general-purpose registers | 31; x0 is fixed zero |
| Vector facilities | NEON; SVE/SVE2 on supporting targets | SSE; AVX/AVX2/AVX-512 on supporting targets | V extension on supporting targets |
| What to record | CPU model, extensions, ABI, compiler target | CPU model, extensions, ABI, compiler target | Exact ISA string, ABI, compiler target |

Do not compile for an extension just because it exists in the architecture family. Query the actual device and toolchain; save the target options with the benchmark.

---

## 3. CPU Microarchitecture — The Pipeline

Understanding the CPU pipeline gives you the vocabulary to reason about any processor's performance: GPUs, NPUs, and custom accelerators all face the same fundamental constraints.

### 3.1 The Classic 5-Stage Pipeline

```
Cycle:   1    2    3    4    5    6    7    8
Instr 1: IF   ID   EX   MEM  WB
Instr 2:      IF   ID   EX   MEM  WB
Instr 3:           IF   ID   EX   MEM  WB
Instr 4:                IF   ID   EX   MEM  WB

IF  = Instruction Fetch (read from instruction cache)
ID  = Instruction Decode + register file read
EX  = Execute (ALU, FP unit, address calculation)
MEM = Memory access (load/store)
WB  = Write-Back (result → register file)
```

One instruction finishes per cycle at steady state — ideal throughput is 1 IPC (instructions per cycle).

### 3.2 Hazards — When the Pipeline Stalls

**Data hazard:** instruction needs a result not yet written back.

```
ADD X1, X2, X3    // produces X1 in WB (cycle 5)
SUB X4, X1, X5    // needs X1 in ID (cycle 3) — too early!
```

Solution: **forwarding** (bypass) routes the EX result directly to the next instruction's input, eliminating the stall in most cases.

```
        ADD  EX stage ──forward──► SUB  EX stage
                         ↑
               result available here, not after WB
```

**Load-use hazard** — forwarding can't help when the data isn't ready yet:

```
LDR X1, [X2]      // data available after MEM (cycle 4)
ADD X3, X1, X4    // EX needs X1 in cycle 3 — too early even with forwarding!
→ 1-cycle stall (bubble), then forward from MEM/WB
```

**Control hazard:** a branch changes the PC; the instructions already in the pipeline may be wrong.

```
BEQ X1, X2, Label    // branch resolved in EX (cycle 3)
  →  already fetched 2 wrong instructions behind it
  →  must flush them: 2-cycle penalty for a 5-stage pipe
```

**AI connection:** when lanes in a warp follow different paths, some lanes are masked while a path executes. This can reduce useful work per instruction. A branch taken uniformly is different from divergence; removing a branch by doing expensive unnecessary work can be slower. Preserve synchronization correctness and measure the actual kernel.

### 3.3 Superscalar Execution

Fetch and issue **multiple instructions per cycle** — wider pipeline, more parallelism:

```
4-wide superscalar (modern x86/ARM):

Cycle 1: Fetch 4 instructions
Cycle 2: Decode 4 → 4 micro-ops
Cycle 3: Issue up to 4 to execution units (if no dependencies)
Cycle 4: Multiple ALUs, FP units, load/store units execute in parallel

Ideal: 4 IPC (limited by dependencies, memory stalls, branch mispredictions)
Observed IPC: measure for the named workload and processor; it need not approach 4
```

**Measure rather than equate widths:** fetch, decode, dispatch, issue, and retirement widths describe different pipeline boundaries. Decode width alone is not peak IPC, and IPC depends on the instruction mix, dependencies, cache behavior, and measurement interval. Use a named processor's optimization manual when filling in these values.

### 3.4 Out-of-Order Execution

Modern high-performance CPUs don't wait for slow instructions — they look ahead and execute whatever is ready.

```
Program order:              Execution order:
1. ADD X1, X2, X3           1. SUB X6, X7, X8   (no dependencies, runs first)
2. MUL X4, X1, X5           2. ADD X1, X2, X3   (data ready)
3. SUB X6, X7, X8           3. MUL X4, X1, X5   (waits for ADD result)
4. DIV X9, X4, X10          4. DIV X9, X4, X10  (waits for MUL)
```

**Register renaming** eliminates false dependencies:

```
Source code:                After renaming:
  ADD R1, R2, R3             ADD P47, P2, P3     ← writes physical reg P47
  MUL R4, R1, R5             MUL P48, P47, P5    ← reads P47 (true dependency)
  ADD R1, R6, R7             ADD P49, P6, P7     ← writes P49 (new physical reg!)
  SUB R8, R1, R9             SUB P50, P49, P9    ← reads P49

Without renaming: 3rd ADD writes R1, creating a WAW hazard with 1st ADD.
With renaming: P47 and P49 are independent — 3rd ADD can execute in parallel with 2nd MUL.
```

Key hardware structures:

| Structure | Role | Resource limit to investigate |
|---|---|---|
| **Reorder Buffer (ROB)** | Tracks in-flight instructions and in-order retirement | Number of entries and allocation rules |
| **Issue queues / reservation stations** | Hold operations until operands and execution resources are ready | Queue capacity and issue ports |
| **Register renaming** | Separates architectural names from physical storage | Physical registers and rename bandwidth |
| **Result forwarding** | Makes produced values available to dependents | Bypass paths, latency, and fan-out |
| **Load/store queues** | Track memory operations and ordering constraints | Queue capacity and memory dependencies |

These structures vary by implementation; a textbook common-data-bus diagram is not a literal floorplan of every modern CPU.

**OoO execution pipeline flow:**

```
Fetch → Decode → Rename → Dispatch → Issue → Execute → Complete → Retire
  │         │        │         │                                    │
  ├─────────┤        │         │                                    │
  in-order           │    Reservation    Out-of-order               │
  (front-end)        │    Stations →     (back-end)                 │
                     │    Execution                                 │
                     │    Units (ALU,                    in-order   │
                     │    FPU, LSU)                     (commit)   │
                     │                                              │
                     └── ROB tracks program order ──────────────────┘
```

**Commit/Retire:** results are computed out-of-order but committed to architectural state in program order. This enables **precise exceptions** — if an instruction faults, all prior instructions have committed and all later instructions are discarded cleanly.

### 3.5 Branch Prediction

A branch predictor guesses the next control-flow path before the branch resolves. A wrong guess discards speculative work and redirects execution. Accuracy and recovery cost depend on the workload and microarchitecture; measure them for a named processor.

**2-bit saturating counter** — simplest dynamic predictor:

```
State:       00            01            10          11
Prediction: not taken     not taken     taken       taken

On a taken outcome: increment, saturating at 11.
On a not-taken outcome: decrement, saturating at 00.
Two opposite outcomes are needed to change prediction from a STRONG state;
one is enough from a weak state. Initialization and branch history matter.
```

**Two-level correlating predictor** — uses global branch history to predict:

```
Global History Register (GHR): last N branch outcomes as a bit string
  e.g., GHR = 10110 (last 5 branches: T N T T N)

Index into Pattern History Table (PHT):
  index = hash(branch_PC, GHR)
  PHT[index] = 2-bit counter

Captures correlations: "this branch is taken if the previous two were also taken"
```

**Longer-history predictors:** several tables can track different lengths of prior branch history. Tagged entries help distinguish contexts that share an index. The useful lesson is that prediction depends on patterns, capacity, and aliasing; do not infer a product's exact predictor design or accuracy from this sketch.

**Additional predictor hardware:**

| Structure | Purpose |
|-----------|---------|
| **Branch Target Buffer (BTB)** | Caches `branch PC → target address` (no need to decode to know target) |
| **Return Stack Buffer (RSB)** | Tracks likely return addresses; nesting and context changes affect prediction |
| **Indirect Branch Predictor** | Predicts target of `jmp [reg]` (virtual function calls) |
| **Loop Predictor** | Counts iterations — predicts loop exit after N iterations |

**Illustrative cost model (assumed values, not a processor measurement):**

```
4-wide superscalar × 15 cycles = 60 wasted instruction slots per mispredict

If 5% of instructions are branches with 96% accuracy:
  Mispredict rate = 5% × 4% = 0.2% of all instructions
  CPI penalty = 0.2% × 15 cycles = 0.03 CPI  (seems small, but adds up)
```

---

## 4. Memory Hierarchy — Account for Data Movement

Fast arithmetic units are useful only when operands arrive in time. Data movement can limit an AI workload even when peak compute is high; the limiting resource depends on shape, reuse, and implementation.

### 4.1 The Memory Mountain

| Resource | What it holds | Constraint to investigate |
|---|---|---|
| Registers | Thread-local live values | Finite capacity, operand ports, instruction dependencies |
| L1 cache / shared memory | Data close to an execution unit | Locality, banking, capacity, allocation |
| L2 / last-level cache | Data reused across a larger region | Working set, sharing, contention |
| DRAM / HBM | Main working memory | Sustained bandwidth, access latency, capacity |
| Storage | Persistent models and datasets | I/O latency, throughput, filesystem behavior |

Registers do not provide zero-latency computation or unlimited bandwidth. Cache and DRAM timings vary by processor, load, access pattern, and clock. A single CPU/GPU timing table cannot supply constants for every platform. Name the boundary and measure: dependent loads reveal latency; independent streams can reveal throughput. Cache reuse can reduce traffic reaching the slower levels.

### 4.2 Cache Organisation

A cache is divided into **sets** of **ways**. An address maps to exactly one set; within the set, it can go into any way.

```
Address bits:  [  Tag  ] [  Index  ] [ Offset ]
                   ↓          ↓
               Compare    Select set   Byte within line

Illustrative 8-way set-associative cache (64-byte lines, 8 MiB):
  Sets = 8 MiB / (8 ways × 64 bytes) = 16,384 sets
  Index bits  = log2(16,384) = 14
  Offset bits = log2(64)     = 6
  Tag bits    = address bits - index bits - offset bits
  (use the relevant physical/virtual address width, not pointer width blindly)
```

**Associativity trade-off:**

| Type | Miss rate | Hardware cost | Used where |
|------|-----------|---------------|------------|
| Direct-mapped (1-way) | High (conflict misses) | Minimal | Not common today |
| 4–8 way set-associative | Low | Moderate | L1/L2/L3 caches |
| Fully associative | Lowest | Very high | TLB, small caches |

**Three kinds of misses:**

1. **Compulsory (cold):** first touch — unavoidable
2. **Capacity:** working set larger than cache — solution: tile/block your algorithm
3. **Conflict:** same index but different tags evict each other — solution: increase associativity or change data layout

### 4.3 Cache Coherence (Why Multi-Core Is Hard)

When multiple cores have private caches, they can hold **stale copies** of the same memory location. Coherence protocols enforce a consistent view.

**MESI protocol** — each cache line has one of four states:

```
M (Modified)   — this core wrote it; dirty; no other core has it
E (Exclusive)  — clean; only this core has it
S (Shared)     — multiple cores have valid read-only copies
I (Invalid)    — stale or not present

Transitions:
  Core 0 reads X:  I → E (if only core)  or  I → S (if others have it)
  Core 0 writes X: any → M; all other cores' copies → I (invalidate)
  Core 1 reads X while Core 0 has M: Core 0 flushes, both → S
```

**False sharing** — two cores write to different variables that share the same cache line:

```
struct Counter {
    int core0_count;    // byte 0–3
    int core1_count;    // byte 4–7  ← same 64-byte cache line!
};

Core 0 writes core0_count → invalidates Core 1's copy
Core 1 writes core1_count → invalidates Core 0's copy
→ cache line ping-pongs between cores (100+ cycle penalty each time)

Fix: pad to separate cache lines
struct Counter {
    alignas(64) int core0_count;
    alignas(64) int core1_count;
};
```

**Why this matters for AI:** Cache coherence and distributed collectives are different mechanisms. GPU interconnects can enable peer memory access, but their presence does not imply transparent cache coherence across every device and allocation. An all-reduce explicitly combines values and communicates results; its cost depends on bytes, topology, synchronization, and algorithm. Verify the platform memory model rather than extending CPU MESI assumptions to GPU fabrics.

### 4.4 Virtual Memory and TLB

Virtual memory gives each process a private address space, mapped to physical memory by the page table.

```
Virtual Address → TLB lookup → Physical Address

TLB hit:   1 cycle (translation cached)
TLB miss:  page table walk → 50–500 cycles!

TLB hierarchy:
  L1 dTLB:  64 entries, 1 cycle
  L2 TLB:   512–2048 entries, 5–10 cycles
  Page walk: traverse 4-level table in memory (multiple cache accesses)
```

**Huge pages** reduce TLB pressure:

```
4 KiB pages: 4 GiB working set = 1,048,576 pages → 64 entries cover 256 KiB
2 MiB pages: 4 GiB working set = 2,048 pages → 64 entries cover 128 MiB

Analytical coverage examples: 4 GiB, 4 KiB / 2 MiB pages, 64-entry TLB.
Actual page size and translation support depend on the allocation and device.
```

### 4.5 DRAM and HBM

**DDR example:** a channel transferring 6,400 million transfers/s over 64 data bits has a theoretical payload rate of `6,400×10⁶ × 8 = 51.2 GB/s`. Eight such channels give `409.6 GB/s` before protocol losses and workload effects. Installed DIMMs, supported rates, channel population, and access pattern determine the useful rate. Capacity and latency are separate properties.

**HBM:** stacked DRAM connects through many short, wide interfaces within an advanced package. The number of stacks, interface rate, enabled capacity, and packaging differ by product. Compare an exact SKU's bandwidth with measured sustained traffic rather than assuming a fixed HBM-to-GDDR multiplier.

**AI connection:** low-batch autoregressive decode often has little reuse of model weights per token. Batching, prefill, attention length, cache behavior, quantization overhead, and communication can change the limiting resource. “All LLM inference is memory-bound” is not a useful universal rule.

---

## 5. SIMD — One Instruction, Many Data

SIMD applies one instruction to several packed data elements. SIMT exposes threads that may follow different control-flow paths, and matrix units expose matrix operations. These models share data parallelism, but their programming and synchronization rules differ.

### 5.1 How SIMD Works

```
Scalar (no SIMD):                   SIMD (4-wide FP32):
  a[0] = b[0] * c[0]   4 scalar multiplies       [a0 a1 a2 a3] = [b0 b1 b2 b3]
  a[1] = b[1] * c[1]                                 × [c0 c1 c2 c3]
  a[2] = b[2] * c[2]                  1 instruction, 4 results
  a[3] = b[3] * c[3]
```

**SIMD widths across ISAs:**

| ISA | Extension | Width | FP32 lanes | Fused-multiply-add? |
|-----|-----------|-------|------------|---------------------|
| ARM64 | NEON | 128 bits | 4 | Yes (FMLA) |
| ARM64 | SVE/SVE2 | 128–2048 bits (scalable) | 4–64 | Yes |
| x86-64 | SSE4.2 | 128 bits | 4 | No |
| x86-64 | AVX2 | 256 bits | 8 | FMA is a separate feature to check |
| x86-64 | AVX-512 | 512 bits | 16 | Yes |
| RISC-V | RVV | 128–65,536 bits (scalable) | variable | Yes |

### 5.2 Example — Vectorised Dot Product (AVX2)

```cpp
// Scalar: N multiplications + N additions
float dot_scalar(const float* a, const float* b, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}

// AVX2: processes 8 FP32 per iteration
#include <immintrin.h>
float dot_avx2(const float* a, const float* b, int N) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(va, vb, acc);   // acc += va * vb (8-wide FMA)
    }
    // horizontal reduction of 8 lanes → scalar
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);
    for (; i < N; ++i) result += a[i] * b[i]; // scalar tail: no out-of-bounds read
    return result;
}
// Requires AVX2 and FMA support; compile with matching flags and check the target.
// Different reduction order and FMA rounding require a stated floating-point tolerance.
```

**Tested here:** this example was compiled with GCC 13.3.0, `-O2 -mavx2 -mfma -fsanitize=address,undefined`, and compared with the scalar reference for `N = 0, 1, 7, 8, 9, 17, 101` on an AVX2/FMA-capable x86 host. All comparisons passed with no sanitizer report. This validates the tail handling for those cases, not a speedup claim.

### 5.3 SIMD → GPU SIMT

A GPU warp (32 threads) executing the same instruction at the same time is SIMD taken to 32-wide, with the additional twist that each "lane" is an independent thread with its own register state.

```
CPU SIMD (AVX-512):              GPU SIMT (warp of 32):
  1 instruction                    1 instruction
  16 FP32 lanes                    32 "lanes" = 32 threads
  all same operation                all same operation (PC)
  same data register               each thread has own registers
  mask/select per vector lane      per-thread paths can diverge
```

Tensor Cores perform matrix multiply-accumulate work. A supported `wmma::mma_sync` call describes a cooperative warp operation on matrix fragments; it is not 4,096 independent scalar lanes or a promise of one machine instruction. Fragment shapes and instruction forms depend on type and architecture.

---

## 6. GPU Architecture (Conceptual)

The CUDA programming model is covered in Section 4. Here we look at the **hardware** — what the silicon actually does.

### 6.1 SM: The Streaming Multiprocessor

The GPU is a collection of SMs. Every kernel launch distributes thread blocks across available SMs.

A concrete memory/occupancy example is A100 (compute capability 8.0): an SM supports up to 64 resident warps, a 64K-entry 32-bit register file, and a 192 KB combined L1/shared-memory resource, with shared-memory carveout up to 164 KB. These are hardware ceilings, not guarantees that any one kernel reaches them. Register use, shared memory, and block size can reduce occupancy. Compute capability 8.6 has different limits; do not mix its 48-warp ceiling into an A100 diagram. [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy).

### 6.2 SIMT Execution and Warp Scheduling

A warp scheduler selects an eligible warp to issue an instruction. Resident threads retain their state in the SM register file, so selecting another ready warp does not require an OS-style context save. Issue opportunities still depend on dependencies, execution pipelines, scheduling rules, and available work; not every warp executes every cycle.

```
Cycle 100: Warp 0 issues LOAD (takes 200+ cycles to return)
Cycle 101: Warp scheduler picks Warp 1 (compute-bound, no stall)
Cycle 102: Warp 2
Cycle 103: Warp 3
...
Cycle 300: Warp 0's data arrives; it becomes eligible again
```

This timeline is illustrative, not a measured latency. Other ready work can hide some latency, but extra occupancy helps only while it provides useful independent work. Both CPUs and GPUs also depend on caches and data reuse.

### 6.3 GPU Memory Hierarchy

| CUDA resource | Scope and purpose | Common mistake |
|---|---|---|
| Registers | Per-thread values backed by a finite SM register file | Assuming every local variable stays in a register |
| Local memory | Per-thread address space backed by device memory and caches | Assuming “local” means physically nearby |
| Shared memory | Explicitly managed block storage; some features extend access across a cluster | Forgetting barriers, bank conflicts, or capacity limits |
| L1 / L2 caches | Hardware-managed reuse within the architecture's hierarchy | Treating every global load as an HBM transaction |
| Global memory | Device-wide allocation space | Ignoring transfers, access patterns, and lifetime |
| Constant / texture access paths | Specialized read behavior and caching | Assuming their performance is identical for every access pattern |

A tiled matrix multiply stages values for reuse. Whether a value is fetched once from HBM depends on the tiling, cache behavior, and full execution, not merely on using shared memory. Query limits for the exact GPU and kernel.

### 6.4 Tensor Cores

Tensor Cores accelerate matrix multiply-accumulate (MMA). The following 16×16×16 operation is a mathematical example; the compiler may lower it to multiple architecture-specific instructions:

```
D = A × B + C
  A: 16×16 matrix (FP16 or BF16)
  B: 16×16 matrix (FP16 or BF16)
  C: 16×16 accumulator (FP32)
  D: 16×16 result (FP32)

```

Peak throughput depends on data type, accumulation mode, sparsity assumptions, device SKU, and clock. A matrix-unit peak and scalar FP32 peak describe different operations; they are not an application speedup measurement. Frameworks may dispatch to cuBLAS, cuBLASLt, cuDNN, compiler-generated kernels, or other paths. Inspect the actual call and generated kernel rather than assuming every matrix operation becomes a particular `wmma` instruction.

---

## 7. AI Accelerator Design Patterns

### 7.1 Systolic Arrays

A systolic array is a grid of processing elements (PEs) where data moves between neighbours in a scheduled pattern. Array shape, data type, buffering, and dataflow are architecture choices; they are not fixed across every TPU or accelerator generation.

```
Input matrix A rows flow →→→→→→→→
Weight matrix B cols flow ↓↓↓↓↓↓↓

     PE(0,0) → PE(0,1) → PE(0,2) → ...
       ↓         ↓         ↓
     PE(1,0) → PE(1,1) → PE(1,2) → ...
       ↓         ↓         ↓
     PE(2,0) → PE(2,1) → PE(2,2) → ...

Each PE: accumulator += A_val × B_val
Result location and movement depend on the chosen dataflow.

Potential benefit: nearby reuse reduces repeated transfers.
Account for input skew, pipeline fill/drain, partial sums, and edge tiles.
```

A matrix instruction specifies an operation, not a public transistor-level layout. Do not infer that a Tensor Core is a particular systolic array merely from the fragment shape.

### 7.2 Dataflow Architectures

In a dataflow view, operations become eligible when their inputs are available. An accelerator can keep intermediate values near the operations that consume them. CPUs and GPUs also retain intermediates in registers and caches; the comparison below deliberately illustrates a poor unfused memory path, not an unavoidable CPU/GPU behavior.

```
Von Neumann:                     Dataflow:
  LOAD A from HBM                  A ──► multiply ──► add ──► result
  LOAD B from HBM                  B ──►    ↑
  MUL C = A × B                    C ──────►┘
  STORE C to HBM
  LOAD C from HBM
  LOAD D from HBM
  ADD E = C + D
  STORE E to HBM                 ← no HBM round-trips for intermediate values
```

SambaNova SN40L, Cerebras WSE-3, and Graphcore IPU are commercial dataflow architectures for AI.

### 7.3 NPUs — Neural Processing Units

An NPU dedicates hardware and software support to selected neural-network operations. Support can vary by operator, tensor shape, precision, layout, and runtime version. Unsupported operations may fall back to a CPU or GPU and introduce copies or synchronization.

| Question before deployment | Evidence |
|---|---|
| Can the model run correctly? | Supported-operator report and reference comparison |
| Does it run on the NPU throughout? | Partition/fallback report and runtime trace |
| Does it help this robot? | End-to-end age, power, thermal, and quality measurements |
| Can the result be reproduced? | Exact SoC, runtime/compiler, model, and configuration revisions |

A vendor TOPS figure alone does not establish model throughput or energy efficiency. Count supported precision and sparsity assumptions, then benchmark the workload within its latency and accuracy constraints.

---

## 8. The Roofline Model — Your Most Important Analysis Tool

The roofline model compares a compute ceiling with a data-movement ceiling. It helps form a bottleneck hypothesis; dispatch overhead, dependencies, synchronization, and poor utilization can keep an implementation below both roofs.

### 8.1 Arithmetic Intensity

```
Arithmetic Intensity (AI) = FLOPs executed / Bytes transferred from memory

Examples:
  Vector addition  y = a + b:        1 FLOP / 12 bytes = 0.08 FLOP/byte  (memory-bound)
  Matrix multiply  C = A × B (N=1000): 2N³ FLOPs / (3N² × 4 bytes)
    = 2×10⁹ / 12×10⁶ ≈ 167 FLOP/byte (ideal one-read/one-write traffic)
  Compare intensity with the chosen device/precision roof; 167 is not automatically compute-bound.
  FP32 matrix-vector multiply:       about 2 FLOP / 4 weight bytes = 0.5
  FP16 matrix-vector multiply:       about 2 FLOP / 2 weight bytes = 1.0
  (large matrix, no weight reuse, input/output traffic omitted)
  Attention: derive from batch, heads, lengths, dtype, and actual tiling
```

### 8.2 The Roofline

```
TFLOPS (log)
 │                          ───────────────── Assumed compute roof (989 TFLOPS)
 │                      ╱
 │                  ╱  ╲── compute-bound region
 │              ╱        (more compute = more speed)
 │          ╱
 │      ╱ ← memory-bandwidth roof slope = HBM BW × AI
 │  ╱       (memory-bound region: more bandwidth = more speed)
 └──────────────────────────────────────────── Arithmetic Intensity (FLOP/byte)
         ↑
    Illustrative ridge ≈ 989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/byte
    Replace both assumptions with the named device and precision mode.

Below this ridge, the bandwidth roof is lower than the selected compute roof.
This predicts a ceiling; latency, dependencies, and low utilization may dominate first.
```

### 8.3 Where AI Workloads Live

Follow the changing shape instead of assigning an operator one permanent intensity:

| Workload change | Predicted effect to test | What may invalidate it |
|---|---|---|
| Reuse weights across a batch | More operations per weight byte | Queue delay, memory capacity, insufficient kernel support |
| Fuse two elementwise operators | Fewer intermediate reads/writes | Register pressure, repeated work, changed numerical behavior |
| Tile a matrix multiply | More reuse near execution units | Small or irregular shapes, capacity limits, edge handling |
| Quantize FP16 weights to INT8 | Half the ideal payload bytes | Scales, metadata, padding, conversion, accuracy loss |
| Increase sequence length | More attention work and cache traffic | Attention variant, caching, tiling, and sparsity |

A robot's latency budget may rule out batching even when batching improves throughput. Use the [freshness lab](../../Phase%202%20-%20Embedded%20Systems/Labs/Observation-Freshness.md) to see how a queue changes capture-to-output age.

---

## 9. Performance Analysis and Profiling

### 9.1 Amdahl's Law

Not all code parallelises. Amdahl's Law predicts the maximum speedup from parallel execution.

```
Speedup = 1 / ((1 − P) + P/N)

P = fraction of baseline execution TIME that benefits from ideal parallelism
N = number of cores/processors
```

**Example (2% serial, 98% parallel):**

| Cores | Speedup | Efficiency |
|-------|---------|------------|
| 1     | 1.0×    | 100%       |
| 4     | 3.77×   | 94%        |
| 16    | 12.31×  | 77%        |
| 64    | 28.32×  | 44%        |
| ∞     | 50.0×   | → 0%       |

The serial 2% becomes the bottleneck. No amount of cores can exceed 50× speedup.

**For AI workloads:** communication, imbalance, pipeline bubbles, input loading, and changed batch size can all violate ideal scaling assumptions. Report useful work per second and quality at a fixed workload, then separate strong scaling (fixed total work) from weak scaling (work grows with device count). A fixed “typical efficiency” is not a prediction for your system.

### 9.2 Profiling Tools

**Command status:** the following are upstream-style exercise templates for your own `./program`; they were not run here. Install compatible tools, query available counters, and account for profiler overhead. The [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) explains metric selection, replay, and collection overhead.

**CPU profiling (Linux):**

```bash
# Hardware counter sampling
perf stat -e cache-references,cache-misses,L1-dcache-misses,branches,\
          branch-misses,instructions,cycles ./program

# Record + flamegraph (the two Perl scripts require a separate FlameGraph checkout)
perf record -F 99 -g ./program
perf script | stackcollapse-perf.pl | flamegraph.pl > profile.svg
```

**GPU profiling (NVIDIA):**

```bash
# Timeline of kernels, memory transfers, CPU-GPU sync
nsys profile -o trace.nsys-rep ./program
nsys-ui trace.nsys-rep

# Per-kernel detailed metrics (roofline, occupancy, memory throughput)
ncu --set full -o report.ncu-rep ./program
ncu-ui report.ncu-rep
```

**Cache simulation:**

```bash
valgrind --tool=cachegrind --cache-sim=yes ./program
cg_annotate cachegrind.out.*
```

**Benchmarking checklist:**

- Record CPU/GPU, clock/power policy, temperature, OS, compiler, and command line. Decide whether the question concerns normal deployment behavior or a controlled clock experiment.
- Record thread placement and background load; pin threads only when it matches the experiment.
- Separate initialization/cold starts from steady state and choose sufficient warmup.
- Save raw repeated samples, sample count, median, and tail observations. Include mean or other statistics when they answer the stated question.
- Check correctness before timing and repeat the comparison on an unfamiliar shape.

### 9.3 Bottleneck Diagnosis

| Observation | Competing explanations | Next measurement |
|---|---|---|
| Low compute utilization | Memory stalls, dependencies, tiny grid, dispatch gaps | Timeline, bytes moved, eligible warps / CPU stalls |
| Many last-level misses | Large streaming workload or poor locality | Requested bytes, cache reuse, tiling comparison |
| Low IPC | Dependencies, cache misses, branch recovery, instruction mix | Cycles and relevant stall/branch counters |
| Low occupancy | Resource limits or small launch | Register/shared-memory allocation and launch size |
| Low memory throughput | Too little parallel work, uncoalesced access, cached data | Traffic at each cache/memory boundary and transaction efficiency |

No single percentage identifies a root cause. For example, low occupancy can still accompany an efficient kernel with enough instruction-level parallelism.

**Worked hypothesis:** a row-major naive matrix multiply repeatedly accesses a column of `B`. Tiling may improve locality, but the size of the gain depends on cache, compiler, shape, and implementation. Start with zero-initialized `C`, compare with a reference, test edge tiles, and report actual results. Do not attach invented GFLOP/s values to an unmeasured snippet.

---

## 10. Workload Case Studies for AI and Robotics

These are design exercises with stated assumptions, not measurements from named products. They connect the architecture vocabulary to a decision you can defend.

### 10.1 A small robot policy

Assume a dense layer with 64 FP32 inputs and 32 outputs, used once per observation. Weights occupy `64×32×4 = 8192 bytes`, and the layer performs about `4096 FLOPs`. A high GPU peak says little about this tiny task: launch, synchronization, and transfers may dominate. A CPU may already hold the weights in cache.

**Experiment:** compare one warmed CPU call with a GPU call, including the transfers actually required by the application. Keep a second GPU-only timing if useful, but label its smaller boundary. Add the result to the sensor-to-action budget. Do not infer that every small model belongs on a CPU: a fused pipeline with data already on a GPU is a different case.

### 10.2 Camera perception

Assume one uncompressed `640×480` RGB8 frame at 30 frames/s. Payload alone is `640×480×3 = 921,600 bytes/frame`, or `27.648 MB/s`. Three full copies add traffic even before resizing and inference. Metadata, stride padding, and transport framing are excluded from this estimate.

**Experiment:** draw where capture, resize, inference, and publication run. Count read/write bytes for each buffer copy and state which memory boundary they cross. Record capture time and the final publication time. An optimization that reduces kernel time while adding another frame of queue delay can make the robot's observation older.

### 10.3 A batched language-model projection

Assume an FP16 `1024×1024` projection. With one input vector, weights occupy 2 MiB and the operation has roughly 1 FLOP per weight byte. Processing eight vectors while reusing those weights increases that weight-only ratio toward 8 FLOP/byte; include input/output traffic before using it as a roofline point.

**Experiment:** sweep batch size and report time per batch, useful outputs/s, peak memory, and per-request age. Hold the input/output quality contract fixed. Throughput, individual latency, and queue delay are separate outcomes.

### 10.4 Compare devices with an evidence table

| Field | Device A | Device B |
|---|---|---|
| Exact SKU, software, power mode | Record | Record |
| Same input shapes and numerical criterion | Record | Record |
| Timing boundary and warmup | Record | Record |
| Raw latency distribution and useful throughput | Measure | Measure |
| Sustained temperature/power and memory | Measure if available | Measure if available |
| Missing capability or unsupported operator | Record | Record |

Use vendor documentation to describe capabilities and your measurements to describe workload behavior. An ISA name, advertised TOPS, or a benchmark score from another workload cannot fill the result rows.

---

## 11. Speculative Execution Security

Speculative execution can improve performance but can expose side channels. Spectre and Meltdown, publicly disclosed in 2018, showed that speculative loads can leave traces in cache timing that leak secret data. See the original [Project Zero disclosure](https://projectzero.google/2018/01/reading-privileged-memory-with-side.html).

```
Spectre attack (simplified):

1. Attacker trains branch predictor to predict "taken"
2. Victim executes:
     if (x < array_size) {          // bounds check
         y = array2[array1[x] * 256]; // speculated if x is out-of-bounds
     }
3. CPU speculatively loads array1[x] (secret byte), uses it as index into array2
4. Speculation is rolled back — but array2 cache state persists
5. Attacker times access to array2 → deduces the secret byte

```

Mitigation behavior and cost depend on the processor, kernel, enabled controls, and workload. Use platform security advisories and the current OS documentation; a fixed percentage beside a mitigation name is not a transferable performance estimate.

**AI hardware impact:** attack surfaces differ across CPU, GPU, driver, and shared-service boundaries. SIMT is not a blanket side-channel immunity claim. Consult the exact platform advisories and measure mitigation overhead on the deployed workload.

---

## 12. Labs

### Lab 0 — A reference before profiling

Complete [Numbers, shapes, and bytes](../Labs/Numbers-Shapes-and-Bytes.md) to establish a numerical reference, negative cases, and an analytical traffic boundary. It runs on a CPU with no third-party packages.

### Lab 1 — Cache Miss Profiling

**Goal:** measure how access pattern affects cache performance. The following commands are **exercise templates**, not a supplied `matrix_transpose` binary or a tested installation recipe. Package names and available performance events depend on distribution, kernel, and CPU; check `perf list` and permissions.

```bash
# Install perf and valgrind
sudo apt install linux-perf valgrind

# L1/L2/L3 miss rates for matrix transpose (row-major vs column-major)
perf stat -e cache-references,cache-misses,L1-dcache-misses \
          ./matrix_transpose 1024

# Cache simulation
valgrind --tool=cachegrind --cache-sim=yes ./matrix_transpose 1024
cg_annotate cachegrind.out.*
```

Write a 1024×1024 float matrix. Implement:
- Row-major transpose (good spatial locality for reads, bad for writes)
- Column-major transpose (bad for reads)
- Tiled transpose (start with 32×32 blocks) — predict and measure whether it improves locality

**Deliverable:** table of L1/L2/L3 miss rates for each version; explain the difference.

### Lab 2 — SIMD Vectorisation

**Goal:** measure scalar vs SIMD vs compiler-auto-vectorised throughput on a dot product kernel.

```cpp
// Compile and compare:
// g++ -O3 -march=native -o dot dot.cpp
// g++ -O3 -fno-tree-vectorize -o dot_scalar dot.cpp
// GCC exercise templates; inspect optimization reports and generated instructions.
// Manual intrinsics version (see section 5.2)
```

Benchmark a size sweep, including sizes not divisible by eight, and verify outputs against the scalar reference. Use compiler vectorization reports and disassembly to inspect instructions; generic `perf stat` counters do not prove vectorization. Record warmup, repetitions, cache assumptions, and timing boundaries. Compare optimized builds so the claimed SIMD effect is not simply an `-O0` versus `-O3` effect.

**Deliverable:** bandwidth (GB/s) for each version; compare to DRAM bandwidth ceiling.

### Lab 3 — Roofline on a Real GPU

**Goal:** compute arithmetic intensity of three kernels and place them on the roofline.

```bash
# Use NVIDIA Nsight Compute CLI
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              l1tex__t_bytes.sum,\
              sm__sass_thread_inst_executed_op_fadd_pred_on.sum \
    python train_step.py

# Kernels to measure:
# 1. Element-wise ReLU (low AI — memory-bound)
# 2. Batch matrix multiply (high AI — compute-bound)
# 3. Softmax (medium AI)
```

**Deliverable:** roofline plot with three kernels marked; conclude which optimisations would help each.

### Lab 4 — Architecture ISA Comparison

**Goal:** compare code generation and performance across ISAs.

```bash
# Cross-compile the same matrix multiply kernel for three targets:
gcc -O3 -mavx2 -mfma -o matmul_x86 matmul.c           # x86-64 with AVX2
aarch64-linux-gnu-gcc -O3 -o matmul_arm matmul.c        # ARM64 with NEON
riscv64-linux-gnu-gcc -O3 -march=rv64gcv -o matmul_rv matmul.c  # RISC-V

# Inspect generated assembly:
objdump -d matmul_x86  | grep -E "vmulps|vfmadd"
objdump -d matmul_arm  | grep -E "fmla|fmul"
```

**Deliverable:** code size, selected instructions, and reference-checked outputs for each target. Use each target toolchain's `objdump` (or a compatible multi-target disassembler). QEMU can support functional testing; emulated timing is not native processor performance. Record native GFLOP/s only on the named hardware and mark unavailable measurements “not run.”

### Lab 5 — Branch Predictor Simulator

**Goal:** implement and compare branch prediction strategies.

```python
class TwoBitPredictor:
    def __init__(self):
        self.table = {}   # branch_addr → 2-bit counter (0–3)

    def predict(self, addr):
        return self.table.get(addr, 1) >= 2   # True = predict taken

    def update(self, addr, taken):
        c = self.table.get(addr, 1)
        if taken:
            self.table[addr] = min(c + 1, 3)
        else:
            self.table[addr] = max(c - 1, 0)
```

Feed branch traces from real programs (gcc, matrix multiply, sort). Compare:
- Always taken, always not-taken
- 1-bit predictor
- 2-bit saturating counter
- Correlating predictor (GHR + PHT)

**Deliverable:** accuracy table for each predictor on a stated trace and initialization; explain which pattern helps or hurts each predictor. Correlating prediction is not guaranteed to win on every nested loop.

### Lab 6 — Access Pattern and Requested-Byte Rate

**Goal:** distinguish a program's requested bytes from physical cache-line and DRAM traffic.

For an array with `N` elements, positive stride `s` measured in **elements**, and `R` repetitions, the number of reads is `R × ceil(N/s)`. Compute requested bytes as that count times the element size, then divide by elapsed seconds. Counting the full array size for every strided pass overstates the rate when elements are skipped.

**Exercise / pseudocode:**

```text
initialize the array outside the timed region
choose N, stride_in_elements, repetitions, and an unsigned checksum type
start a monotonic timer
repeat R times:
    for i = 0; i < N; i += stride_in_elements:
        checksum += data[i]
stop the timer and preserve checksum so the computation remains observable
requested_GB_per_second = reads × bytes_per_element / elapsed_seconds / 1e9
```

A serial checksum dependency may limit this exercise before memory bandwidth does; inspect generated code and use independent accumulators or an established streaming benchmark when estimating sustained bandwidth. Signed integer overflow is not an acceptable way to prevent optimization.

Sweep working-set sizes relative to the actual processor's cache capacities, and sweep stride separately. A stride of 16 four-byte elements is 64 bytes; it is not 16 cache lines. Report requested-byte rate, checksum, elapsed time, and compiler options. Hardware counters are needed to estimate traffic at a particular cache or DRAM boundary. Cache transitions may be visible, but clean plateaus are not guaranteed.

### Lab 7 — OoO CPU Simulator (Capstone)

**Goal:** build a simplified out-of-order CPU simulator; measure IPC and compare to in-order.

Implement:
- In-order baseline (5-stage pipeline with stalls)
- OoO with ROB (32 entries), reservation stations (16 entries), 2 ALUs + 1 MUL + 1 LSU
- Register renaming (64 physical registers, 32 architectural)
- Forwarding via CDB

Test on instruction traces: matrix multiply, Fibonacci, quicksort.

**Deliverable:**
- IPC comparison: in-order vs OoO across benchmarks
- Execution timeline visualisation
- Identify which benchmark benefits most from OoO and why

---

## 13. ISA and Architecture Across the AI Stack

| Layer in your roadmap | Relevant ISA/arch |
|----------------------|-------------------|
| PyTorch model training | x86-64 (AVX-512) + CUDA (H100/A100) |
| ML compiler backends | ARM64, x86-64, RISC-V, GPU PTX |
| FPGA accelerator control | ARM Cortex-M/A (Zynq PS) |
| Jetson edge deployment | ARM64 (Cortex-A78AE) + NVIDIA Ampere |
| Custom AI chip design | RISC-V (control core) + custom dataflow |
| Mobile inference | ARM64 (Neural Engine, Hexagon) |

**Key takeaway:** Memory hierarchy and arithmetic intensity bound performance; the compiler and implementation determine how closely you approach the bound. With an assumed 3.35 TB/s roof, 10 FLOP/byte gives an upper bound of 33.5 TFLOP/s, not a guaranteed result. Python may invoke an efficient compiled kernel or introduce substantial overhead; the execution path matters.

---

## 14. Where This Takes You

```
This section
      │
      ├── Phase 1 §3 — Operating Systems
      │     Memory management, virtual memory, MMU (ties to TLB section above)
      │     Device drivers (how OS talks to your GPU/FPGA)
      │
      ├── Phase 1 §4 — C++ and Parallel Computing
      │     SIMD intrinsics in practice, OpenMP, CUDA kernels
      │     The CUDA memory model directly mirrors the GPU hierarchy in §6
      │
      ├── Phase 4A — Xilinx FPGA
      │     Implement your own systolic array in SystemVerilog
      │     Understand timing and pipeline stages at the RTL level
      │
      ├── Phase 4B — NVIDIA Jetson
      │     ARM64 host CPU + Ampere GPU — both covered here
      │
      └── Phase 5F — AI Chip Design
            Design a custom accelerator using systolic array / dataflow patterns
            Roofline model guides your PE count and memory bandwidth decisions
```

---

## Resources

### Check your understanding before moving on

1. **Does 167 FLOP/byte guarantee that a matrix multiply is compute-bound?** No. Compare with the selected compute/bandwidth ridge. With the illustrative 295 FLOP/byte ridge it lies below the compute roof, and other constraints may reduce performance further.
2. **A workload spends 2% of baseline time in serial work. What is its ideal speedup on four cores?** `1 / (0.02 + 0.98/4) ≈ 3.77`. Use time fractions, not the percentage of source-code lines.
3. **Does a 32-thread warp mean every thread issues an instruction every clock?** No. Dependencies, scheduling, masking, and execution resources determine issue and completion.
4. **Why test a vector dot product at N=9?** It exercises a full eight-element vector plus a scalar tail. Testing only multiples of eight hides an out-of-bounds bug.
5. **A strided loop touches every sixteenth four-byte element. Can you count every array byte as read?** No. Count requested elements separately from transferred cache lines; use hardware evidence for a physical memory boundary.

If one answer is uncertain, repeat that small example before adding a more complex accelerator. For the phase artifact, keep the raw measurements and explain one hypothesis that the data rejected.

### Further reading

| Resource | What for |
|----------|----------|
| **Patterson & Hennessy — Computer Organization and Design (ARM edition)** | Pipeline, cache, memory — the textbook standard |
| **Hennessy & Patterson — Computer Architecture: A Quantitative Approach** | Deep OoO, superscalar, memory system design |
| **NVIDIA H100 Architecture Whitepaper** | SM internals, Tensor Core specs, NVLink |
| **"What Every Programmer Should Know About Memory" — Ulrich Drepper** | Cache hierarchy, NUMA, prefetching (free PDF) |
| **"Roofline: An Insightful Visual Performance Model" — Williams et al. (2009)** | Original roofline paper — 10 pages, essential |
| **Chips and Cheese (blog)** | Reverse-engineered microarchitecture analysis (AMD, Intel, Apple) |
| **Wikichip** | Die shots, cache sizes, core counts, process nodes |
| **NVIDIA Nsight Compute** | GPU roofline and memory hierarchy profiling |
| **"Computer Systems: A Programmer's Perspective" — Bryant & O'Hallaron** | Linking, virtual memory, caching from the programmer's view |

## Turn the operator trace into a memory budget

For an FP16 `1024×1024` matrix-vector product, weights alone occupy 2 MiB and the operation performs about 2.10 million FLOPs. With no weight reuse, that is about 1 FLOP per weight byte; add input/output and metadata traffic before comparing with measurements. Reusing weights across a batch changes this model; increasing peak MAC count alone does not.

Make a worksheet for weights, activations, scratchpad tiles, DMA bursts, and partial sums. Give separate estimates for cycles, external bytes, and on-chip capacity. Then measure the software reference and explain the gap. Hardware-product tables elsewhere in the guide are examples, not procurement advice or a substitute for a SKU-specific datasheet.

**Carry forward:** A shape-specific traffic/capacity worksheet and predicted limit. Continue with [the next course](../3.%20Operating%20Systems/Guide.md) and reuse that evidence.
