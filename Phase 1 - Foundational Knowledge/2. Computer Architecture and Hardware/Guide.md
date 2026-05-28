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

> **Goal:** By the end of this section you can read a GPU or NPU architecture paper, understand why a transformer is memory-bandwidth-bound, and reason about the trade-offs in any custom accelerator design.

---

## 1. The Two Computing Paradigms

Every chip designer makes the same fundamental choice: optimise for **latency** or **throughput**. CPUs and GPUs represent the two extremes.

```
CPU — latency optimised                 GPU — throughput optimised
────────────────────────────────────    ────────────────────────────────────
Few, powerful cores (4–128)             Thousands of simple cores (1,000s–10,000s)
Deep OoO execution engine               In-order, warp-scheduled execution
Large private caches (L1: 32–64 KB)     Small per-SM caches, huge shared pool
Branch prediction, speculative exec     No branch prediction (divergence penalty)
Single-thread latency: 1–4 ns          Single-thread latency: 50–100 ns
Total throughput: 1–10 TFLOPS           Total throughput: 100–1,000 TFLOPS
Power: 5–350 W                          Power: 70–1,000 W

Use-case: operating system, compilers,  Use-case: matrix multiply, convolution,
  database queries, anything with         attention — embarrassingly parallel,
  complex control flow                    regular data access patterns
```

**The transistor budget question:** A modern chip has ~50–100 billion transistors. How do you spend them?

```
CPU (Apple M4 Pro die)               GPU (NVIDIA H100 SXM)
┌──────────────────────────┐         ┌──────────────────────────┐
│ OoO engines (P-cores) 45%│         │ CUDA/Tensor cores    60% │
│ Cache hierarchy       30%│         │ HBM memory interface 20% │
│ Memory controller      8%│         │ L2 cache             10% │
│ I/O, NPU, other       17%│         │ Control / scheduler  10% │
└──────────────────────────┘         └──────────────────────────┘
→ Optimises for the 1 hard task       → Optimises for 10,000 easy tasks
```

**AI connection:** Transformer attention is `O(n²)` parallel multiply-accumulate — exactly what the GPU transistor budget is built for. The CPU's OoO engine is wasted on this workload; the GPU's thousands of Tensor Cores are perfect.

---

## 2. Instruction Set Architecture (ISA)

The ISA is the **contract between software and hardware**. Hardware can be redesigned completely (new microarchitecture) as long as the ISA stays compatible — software keeps running.

### RISC vs CISC

| Property | RISC (ARM64, RISC-V) | CISC (x86-64) |
|----------|----------------------|---------------|
| Instruction length | Fixed 32 bits | Variable 1–15 bytes |
| Memory access | Load-store only | Any instruction can access memory |
| Decoder complexity | Simple | Complex (internal RISC micro-ops) |
| Code density | Lower | Higher |
| Power efficiency | Excellent | Good |
| AI hardware usage | Edge, mobile, Apple Silicon, Jetson | Data-centre training servers |

> In practice, modern x86 CPUs decode CISC instructions into RISC-like micro-ops internally — the distinction blurs at the microarchitecture level.

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
32 general-purpose 64-bit registers

add  x1, x2, x3        // x1 = x2 + x3
vsetvli t0, a0, e32,m4 // V extension: set vector length, FP32, 4x LMUL
vfmacc.vv v0, v4, v8   // vector FMA (scalable width)
```

Used in: Emerging edge AI chips (Tenstorrent, SiFive), custom accelerator control cores.

### ISA Comparison

| Feature | ARM64 | x86-64 | RISC-V |
|---------|-------|--------|--------|
| Registers (GP) | 31 × 64-bit | 16 × 64-bit | 31 × 64-bit |
| Instruction size | Fixed 32-bit | Variable 1–15 bytes | Fixed 32-bit (base) |
| SIMD width | 128-bit NEON, scalable SVE2 | 128/256/512-bit | Scalable RVV |
| Endianness | Little (bi in theory) | Little | Little |
| Addressing modes | ~9 | ~20+ | ~4 (simple) |
| Licensing | ARM license fee | Intel/AMD proprietary | Open (free) |
| Code density | Good | Best | Moderate |

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

**AI connection:** GPU warps execute in lockstep — there is no branch prediction. If threads in a warp take different branches, they execute **both paths serially** (warp divergence). Good kernel design eliminates branches inside hot loops.

### 3.3 Superscalar Execution

Fetch and issue **multiple instructions per cycle** — wider pipeline, more parallelism:

```
4-wide superscalar (modern x86/ARM):

Cycle 1: Fetch 4 instructions
Cycle 2: Decode 4 → 4 micro-ops
Cycle 3: Issue up to 4 to execution units (if no dependencies)
Cycle 4: Multiple ALUs, FP units, load/store units execute in parallel

Ideal: 4 IPC (limited by dependencies, memory stalls, branch mispredictions)
Real: 2–3.5 IPC on typical code (Apple M4 P-core: ~3+ IPC sustained)
```

**Superscalar widths in practice:**

| CPU | Decode width | Issue width | Peak IPC |
|-----|-------------|-------------|----------|
| ARM Cortex-A510 (E-core) | 3 | 3 | 3 |
| ARM Cortex-A720 (P-core) | 5 | 5 | 5 |
| Apple M4 P-core | 8 | 8 | ~6+ |
| AMD Zen 5 | 6 | 6 | ~5 |
| Intel Golden Cove | 6 | 6 | ~5 |

Beyond 6–8 wide, the dispatch logic complexity and diminishing ILP make wider designs impractical for general-purpose code.

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

| Structure | Role | Size (modern CPU) |
|-----------|------|--------------------|
| **Reorder Buffer (ROB)** | Holds all in-flight instructions; enforces in-order commit | 256–512 entries |
| **Reservation Stations** | Instructions wait here until operands are ready | 64–128 entries |
| **Register Renaming** | Maps architectural → physical registers, eliminates false deps | 256–384 physical regs |
| **Common Data Bus (CDB)** | Broadcasts results to all waiting reservation stations | 6–8 buses |
| **Load/Store Queue** | Tracks memory operations; enforces ordering | 64–128 entries |

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

Modern predictors achieve 94–98% accuracy on typical code. Every misprediction flushes the pipe — 10–20 cycles wasted on a modern deep pipeline.

**2-bit saturating counter** — simplest dynamic predictor:

```
States: Strongly Not Taken (00) → Weakly Not Taken (01) →
        Weakly Taken (10) → Strongly Taken (11)

Must mispredict TWICE to switch direction.
Good for loops: only mispredicts on entry and exit (vs 1-bit: every exit).
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

**TAGE (Tagged Geometric History Length)** — state-of-the-art in modern CPUs:

```
Multiple predictor tables with geometrically increasing history lengths:
  Table 0: bimodal (no history)
  Table 1: 8-cycle history
  Table 2: 32-cycle history
  Table 3: 128-cycle history
  Table 4: 512-cycle history

Prediction = table with longest matching history
Accuracy: 97–98% on SPEC benchmarks
Used in: AMD Zen, Intel Golden Cove, ARM Neoverse
```

**Additional predictor hardware:**

| Structure | Purpose |
|-----------|---------|
| **Branch Target Buffer (BTB)** | Caches `branch PC → target address` (no need to decode to know target) |
| **Return Stack Buffer (RSB)** | Stack of return addresses — predicts `ret` targets with ~100% accuracy |
| **Indirect Branch Predictor** | Predicts target of `jmp [reg]` (virtual function calls) |
| **Loop Predictor** | Counts iterations — predicts loop exit after N iterations |

**Cost of misprediction:**

```
5-stage pipeline: flush 2 instructions → 2-cycle penalty
20-stage pipeline (modern OoO): flush 10–20 instructions → 10–20 cycle penalty
4-wide superscalar × 15 cycles = 60 wasted instruction slots per mispredict

If 5% of instructions are branches with 96% accuracy:
  Mispredict rate = 5% × 4% = 0.2% of all instructions
  CPI penalty = 0.2% × 15 cycles = 0.03 CPI  (seems small, but adds up)
```

---

## 4. Memory Hierarchy — The Real Bottleneck

Memory access time is the dominant constraint for AI workloads. A GPU with 1,000 TFLOPS of compute can be throttled to 10 TFLOPS effective throughput by insufficient memory bandwidth.

### 4.1 The Memory Mountain

```
Level          | Latency    | Bandwidth     | Capacity  | Location
───────────────────────────────────────────────────────────────────
Registers      | 0 cycles   | unlimited     | ~KB       | on-core
L1 cache       | 4 cycles   | ~1 TB/s       | 32–64 KB  | per-core
L2 cache       | 12 cycles  | ~500 GB/s     | 256 KB–4 MB  per-core/cluster
L3 cache       | 40 cycles  | ~200 GB/s     | 8–64 MB   | shared (CPU)
               |            |               |           |
DRAM (DDR5)    | 70 ns      | 50–100 GB/s   | 16–512 GB | off-chip
HBM3 (GPU)     | 100 ns     | 3.35 TB/s     | 80–192 GB | off-chip stacked
NVMe SSD       | ~100 µs    | 7 GB/s        | TB        | storage
```

**The gap is enormous.** Registers are 10,000× faster than DRAM. Cache hierarchy exists entirely to bridge this gap.

### 4.2 Cache Organisation

A cache is divided into **sets** of **ways**. An address maps to exactly one set; within the set, it can go into any way.

```
Address bits:  [  Tag  ] [  Index  ] [ Offset ]
                   ↓          ↓
               Compare    Select set   Byte within line

8-way set-associative example (64-byte lines, 8 MB L3):
  Sets = 8 MB / (8 ways × 64 bytes) = 16,384 sets
  Index bits  = log2(16,384) = 14
  Offset bits = log2(64)     = 6
  Tag bits    = 64 - 14 - 6  = 44
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

**Why this matters for AI:** In multi-GPU systems, each GPU has its own HBM. Coherence is handled by NVLink/NVSwitch (NVIDIA) or Infinity Fabric (AMD). Understanding MESI is the mental model for understanding why all-reduce operations are expensive and why NCCL ring-allreduce is designed the way it is.

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
4 KB pages: 4 GB working set = 1M pages → TLB covers only 64 × 4 KB = 256 KB
2 MB pages: 4 GB working set = 2K pages → TLB covers 64 × 2 MB = 128 MB

For AI: HBM-resident model weights benefit from huge pages (fewer TLB misses on GPU)
Linux: echo 1024 > /proc/sys/vm/nr_hugepages
```

### 4.5 DRAM and HBM

**DDR5 (CPU memory):**
- Bandwidth: ~50–100 GB/s per channel (8 channels on EPYC = 800 GB/s)
- Latency: ~70–80 ns
- Capacity: up to 512 GB per socket

**HBM3 (GPU memory) — the key innovation for AI:**

```
Traditional GDDR:            HBM (High Bandwidth Memory):
  Package ─── PCB ──── GPU     GPU die
                               ━━━━━━━━━
                               ▓▓▓▓▓▓▓▓  ← HBM stack (stacked DRAM dies)
                               ━━━━━━━━━
                               Silicon interposer (CoWoS)

Bus width: 32 bits            Bus width: 1,024 bits per stack
Bandwidth: ~600 GB/s          Bandwidth: 3.35 TB/s (H100 SXM)
```

HBM achieves 5× higher bandwidth than GDDR because it uses a **wide, short bus** (1,024 bits × multiple stacks) instead of a narrow, long one. The GPU die and HBM stacks sit on the same silicon interposer — this is CoWoS (Chip-on-Wafer-on-Substrate) packaging.

**AI connection:** LLM inference is almost entirely memory-bandwidth-bound. The rate at which you can load model weights from HBM determines tokens/second — compute is secondary.

---

## 5. SIMD — One Instruction, Many Data

SIMD is the bridge between CPU vector units and GPU Tensor Cores. The same idea — execute one operation on a wide register of packed data — scales from 128-bit NEON to 512-bit AVX-512 to a 32-wide GPU warp.

### 5.1 How SIMD Works

```
Scalar (no SIMD):                   SIMD (4-wide FP32):
  a[0] = b[0] * c[0]   4 cycles       [a0 a1 a2 a3] = [b0 b1 b2 b3]
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
| x86-64 | AVX2 | 256 bits | 8 | Yes (FMA3) |
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
    for (int i = 0; i < N; i += 8) {
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
    return _mm_cvtss_f32(lo);
}
// Speedup vs scalar: ~6–7× (limited by memory bandwidth for large N)
```

### 5.3 SIMD → GPU SIMT

A GPU warp (32 threads) executing the same instruction at the same time is SIMD taken to 32-wide, with the additional twist that each "lane" is an independent thread with its own register state.

```
CPU SIMD (AVX-512):              GPU SIMT (warp of 32):
  1 instruction                    1 instruction
  16 FP32 lanes                    32 "lanes" = 32 threads
  all same operation                all same operation (PC)
  same data register               each thread has own registers
  no divergence possible           divergence possible but costly
```

Tensor Cores go further — a single `wmma::mma_sync` instruction operates on an entire 16×16×16 matrix fragment, effectively 4,096-wide for FP16.

---

## 6. GPU Architecture (Conceptual)

The CUDA programming model is covered in Section 4. Here we look at the **hardware** — what the silicon actually does.

### 6.1 SM: The Streaming Multiprocessor

The GPU is a collection of SMs. Every kernel launch distributes thread blocks across available SMs.

```
One SM (NVIDIA Ampere A100):
┌─────────────────────────────────────────────────────────┐
│  4 × Warp Schedulers (issue 1 warp/cycle each)          │
│                                                         │
│  4 × Dispatch Units                                     │
│  ───────────────────────────────────────────────────    │
│  64 × CUDA Cores (FP32)   │  32 × INT32 cores          │
│   4 × Tensor Core units   │   4 × FP64 cores           │
│  16 × Load/Store units    │   4 × SFU (sin, cos, etc.) │
│  ───────────────────────────────────────────────────    │
│  192 KB unified shared memory / L1 cache (configurable) │
│  256 KB register file (32-bit registers per SM)         │
│                                                         │
│  Max 1,536 resident threads (48 warps × 32 threads)     │
└─────────────────────────────────────────────────────────┘
```

A100 has 108 SMs → 108 × 64 = 6,912 CUDA cores; 108 × 4 = 432 Tensor Core units.

### 6.2 SIMT Execution and Warp Scheduling

All 32 threads in a warp execute the **same instruction** every cycle. When a warp stalls (e.g., waiting for a global memory load that takes 300+ cycles), the warp scheduler **instantly switches** to another ready warp — zero overhead context switch because each warp has its own dedicated register file.

```
Cycle 100: Warp 0 issues LOAD (takes 200+ cycles to return)
Cycle 101: Warp scheduler picks Warp 1 (compute-bound, no stall)
Cycle 102: Warp 2
Cycle 103: Warp 3
...
Cycle 300: Warp 0's data arrives; it becomes eligible again
```

This **latency hiding through massive parallelism** is the GPU's secret: it tolerates 300-cycle memory latency by having 40+ other warps to run while waiting. The CPU hides latency with 10 MB of cache; the GPU hides latency with thousands of threads.

### 6.3 GPU Memory Hierarchy

```
Thread-private:
  Registers (fastest)      — local variables, per-thread, ~255 registers each
  Local memory (slowest)   — register spill → goes to global memory

Block-shared:
  Shared memory (fast)     — 16–164 KB per SM, programmer-managed, ~4–10 cycles
  L1 cache (fast)          — same SRAM bank as shared memory, automatic

Device-wide:
  L2 cache                 — 40–50 MB on H100, shared across all SMs, ~100 cycles
  Global memory (HBM)      — 80–192 GB, 3.35 TB/s, ~300–400 cycles
  Constant memory          — 64 KB, cached in L1, read-only
  Texture memory           — spatial locality cache, hardware interpolation
```

**Shared memory is the programmer's L1.** Tiled matrix multiplication explicitly loads a tile from global memory into shared memory, so each value is read from HBM once but used many times by threads in the block.

### 6.4 Tensor Cores

Tensor Cores are hardwired matrix-multiply-accumulate (MMA) units that compute a 16×16×16 matrix product in a single "instruction":

```
D = A × B + C
  A: 16×16 matrix (FP16 or BF16)
  B: 16×16 matrix (FP16 or BF16)
  C: 16×16 accumulator (FP32)
  D: 16×16 result (FP32)

Throughput on H100 SXM:
  FP16  Tensor Core: 989  TFLOPS
  BF16  Tensor Core: 989  TFLOPS
  FP8   Tensor Core: 1,979 TFLOPS
  FP32  CUDA Core:   67   TFLOPS  ← 14.7× slower!
```

Every major AI training and inference framework ultimately generates `cublasSgemm` / `cublasHgemm` calls that map to Tensor Core wmma instructions.

---

## 7. AI Accelerator Design Patterns

### 7.1 Systolic Arrays

A systolic array is a grid of identical processing elements (PEs) where data flows rhythmically through neighbours. Google's TPU uses a 256×256 systolic array.

```
Input matrix A rows flow →→→→→→→→
Weight matrix B cols flow ↓↓↓↓↓↓↓

     PE(0,0) → PE(0,1) → PE(0,2) → ...
       ↓         ↓         ↓
     PE(1,0) → PE(1,1) → PE(1,2) → ...
       ↓         ↓         ↓
     PE(2,0) → PE(2,1) → PE(2,2) → ...

Each PE: accumulator += A_val × B_val
Result flows out the bottom

Advantage: weights loaded once, reused across entire array → minimal memory traffic
```

**NVIDIA Tensor Cores are essentially a 16×16 systolic array** implemented in SRAM-adjacent logic.

### 7.2 Dataflow Architectures

Traditional CPUs/GPUs: **von Neumann** — load operands → execute → store result → repeat. Memory traffic dominates.

Dataflow: operations fire as soon as operands arrive. No central memory round-trip.

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

Dedicated inference accelerators in mobile/edge SoCs:

| Chip | NPU | Peak INT8 | Architecture |
|------|-----|-----------|--------------|
| Apple M4 | 16-core Neural Engine | 38 TOPS | Apple proprietary |
| Qualcomm Snapdragon X Elite | Hexagon NPU | 45 TOPS | Qualcomm proprietary |
| Intel Core Ultra 200V | AI Boost NPU | 48 TOPS | Intel proprietary |
| Google Tensor G4 | TPU v5 lite | ~30 TOPS | Systolic array |
| NVIDIA Jetson Orin NX | DLA v2 | 57 TOPS (DLA+GPU) | Mixed |

NPUs trade flexibility for efficiency: they are optimised for a fixed set of layer types (conv, matmul, elementwise) at reduced precision (INT8/FP16), consuming 10–50× less power than a discrete GPU for the same throughput.

---

## 8. The Roofline Model — Your Most Important Analysis Tool

The roofline model answers the single most important question about any kernel or workload: **are you memory-bound or compute-bound?**

### 8.1 Arithmetic Intensity

```
Arithmetic Intensity (AI) = FLOPs executed / Bytes transferred from memory

Examples:
  Vector addition  y = a + b:        1 FLOP / 12 bytes = 0.08 FLOP/byte  (memory-bound)
  Matrix multiply  C = A × B (N=1K): 2N³ FLOPs / 3N² × 4 bytes          (compute-bound)
    = 2×10⁹ / 12×10⁶ ≈ 167 FLOP/byte
  Transformer attention (seq 2048):  ~10 FLOP/byte                        (memory-bound)
  LLM weight loading (batch=1):      ~0.5 FLOP/byte                       (severely memory-bound)
```

### 8.2 The Roofline

```
TFLOPS (log)
 │                          ───────────────── Peak compute (989 TFLOPS FP16)
 │                      ╱
 │                  ╱  ╲── compute-bound region
 │              ╱        (more compute = more speed)
 │          ╱
 │      ╱ ← memory-bandwidth roof slope = HBM BW × AI
 │  ╱       (memory-bound region: more bandwidth = more speed)
 └──────────────────────────────────────────── Arithmetic Intensity (FLOP/byte)
         ↑
    Ridge point ≈ 989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/byte (H100)

Any kernel with AI < 295 FLOP/byte is memory-bandwidth-limited on H100.
Transformer attention (~10 FLOP/byte) is 29× below the ridge point.
```

### 8.3 Where AI Workloads Live

```
AI (FLOP/byte)      Workload                    Bound
─────────────────────────────────────────────────────────
0.5                 LLM decoding (batch=1)       Severely memory-bound
2–5                 Batch norm, layer norm        Memory-bound
10–30               Transformer attention         Memory-bound
50–100              Small batch matmul            Borderline
>295                Large matmul (batch≥64)       Compute-bound
```

**Practical implication:** Improving LLM inference token rate requires:
1. Larger batches (increase arithmetic intensity)
2. More HBM bandwidth (H100 → H200: 3.35 → 4.8 TB/s)
3. Weight quantisation (INT4 = 2× bandwidth; INT8 = same compute but half the bytes)

Not a faster GPU clock.

---

## 9. Performance Analysis and Profiling

### 9.1 Amdahl's Law

Not all code parallelises. Amdahl's Law predicts the maximum speedup from parallel execution.

```
Speedup = 1 / ((1 − P) + P/N)

P = fraction of code that is parallel
N = number of cores/processors
```

**Example (2% serial, 98% parallel):**

| Cores | Speedup | Efficiency |
|-------|---------|------------|
| 1     | 1.0×    | 100%       |
| 4     | 3.5×    | 87%        |
| 16    | 10.9×   | 68%        |
| 64    | 28.6×   | 45%        |
| ∞     | 50.0×   | → 0%       |

The serial 2% becomes the bottleneck. No amount of cores can exceed 50× speedup.

**For AI workloads:**
- **Data parallelism (batching):** near-linear scaling if minimal synchronisation
- **Model parallelism (split across GPUs):** limited by all-reduce communication
- **Pipeline parallelism:** scales linearly if stages are balanced
- **Real-world multi-GPU scaling:** 80% efficiency typical; doubling GPUs → expect ~1.6–1.8× throughput

### 9.2 Profiling Tools

**CPU profiling (Linux):**

```bash
# Hardware counter sampling
perf stat -e cache-references,cache-misses,L1-dcache-misses,branches,\
          branch-misses,instructions,cycles ./program

# Record + flamegraph
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
- Disable CPU turbo boost for reproducibility
- Pin threads to cores (`taskset -c 0-7 ./program`)
- Warm up (run once before timing)
- Report median of N runs (not mean — skew from outliers)

### 9.3 Bottleneck Diagnosis

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| Using <20% peak FLOPS | Memory-bound | Increase arithmetic intensity (batch, tile, fuse) |
| High L3 cache miss rate | Poor locality | Tile/block, change data layout, prefetch |
| Low IPC (<1.5 on OoO CPU) | Data dependencies | Reorder code, unroll loops, reduce dependency chains |
| High branch mispredict % | Irregular control | Branchless code (`cmov`), predication, sort input |
| GPU SM occupancy <50% | Register/shared mem pressure | Reduce per-thread registers, smaller blocks |
| GPU memory throughput low | Uncoalesced access | Ensure threads in warp access contiguous addresses |

**Case study — naive vs tiled matrix multiply:**

```c
// Naive: ~50 GFLOPS (memory-bound, AI ≈ 0.25 FLOP/byte)
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];

// Tiled with 64×64 blocks: ~450 GFLOPS (compute-bound, AI ≈ 85 FLOP/byte)
// 9× improvement from cache locality — not SIMD, not clock speed.
```

---

## 10. Real-World Case Studies

### 10.1 Apple M4

```
Performance Cores (P-cores): 4× custom (wide OoO, ~8-wide decode)
  L1: 192 KB (64 KB I$, 128 KB D$)
  L2: 12 MB per 2 cores

Efficiency Cores (E-cores): 4× custom
  L1: 128 KB, L2: 4 MB per 2 cores

System Cache (L3): 20 MB shared
Memory: LPDDR5x unified (up to 32 GB, ~120 GB/s)
GPU: 10-core, ~4.3 TFLOPS FP32
Neural Engine: 16-core, 38 TOPS INT8
Process: TSMC N3E, ~20B transistors, 120 mm²
```

**Key design choices:**
- **Unified memory** — CPU, GPU, and Neural Engine share the same LPDDR5x pool. No PCIe copies between CPU and GPU memory (unlike discrete GPU systems). This makes inference on Apple Silicon uniquely efficient for models that fit in RAM.
- **Wide P-cores** — ~8-wide decode, 6+ IPC. Apple invests in single-thread performance because macOS/iOS workloads (UI, compilers, browsers) demand low latency.
- **Efficiency-first design** — 20–30W total package. The entire M4 SoC uses less power than a single AMD desktop core under load.

### 10.2 AMD Ryzen 9 9950X (Zen 5)

```
Cores: 16× Zen 5 (OoO, 6-wide superscalar)
  L1: 32 KB I$ + 32 KB D$ per core
  L2: 1 MB per core
  L3: 32 MB total (16 MB per CCX, 2 CCX chiplets)

Memory: Dual-channel DDR5-5600 (up to 192 GB)
Process: TSMC N4, Socket AM5
TDP: 170W, boost up to 5.7 GHz
```

**Key design choices:**
- **Chiplet architecture** — two 8-core CCD chiplets + separate I/O die (IOD). Chiplets are smaller (higher yield), and the IOD can use an older, cheaper process. This is how AMD competes with Intel on cost.
- **3D V-Cache option** (9950X3D variant) — extra 64 MB L3 cache stacked vertically on top of CCD. Dramatically reduces gaming/simulation workloads that are L3-miss-bound.
- **Per-core boost** — individual cores can boost to 5.7 GHz independently. Workload-adaptive power management maximises single-thread or multi-thread performance as needed.

### 10.3 Qualcomm Snapdragon X Elite (ARM-based PC)

```
Cores: 12× Oryon custom ARM (OoO, wide superscalar, Nuvia-derived)
  L1: 64 KB I$ + 64 KB D$ per core
  L2: 1 MB per core, L3: 12 MB shared

GPU: Adreno X1 (~3.8 TFLOPS FP32)
NPU: Hexagon, 45 TOPS INT8
Memory: LPDDR5x-8448 (up to 64 GB)
Process: TSMC N4, TDP 12W base / 30W sustained
```

**Key design choices:**
- **ARM for laptops** — demonstrates that ARM can match x86 single-thread performance while consuming 5–10× less power. First serious competitor to Apple Silicon on Windows.
- **On-device NPU** — 45 TOPS enables on-device LLM inference (Phi-2, Llama 7B quantised), voice recognition, and image generation without cloud connectivity.
- **Software compatibility** — Windows Prism emulates x86 applications on ARM64 with ~80–90% native performance. Native ARM64 apps run at full speed.

### 10.4 Architecture Comparison

| Metric | Apple M4 | AMD 9950X | Snapdragon X Elite |
|--------|----------|-----------|-------------------|
| ISA | ARM64 | x86-64 | ARM64 |
| Cores | 4P + 4E | 16 | 12 |
| Single-thread | ~2500 (GB6) | ~2700 (GB6) | ~2400 (GB6) |
| Multi-thread | ~10000 (GB6) | ~21000 (GB6) | ~9500 (GB6) |
| TDP | 20–30W | 170W | 12–30W |
| Perf/Watt | Excellent | Moderate | Excellent |
| NPU | 38 TOPS | None | 45 TOPS |
| Memory BW | ~120 GB/s | ~89 GB/s (DDR5) | ~135 GB/s |

**Takeaway:** Apple M4 and Snapdragon X Elite show that ARM64 + unified memory + integrated NPU is the future of edge AI. AMD Zen 5 dominates in multi-core throughput for server/workstation workloads. The ISA matters less than the microarchitecture, memory system, and power budget.

---

## 11. Speculative Execution Security

Speculative execution is essential for performance but creates side channels. Spectre and Meltdown (2017) showed that speculative loads leave traces in cache timing that leak secret data.

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

Mitigation cost: 2–10% performance overhead on server workloads
```

**Hardware mitigations:**

| Mitigation | What it does | Performance cost |
|-----------|--------------|------------------|
| LFENCE | Serialise — stop speculating past this point | 2–5% |
| IBPB | Flush branch predictor on context switch | 1–2% |
| Retpoline | Replace indirect jumps with return-based trampoline | 1–5% |
| STIBP | Restrict predictor sharing between SMT threads | 2–8% |
| SSBD | Speculative Store Bypass Disable | 2–5% |

**AI hardware impact:** GPU SIMT execution is less vulnerable — no speculative execution, no branch prediction, and warps don't share predictor state. But CPU-side code (model loading, preprocessing, serving) must still be patched.

---

## 12. Labs

### Lab 1 — Cache Miss Profiling

**Goal:** measure how access pattern affects cache performance; tie latency numbers to the memory mountain table.

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
- Tiled transpose (32×32 blocks) — should be fastest

**Deliverable:** table of L1/L2/L3 miss rates for each version; explain the difference.

### Lab 2 — SIMD Vectorisation

**Goal:** measure scalar vs SIMD vs compiler-auto-vectorised throughput on a dot product kernel.

```cpp
// Compile and compare:
// gcc -O2 -march=native -o dot dot.cpp        (auto-vectorise)
// gcc -O0 -fno-tree-vectorize -o dot_scalar dot.cpp  (scalar)
// Manual intrinsics version (see section 5.2)
```

Benchmark with N = 10M floats. Use `perf stat` to verify SIMD instructions appear.

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

**Deliverable:** instruction count, code size, and measured GFLOPS for each ISA (use QEMU for emulation if no native hardware).

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

**Deliverable:** accuracy table for each predictor; explain why correlating wins on nested loops.

### Lab 6 — Memory Bandwidth Measurement

**Goal:** measure actual memory bandwidth at each level of the hierarchy.

```c
#define MB (1024 * 1024)
double measure_bandwidth(int* data, int size_bytes, int stride) {
    volatile int sum = 0;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < 100; iter++)
        for (int i = 0; i < size_bytes / sizeof(int); i += stride)
            sum += data[i];
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return (100.0 * size_bytes) / elapsed / 1e9;  // GB/s
}
```

Test with array sizes: 16 KB (fits L1), 256 KB (fits L2), 8 MB (fits L3), 128 MB (DRAM).
Vary stride: 1, 2, 4, 8, 16, 32 cache lines.

**Deliverable:** bandwidth vs array size plot — you should see clear L1/L2/L3/DRAM plateaus.

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

**Key takeaway:** The ISA you write code for is secondary. The memory hierarchy and arithmetic intensity of your workload determines performance. A 10 FLOP/byte kernel runs at the same ~33 TFLOPS on H100 regardless of whether you write it in CUDA PTX or high-level Python — it's hitting the HBM bandwidth ceiling.

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
