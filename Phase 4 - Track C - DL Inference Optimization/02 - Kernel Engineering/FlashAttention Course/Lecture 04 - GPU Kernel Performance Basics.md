# Lecture 4 — GPU execution concepts without invented latency constants

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** identify the hardware resource behind a kernel design choice. **Artifact:** a memory/ownership diagram and, when hardware is available, a validated microbenchmark report.

## 1. Treat capacity and latency differently

H100's SM has 64K 32-bit registers (256 KiB of register storage), with a 228 KB shared-memory capacity and a per-block limit described in NVIDIA's guide. Registers do not have zero latency or unlimited bandwidth. Shared-memory bandwidth is not an arbitrary “30 TB/s per SM” constant. Use [NVIDIA's Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) for limits and measure workload-dependent latency separately.

Global memory is device-addressable storage; L2 caches accesses; shared memory enables explicit block cooperation; registers hold thread-local values subject to allocation limits. Spills can turn a register-heavy design into additional memory traffic. A fused attention kernel can reread K/V even though it never stores the full score matrix.

## 2. Understand warps, synchronization and occupancy

A warp contains 32 threads, but modern independent thread scheduling means code must obey the documented synchronization rules. A shuffle mask names participating lanes; all lanes required by a collective must participate consistently. A full mask in a partially active warp is not a universal safe reduction.

Occupancy is the ratio of resident active warps to the architectural maximum. Registers, shared memory, block size and other limits constrain it. Higher occupancy can hide latency; it is not an optimization objective that supersedes useful work or locality.

For a shuffle reduction exercise, either launch complete warps and give inactive data lanes a neutral value, or use a correct active-lane protocol. Initialize reduction outputs before any atomic accumulation. Repeated launches without clearing an atomic output do not produce independent correctness or timing samples.

## 3. Compare equal work when studying coalescing

A strided kernel that only copies every 32nd element performs less work than a dense copy. It cannot support a fair “same-size copy is 20 times slower” conclusion.

A better experiment visits all N elements in both variants. One assigns contiguous indices to neighboring threads; another applies a bijective index permutation to thread assignments while still copying `y[index] = x[index]`. Verify every output element and count useful bytes identically. Document cache state and address alignment before interpreting transaction efficiency.

For an aligned warp of 32 float loads, contiguous addresses span 128 bytes; scattered addresses may touch many more memory sectors. Exact transaction rules depend on the operation and architecture. Read the [CUDA best-practices memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) rather than treating one 128-byte region as one universal HBM transaction.

## 4. Distinguish instruction families

- Warp-level MMA instructions use architecture-specific shapes and data types. One mnemonic does not imply a universal cycle count.
- Hopper WGMMA coordinates a warpgroup and supports asynchronous matrix operations with their synchronization rules.
- TMA handles multidimensional transfers using descriptors and barriers. It does not make movement or synchronization free.
- Blackwell `tcgen05` operations and tensor memory introduce another programming model; they are not simply Hopper WGMMA renamed.

C++ CuTe is a template-based layout library. Python CuTeDSL is a separate programming interface. Lecture 09 places them in the correct FA3/FA4 source families. [PTX instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/).

## 5. Measurement assignment and answers

If hardware is available, report exact device/toolkit/compiler, launch geometry, useful bytes, verified output, warmup, samples and profiler metrics. Keep input allocation and initialization outside the timed kernel region unless the experiment explicitly includes them. An asynchronous launch needs an event or synchronization boundary before its duration is meaningful.

**No GPU route:** draw the ownership/lifetime of one Q tile, one K/V tile and the running softmax statistics from the CPU lab. Explain where a two-stage pipeline needs an extra buffer and where a producer must wait before reusing it.

**Answer check:** a low occupancy result is not automatically a defect; a large live tile may improve reuse. A clean source compilation is not evidence of correct synchronization or GPU speed. The course's supplied CPU tests establish math/contract behavior, not those hardware outcomes.
