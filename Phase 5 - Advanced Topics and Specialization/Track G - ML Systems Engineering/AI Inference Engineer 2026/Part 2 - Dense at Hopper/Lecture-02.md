# Part 2 · Lecture 02 — Hopper hardware: capacities and asynchronous contracts

H100 and H200 remain useful worked targets for studying inference. Hardware features help only when a kernel uses them correctly. This lesson explains the capacities, data paths, and synchronization you need to read a trace; it does not claim a GPU benchmark was run in this course's CPU validation.

## Read a specification without mixing quantities

NVIDIA's [H100 product specification](https://www.nvidia.com/en-us/data-center/h100/) distinguishes H100 SXM from H100 NVL. For the SXM example, HBM capacity is 80 GB and quoted HBM bandwidth is 3.35 TB/s. Dense BF16 Tensor Core throughput is about 989 TFLOP/s; marketing tables with sparsity can show roughly twice that. Match the footnote before using a roofline. FP64 CUDA-core and FP64 Tensor Core rates are separate entries.

H200 changes memory capacity/bandwidth as well as product configuration; use the [H200 specification](https://www.nvidia.com/en-us/data-center/h200/) for its 141 GB and up-to-4.8 TB/s figures. Dividing bandwidths gives an optimistic memory-transfer ratio, not a guaranteed decode speedup. Weight capacity, KV allocation, clock/power settings, kernels, batching, and communication also matter.

PCIe Gen5 x16's approximately 128 GB/s aggregate figure counts both directions; it is about 64 GB/s per direction before protocol overhead. Similarly, an aggregate bidirectional NVLink figure is not a one-way application bandwidth. Do not divide one-way payload by the sum of both directions.

## SM resources and occupancy

The [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) documents 64K 32-bit registers per SM, a 255-register maximum per thread, up to 228 KB shared memory per SM, and a combined 256 KB L1/texture/shared pool. The maximum shared-memory allocation per block is smaller than the SM total and large allocations require the documented opt-in.

A kernel's registers, shared memory, threads, and block limits jointly bound residency. More occupancy is not automatically faster: a larger tile may reuse data better with fewer resident blocks. Conversely, aggressive tiling may spill registers or leave too little parallelism. Report resource usage and actual duration for the same shape.

## Thread-block clusters and distributed shared memory

A cluster groups blocks that can coordinate and access each other's shared memory. The portable maximum is 8 blocks; supported H100 configurations can opt into larger nonportable clusters (up to 16), subject to device/MIG/resource restrictions. Query occupancy rather than treating 8 as every hardware maximum.

Blocks in a cluster are co-scheduled within a GPU processing cluster. This does not mean “one block occupies one entire SM.” Distributed shared-memory accesses require that the owning block's shared allocation remains alive. Establish initialization visibility before reading remote data and synchronize before an owner exits. A cluster barrier cannot fix an out-of-bounds address or a race outside the participating scope.

## TMA: move a tile while work continues

The Tensor Memory Accelerator supports asynchronous tensor transfers described by a tensor map. The descriptor encodes dimensions, strides, element type, and supported layout details. Alignment and extent requirements come from the CUDA/PTX API, not a guess based on a C array's shape.

A producer issues a transfer and associates it with the required transaction/barrier protocol. A consumer waits for the appropriate completion/phase before reading the destination. Reusing the buffer before its previous consumers finish is a separate hazard. Double buffering therefore needs two contracts: “new data has arrived” and “old data is no longer in use.” An ordinary block barrier is not a universal replacement for the asynchronous transfer's completion mechanism.

TMA can reduce address-generation and copy instruction work; it does not make data movement free or guarantee that it overlaps useful computation. Use the [CUDA asynchronous programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and the matching toolkit's tensor-map documentation when implementing a kernel.

## WGMMA: four warps cooperate

A warp group has 128 threads, or four warps. Hopper's asynchronous warp-group matrix instructions require a supported matrix shape/layout and the documented shared-memory descriptors or register operands. Fence, commit-group, and wait-group operations establish different parts of the protocol. Reading an accumulator or overwriting a shared tile too early can corrupt a result even when the code compiles.

The authoritative instruction contract is [PTX WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions). Learn through a maintained CUTLASS example for your toolkit before writing inline PTX. FlashAttention-3 is a useful [Hopper-specific research example](https://arxiv.org/abs/2407.08608) of overlapping transfers, matmul, and softmax; its reported results are tied to the paper's kernels and shapes.

## Hardware generation versus library version

Hopper introduced the first-generation Transformer Engine hardware and FP8 support. Transformer Engine 2.x is a **software library version**, not proof that Hopper has Blackwell's second-generation hardware features. MXFP8/NVFP4 paths have additional hardware and recipe requirements; use the [versioned precision guide](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.19/examples/fp8_primer.html).

**Worked design check.** A proposed two-stage kernel reserves 120 KiB per tile buffer: two buffers already need 240 KiB, beyond Hopper's per-SM shared-memory capacity before barriers or other storage. Reduce the tile/stages or change the layout; a compiler flag cannot create capacity.

**Optional hardware exercise.** Record GPU SKU, driver/toolkit, resource limits, one maintained GEMM's shape/dtype, correctness tolerance, warmup, synchronized timing, and generated code. Compare one resource change. A compile-only result demonstrates API compatibility, not timing or GPU correctness. Next: [quantization](Lecture-03.md).
