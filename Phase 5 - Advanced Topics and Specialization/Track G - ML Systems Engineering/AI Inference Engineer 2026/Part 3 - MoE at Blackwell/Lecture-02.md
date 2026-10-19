# Part 3 · Lecture 02 — Blackwell: identify the product and the data path

“Blackwell” names an architecture family, not one memory capacity or one CUDA target. Separate HGX B200, GB200 systems, Blackwell Ultra products, and workstation/consumer devices. Record the actual SKU and query its compute capability before building kernels.

## Capacity is SKU-specific

The official [HGX B200 product footprint report](https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf) describes eight B200 GPUs with **180 GB each**. Do not carry an early 192 GB claim into that deployed configuration. NVIDIA's [versioned GPU support table](https://docs.nvidia.com/nim/bionemo/openfold2/2.5.0/support-matrix.html) distinguishes B200 and B300 capacities; B300's 288 GB is a different product.

The [GB200 NVL72 specification](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) lists 36 Grace CPUs, 72 Blackwell GPUs, 13.4 TB aggregate HBM, and 372 GB HBM per two-GPU superchip. Its rack aggregate bandwidth is not a single GPU's usable bandwidth. The same table distinguishes sparse and dense Tensor Core rates through footnotes. Do not substitute GB200 numbers into an HGX B200 estimate just because both are Blackwell.

A large NVLink domain is valuable for distributing experts, but is not one ordinary CUDA device with a single uniformly fast memory pool. Placement, communicator groups, link contention, and synchronization remain visible to software.

## Memory access has several paths

Local HBM, another GPU's memory, Grace CPU memory through NVLink-C2C, and another node through a NIC have different latency/bandwidth contracts. CPU/GPU coherence does not make CPU DRAM equivalent to local HBM. Offloading an expert saves local capacity but adds transfer or remote-read cost when that expert is used.

Keep die-to-die links inside a GPU package distinct from GPU↔GPU NVLink and CPU↔GPU NVLink-C2C. A bandwidth summed across links or both directions cannot be divided into a one-way payload without explaining which links and directions the transfer actually uses.

## Low precision needs a format and a kernel

Blackwell adds hardware paths for lower-precision arithmetic, but an arbitrary INT4 checkpoint cannot simply be relabeled NVFP4. Values, block scales, layout, accumulation, and supported shapes must agree with the kernel. The [Transformer Engine 2.19 precision guide](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.19/examples/fp8_primer.html) distinguishes MXFP8's block scaling from NVFP4's format and scaling. NVFP4 and MXFP4 are distinct encodings/recipes.

Four-bit values have half the raw payload of eight-bit values. Scales, unquantized tensors, padding, conversion, and workspaces prevent a universal twofold reduction in complete model memory. Higher arithmetic throughput and lower memory traffic constrain different portions of execution; their nominal ratios do not multiply into a guaranteed end-to-end speedup.

Use the [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) for architecture-specific capacities and code-generation considerations. A CUDA toolkit accepting an architecture flag establishes compilation support; executing correctness tests on that device is a separate step. Hopper-specific instruction/scheduling assumptions do not transfer unchanged.

## Work a placement budget

Suppose a toy checkpoint has 200 GB of actual allocated weights and two devices offer 160 GB each after platform reservation. Even splitting weights leaves 60 GB/device for KV, activations, and workspaces. If some attention weights are replicated, subtract that replication explicitly. If expert assignment is uneven, the largest rank may run out first despite enough aggregate free memory.

For a real MoE, measure per-rank peaks and active expert distributions. “Total weights divided by GPU count” is only an initial lower-complexity model. Quantized files, runtime-packed weights, and temporary loading copies can have different sizes. Account for startup as well as steady state.

## Optional hardware exercise

Inventory exact GPU product/capability, memory, driver/toolkit, NCCL/runtime versions, topology, and power configuration. Run one supported GEMM at a fixed shape/dtype against a higher-precision reference. Then benchmark representative message sizes with nccl-tests and a real routing workload. Keep local and cross-device measurements separate.

No B200/B300/NVL72 performance measurement is claimed by the CPU lab. A useful result may be that a supported higher-precision configuration is faster for your small shapes. Record that outcome instead of forcing a vendor-peak narrative.

**Check.** Does doubling aggregate NVLink bandwidth halve dispatch latency? **Answer:** only in an ideal bandwidth-dominated case with the same usable paths; fixed overhead, imbalance, contention, and packing may dominate. Next: [expert parallelism](Lecture-03.md).
