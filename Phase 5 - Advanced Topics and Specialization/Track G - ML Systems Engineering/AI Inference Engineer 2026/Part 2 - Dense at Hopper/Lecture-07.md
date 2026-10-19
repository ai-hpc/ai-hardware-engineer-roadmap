# Part 2 · Lecture 07 — Read a collective trace without inventing its algorithm

A communication library provides operations such as sum-all-reduce, all-gather, and reduce-scatter. A runtime may select custom kernels or fuse surrounding operations. The selected path depends on version, topology, message size, dtype, workspace, and graph-capture constraints. There is no universal “custom kernel → ring → host” fallback ladder shared by all runtimes.

## Semantics before implementation

An all-reduce sums corresponding elements across ranks and returns the result on every participating rank. All ranks must agree on count, datatype, communicator, and operation ordering. Mismatches can hang or corrupt a run. A GPU stream event establishes completion/dependencies; returning from an asynchronous launch does not mean the receiving tensor is ready for unrelated CPU access.

Read the [NCCL collective API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html). The reviewed NCCL 2.32.3 documentation includes `ncclAlltoAll` for **equal per-peer counts**. Older claims that NCCL never has an all-to-all API are stale. Variable-count dispatch can be expressed with matched grouped `ncclSend`/`ncclRecv`, as described in the [point-to-point guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html); do not invent a generic `ncclAlltoAllV` API.

## Small messages and large messages

For `[1,8192]` BF16 activations, the payload is 16 KiB. Fixed launch/protocol costs can dominate such a message. Larger prefill tensors make data movement more significant. NCCL's selected algorithm may be ring, tree, or another supported path; a ring's logical steps do not each imply a separate CUDA launch.

Custom collectives can exploit peer access and constrained layouts, trading generality for low overhead. Some implementations have topology-specific PCIe support; “custom means NVLink only” is not a universal rule. Unsupported cases may fall back, fail initialization, or be rejected. Inspect the version's capability checks and runtime logs.

Host shared-memory messaging carries scheduling/control data in some runtimes. It is not a general replacement implementation for GPU all-reduce, but CPU scheduling overhead can still affect per-token latency. Do not dismiss all host activity as irrelevant.

## What fusion can save

Consider `all_reduce(partial) → residual_add → RMSNorm`. A fused implementation may avoid separate launches and intermediate global-memory reads/writes. It must still produce the required reduced value, preserve residual semantics, apply the correct epsilon/scaling, and honor communication completion before normalization uses the data.

Workspace is more than `tokens × hidden × dtype_bytes`: flags, peer buffers, staging, alignment, and double buffering may contribute. The exact supported maximum and lifecycle come from the implementation. A fused kernel's name alone does not prove it selected the desired numerical mode or was faster.

## A reproducible investigation

Start from a supported, working distributed configuration. Save the runtime commit/container, NCCL version, GPU/PCIe/NVLink/NIC topology, rank map, and relevant configuration. Run a correctness test before collecting performance. Use [nccl-tests](https://github.com/NVIDIA/nccl-tests) with the message sizes your model actually produces; understand its algorithm-bandwidth versus bus-bandwidth normalization before comparing either with a product specification.

Capture a short representative serving trace and identify the collective call, stream, dependent compute, and idle gaps. A NCCL kernel is not proof that a custom path is broken; it may be the supported or faster choice. Compare an available configuration switch only after confirming its name in the installed version. Current vLLM documents `--disable-custom-all-reduce` in its [engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/); do not substitute an unverified environment variable from an old tutorial.

Run both settings with matched inputs, quality checks, warmup, cache state, and offered load. Report measured outcomes even if the supposedly optimized path loses. No 2×B300, H100, or NVL72 bandwidth result is claimed as locally measured in this lesson.

**Exercise.** A profile shows 1 ms of communication overlapping 0.8 ms of independent work and 0.4 ms of dependent work afterward. Under perfect overlap, that segment takes `max(1,0.8)+0.4 = 1.4 ms`, not 2.2 ms. If a dependency prevents overlap, it takes 2.2 ms. The trace must establish which schedule actually occurs.

Next: [MoE at Blackwell](../Part%203%20-%20MoE%20at%20Blackwell/README.md). Sparse expert routing adds irregular destinations and load imbalance to these communication contracts.
