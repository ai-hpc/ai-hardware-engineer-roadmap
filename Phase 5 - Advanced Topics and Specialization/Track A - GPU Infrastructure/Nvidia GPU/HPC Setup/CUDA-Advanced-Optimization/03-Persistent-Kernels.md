# 03 — Persistent work: reuse blocks, preserve ownership

**Outcome:** distinguish finite tile scheduling from a long-running host/device server, and explain why a queue needs more than a flag. Follow `finite_tile_worker` in the [compiled lab source](labs/cuda_examples.cu).

## 1. Persistence does not mean “runs forever”

A persistent kernel keeps a set of blocks working through multiple tasks or tiles during one launch. It can finish after a known finite workload. A long-running polling service is one possible design with additional progress, shutdown and communication requirements.

| Design | Work becomes available | Completion |
|---|---|---|
| Ordinary grid | Defined before launch | Blocks finish their assigned indices |
| Finite tile scheduler | Defined before launch | All tiles claimed and finished |
| Persistent service | May arrive during execution | Explicit shutdown/drain protocol |

The useful question is whether reusing resident work helps a measured workload. Polling, task claiming and synchronization have costs. Persistent work can consume resources that other kernels need and can affect power, fairness and responsiveness.

## 2. Start with a finite, understandable scheduler

The lab fixes all input data before launch. A device counter starts at zero. One thread per block atomically claims a tile ID and writes it into shared memory. A block barrier makes the ID available to all members. The block computes that tile, then synchronizes before claiming another.

```text
one thread claims tile -> block-wide publication -> process valid elements
            ^                                             |
            +------------- finish all consumers ---------+

If the claimed tile is outside the finite range, the whole block exits.
```

The atomic counter gives each valid tile a unique claim. It does not publish a concurrently written host payload: no such producer exists in this example. Inputs were copied before the kernel in the same stream, and output is read after completion.

**Worked case:** `n=1003`, tile width 256. There are four tiles with 256, 256, 256 and 235 valid elements. A serial illustrative assignment across three workers gives 491, 256 and 256 elements. Actual GPU claim order may differ; the required property is complete, exactly-once coverage, not a particular worker count distribution.

Run the [CPU contract exercise](labs/README.md) to check that coverage across boundary sizes, then compare the compiled GPU program against its numerical reference when hardware is available.

## 3. Blocks are not named SMs

`blockIdx.x` identifies a logical block. It does not identify the SM that schedules it. Choosing a block count using the device's SM count is a scheduling heuristic, not an affinity guarantee.

A persistent kernel may be limited by registers, shared memory, occupancy, load imbalance or work-queue contention. More resident blocks can improve latency hiding while also competing for resources. Measure multiple block counts on the same workload and inspect the generated code and profiler results.

For established persistent GEMM scheduling, inspect the selected [CUTLASS implementation](https://github.com/NVIDIA/cutlass) and its exact architecture-specific example. A finite GEMM tile scheduler does not imply that an entire LLM decode loop stays inside one indefinitely running kernel.

## 4. A concurrent queue needs a protocol

A bounded queue slot has a lifecycle such as:

```text
FREE -> producer owns slot -> READY -> consumer owns slot -> DONE -> FREE
```

This is **protocol pseudocode**, not a portable CPU/GPU implementation. A complete implementation must define:

- addressability: which host/device allocation each participant may access;
- atomic scope and supported operations for that memory and platform;
- release/acquire publication of payload and completion;
- sequence/generation identity to prevent confusion when a slot is reused;
- ownership while input/output buffers are in use;
- bounded capacity, backpressure, cancellation and shutdown;
- forward progress when work competes with other GPU activity.

A host cannot normally dereference an ordinary device pointer. A host fence is not automatically a GPU-system publication protocol. Device-scope atomics are not automatically system-scope atomics. `volatile` does not make a data race correct. Read the [CUDA C++ memory model](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html) and the platform's supported memory/atomic capabilities before building such a service.

**Ownership failure:** changing a slot from READY back to FREE when a consumer merely claims it lets the producer overwrite inputs still being processed. Use a separate owned/running state and release the slot only after the required completion and consumption steps.

## 5. Signal completion after all work finishes

A fence orders the calling thread's memory operations; it does not automatically wait for every other thread to finish producing output. A completion signal must follow the synchronization that establishes all relevant writes are complete and their publication visible to the intended consumer.

Likewise, a warp shuffle from lane zero only broadcasts within its warp. The lab uses a shared tile variable and a block barrier to publish a block-wide work ID. Mixing these scopes can leave some warps processing the wrong tile or exiting inconsistently.

A service shutdown must let all participants reach compatible collective decisions. One thread observing `running=false` while another waits at a collective can hang the kernel. Specify who requests shutdown, whether pending tasks drain, how cancellation is reported, and when memory may be freed.

## 6. Persistence, graphs and batching solve different problems

Graphs reduce repeated launch setup for a dependency structure. Batching changes how requests share work. A persistent tile scheduler reuses blocks during a kernel. A serving engine can combine these ideas, but one does not prove the presence of another.

Do not label an invented queue sketch “the TensorRT-LLM implementation”. Read a pinned runtime source and trace the actual path. Similarly, the `__grid_constant__` parameter annotation concerns parameter semantics; it does not make a kernel persistent.

**Hypothetical overhead comparison:** if a request contains 10 ms of device work and 20 µs of exposed submission delay, removing that delay can save at most about 0.2% of that simplified total. A much shorter device workload may present a different tradeoff. This bound assumes the delay is on the critical path and ignores any new queue/polling cost.

## 7. Decide with evidence

Compare a correct ordinary grid, a graph replay where applicable, and a finite persistent alternative. Keep inputs, numerical contract and workload distribution fixed. Report throughput, per-item latency, tail behavior, fairness, memory use and measured power under named conditions.

There is no universal extra wattage or speedup for persistence. Never infer that a robot control deadline holds just because kernel launches are fewer. The complete acquisition-to-command age still matters.

**Exercise:** increase tile-count imbalance, then explain when dynamic claiming can improve load balance. Add a deliberately duplicated claim to the CPU model and show that the coverage check catches it. A CPU model proves the indexing property it tests; it does not prove CUDA atomic visibility or progress.

**Completion:** explain the finite scheduler, its tail and termination cases, and the additional obligations of a host/device queue. Runtime results require a GPU; compilation and CPU coverage are separate evidence levels.

**Next:** [kernel fusion](04-Kernel-Fusion.md).
