# Part 2 · Lecture 04 — Tensor parallelism: partition the math, then count communication

Tensor parallelism (TP) splits a layer's computation across devices. It can make weights fit and reduce local work, but introduces communication and synchronization. More GPUs are not automatically faster or cheaper. First reproduce the algebra on a CPU; then measure a real topology if available.

## Column and row partitions

Write a linear layer as `Y=XW`, with `X:[M,D]`, `W:[D,F]`. A column partition splits output features: `W=[W0 W1]`. Each rank computes its own `Y_r=XW_r`. No sum is needed to form those local output columns. If the next operation can consume that partition, avoid gathering them.

A row partition splits the reduction dimension: stack `W0` and `W1` vertically and split X correspondingly. Each rank computes `Z_r=X_r W_r`; the complete output is `Y=sum_r Z_r`. An all-reduce or a suitable reduce-scatter is now required, depending on the next layout.

For a two-layer MLP, a column-partitioned first projection followed by a row-partitioned second projection can keep the intermediate partitioned. SwiGLU's gate/up partitions must select matching channels. Attention uses related head partitions, but GQA, head-count divisibility, output projection, and KV replication complicate the layout. Draw tensors per rank before multiplying capacities by TP.

## A CPU proof

Let `X=[2,3]` and `W=[[1,4],[5,6]]`. The full output is `[17,26]`. A row partition yields `[2,8]` and `[15,18]`; summing restores `[17,26]`. Concatenating those partial results would be wrong. A column partition yields `[17]` and `[26]`; concatenating restores the output, while summing would be wrong.

This distinction explains why collective selection is a semantic requirement, not only a performance choice. Floating-point summation order can change rounding; check a numerical tolerance and the full model quality contract.

## The ring cost model

For P ranks and an N-byte input tensor per rank, a conventional ring all-reduce transfers approximately `2(P-1)/P × N` bytes per rank across reduce-scatter and all-gather. An illustrative latency model is:

```text
T_ring ≈ 2(P-1) * alpha + 2(P-1)/P * N / beta
```

Here alpha is a per-step latency parameter and beta is an effective link rate for that implementation/topology. This does not mean a ring launches a separate CUDA kernel for every logical step. NCCL can select other algorithms/protocols, use multiple channels, or exploit hardware features. The model is a teaching approximation, not a trace of your runtime.

For `M=1`, D=8192 and BF16, the activation tensor is 16 KiB. At P=8, the ring's per-rank byte volume is 28 KiB. At M=64, it is 1.75 MiB. Neither number determines duration without launch/protocol/topology information. See [NCCL collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html) and [nccl-tests](https://github.com/NVIDIA/nccl-tests).

## Capacity and the actual communicator

A rough per-rank budget includes sharded weights plus replicated weights, sharded/replicated KV, activations, collective workspaces, captures, and reserve. GQA with eight KV heads does not imply arbitrary TP degrees divide the cache evenly. A 70B BF16 model's approximately 140 GB payload is not the entire memory requirement.

Record rank-to-GPU mapping, NVLink/NVSwitch/PCIe links, NUMA placement, NICs for cross-node paths, and competing traffic. Aggregate advertised fabric bandwidth is not one-way per-collective throughput. Collectives only overlap useful independent work; moving them to a different stream does not remove data dependencies.

## The experiment

Compare TP degrees that fit at the same quality, request distribution, and latency target. Report per-replica and per-GPU throughput, TTFT/ITL, memory, and total cost. Keep both low-concurrency decode and long-prefill cases. A two-GPU replica may offer lower throughput than an eight-GPU replica but permit four replicas for the same fleet allocation.

A classic transformer TP layout often has reductions after attention output and MLP down projection. “Exactly two all-reduces per layer” is not universal: fusion, sequence parallelism, reduce-scatter/all-gather layouts, architectures, and implementations can change the collective sequence. Inspect the actual trace.

**Exercise.** If a shard has half the matmul time but adds 2 ms of communication, does TP help a 3 ms layer? **Answer:** idealized time becomes `1.5+2=3.5 ms`, so it loses latency under these assumptions. It may still be necessary for capacity. Next: [serving and scheduling](Lecture-05.md).
