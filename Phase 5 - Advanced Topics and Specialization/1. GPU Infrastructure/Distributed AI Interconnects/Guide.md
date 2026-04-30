# Distributed AI Interconnects: vLLM, PyTorch, UCX, and UCC

**Phase:** 5 - Advanced Topics and Specialization  
**Track:** GPU Infrastructure  
**Level:** Advanced infrastructure / distributed inference systems

---

## Why this course exists

Modern AI serving is no longer only:

```text
one model
one GPU
one process
```

Large inference systems often become:

```text
vLLM
  -> PyTorch distributed process groups
  -> collective communication backend
  -> NCCL / RCCL / UCC
  -> UCX / InfiniBand / RoCE / TCP / shared memory / NVLink
  -> physical topology
```

The hard part is not just starting a model server.

The hard part is knowing what happens when the physical topology does not match the logical distributed topology.

This course focuses on a practical question:

> If we have three nodes arranged like `A <-> B <-> C`, and `A` cannot reach `C` directly, what do vLLM, PyTorch, UCX, and UCC actually do?

Short answer:

> vLLM and PyTorch do not magically route around a broken or non-routable fabric. They create logical process groups and invoke communication operations. The selected backend and the underlying network must provide the required reachability. UCC can provide a collective abstraction and choose algorithms/transports, but it is not a magic application-layer router for an unreachable topology.

---

## Learning objectives

By the end of this course, you should be able to:

1. Explain the difference between vLLM orchestration, PyTorch process groups, collective libraries, transport layers, and physical fabric.
2. Understand why tensor parallel inference needs much stronger communication than pipeline parallel inference.
3. Explain what happens when a distributed job assumes full rank reachability but the real network is only a chain.
4. Distinguish NCCL/RCCL, UCX, and UCC.
5. Design a topology-aware vLLM deployment plan.
6. Debug distributed inference hangs using logs, topology checks, and small collective tests.
7. Decide when to use pipeline parallelism, tensor parallelism, data parallelism, or separate replicas.
8. Build a research plan for custom or non-standard interconnects.

---

## 1. Stack mental model

Use this map first.

```text
Application
  vLLM API server / offline engine

Distributed execution runtime
  Ray or multiprocessing

Framework communication layer
  PyTorch torch.distributed ProcessGroup

Collective / communication backend
  NCCL, RCCL, Gloo, UCC, MPI, XCCL

Transport / fabric abstraction
  UCX/UCP, InfiniBand verbs, RoCE, TCP, shared memory, CUDA IPC, ROCm IPC

Physical links
  NVLink, NVSwitch, PCIe, NICs, cables, switches, subnets
```

Each layer has a different job.

| Layer | Responsibility | What It Does Not Do |
|---|---|---|
| vLLM | Serve LLMs with batching, KV cache, tensor/pipeline/data parallel execution | Fix broken network reachability |
| Ray | Place workers and manage distributed Python tasks | Guarantee high-performance GPU collectives |
| PyTorch distributed | Create ranks, process groups, and call collectives | Automatically invent fabric routing |
| NCCL/RCCL | GPU collectives for NVIDIA/AMD-like accelerator stacks | Make disconnected peers reachable |
| UCC | Unified collective API with multiple transport layers | Replace topology design |
| UCX | Transport abstraction for RDMA, TCP, shared memory, GPU memory | Decide application parallelism strategy |
| Fabric | Actual packets/transactions between devices | Understand your model graph |

---

## 2. What vLLM uses distributed communication for

vLLM supports distributed inference using tensor parallelism and pipeline parallelism.

The important difference:

```text
Tensor parallelism:
  split one layer across devices
  frequent collectives inside model execution

Pipeline parallelism:
  split layers across stages
  activation traffic mostly between adjacent stages
```

For tensor parallelism, vLLM calls communication operations such as:

```text
all-reduce
all-gather
reduce-scatter
gather
send / recv
```

That means all ranks in the tensor-parallel group must communicate correctly through the selected backend.

For pipeline parallelism, the model layers are partitioned into stages:

```text
node A: layers 0-15
node B: layers 16-31
node C: layers 32-47
```

The forward path is naturally chain-like:

```text
A -> B -> C
```

That is why pipeline parallelism is often easier to fit onto weak or non-uniform interconnects.

But there is a warning:

> even if data movement is mostly adjacent, the runtime may still need global control-plane communication for startup, rendezvous, health checks, metadata, barriers, and process-group setup

So "pipeline parallelism fits a chain" does not mean "the cluster can have no route between endpoint ranks at all."

---

## 3. The `A <-> B <-> C` topology problem

Assume:

```text
A can reach B
B can reach A and C
C can reach B
A cannot reach C directly
C cannot reach A directly
```

There are three different cases.

### Case 1: IP routing exists through B

If `A` can route to `C` through `B` at the network layer, then from the application perspective `A` and `C` are reachable.

The path is indirect, but reachable:

```text
A -> B -> C
```

This may work functionally.

But performance may be poor because all endpoint traffic competes for the middle node/link.

Expected symptoms:

- high latency
- lower all-reduce bandwidth
- B becomes a bottleneck
- collectives are slower than topology diagrams suggest
- tail latency increases under load

### Case 2: no IP/RDMA route exists from A to C

If `A` truly cannot address or connect to `C`, most distributed frameworks will fail or hang during initialization or first collective.

Typical failure points:

- Ray cannot form a healthy cluster
- PyTorch `init_process_group()` cannot complete
- NCCL/RCCL communicator initialization hangs or errors
- first all-reduce or all-gather stalls
- UCX endpoint setup fails

The important answer:

> this is not something vLLM fixes at the model layer

vLLM assumes the distributed runtime and communication backend can connect the required ranks.

### Case 3: custom application intentionally uses only adjacent communication

If you build a custom runtime that only sends:

```text
A -> B
B -> C
```

and never requires:

```text
A <-> C
```

then a line topology can work.

But this requires careful design:

- no global all-reduce across all three ranks
- no all-gather across all three ranks
- no all-to-all across all three ranks
- no process-group backend that assumes full pairwise reachability
- explicit routing or staged forwarding logic

That is a custom distributed system, not a normal "just set tensor_parallel_size=3" deployment.

---

## 4. The direct answer: is this a UCC glue-layer problem?

Usually, no.

UCC is a collective communication API and library.

It can select collective algorithms and transport layers, and it can use components such as UCX/UCP, NCCL, SHARP, CUDA, or HIP depending on the build and runtime.

But UCC is not a substitute for fabric reachability.

Think of UCC like this:

```text
"I need all-reduce across these ranks.
Which collective implementation should I use?"
```

Not:

```text
"A and C cannot talk.
Please invent a safe, high-performance routed network through B."
```

If the underlying transport can route traffic, UCC may use it.

If the underlying transport cannot establish endpoints or move data between required ranks, UCC cannot make the job correct by itself.

---

## 5. UCX versus UCC

These names are easy to confuse.

| Component | Simple Meaning | Example Use |
|---|---|---|
| UCX | Transport framework | Move data over InfiniBand, RoCE, TCP, shared memory, CUDA, ROCm |
| UCC | Collective communication library | Implement all-reduce, broadcast, all-gather, all-to-all through selected transport layers |

UCX provides communication primitives and transport selection.

UCC provides collective operations.

Relationship:

```text
UCC collective
  -> may use UCP transport layer
  -> UCP comes from UCX
  -> UCX uses RDMA/TCP/shared memory/GPU transports
```

So:

```text
UCX = how bytes move
UCC = how collective algorithms are expressed and selected
```

---

## 6. PyTorch distributed in this stack

PyTorch distributed creates a process group.

A process group has:

- `rank`
- `world_size`
- backend
- rendezvous method
- collective operations

Example:

```python
import torch
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",
    init_method="env://",
)

x = torch.ones(1, device="cuda")
dist.all_reduce(x)
```

For CUDA GPUs, PyTorch commonly uses NCCL.

PyTorch also exposes other backends, and the documentation lists backends including Gloo, NCCL, UCC, MPI, XCCL, FAKE, and registered third-party backends. The UCC backend is marked experimental in current PyTorch documentation.

Practical implication:

> if PyTorch process-group setup or a small all-reduce fails, vLLM distributed serving will not be stable above it

Always debug the backend with tiny tests before blaming the model server.

---

## 7. Why tensor parallelism is topology-sensitive

Tensor parallelism splits layer math.

Example:

```text
linear layer weight matrix
  split across ranks
  each rank computes partial output
  ranks synchronize partial results
```

That synchronization creates collectives.

For a tensor-parallel group with ranks:

```text
[A, B, C]
```

the backend may need communication patterns equivalent to:

```text
A <-> B
B <-> C
C <-> A
```

Even if the algorithm is ring-based and sends only neighbor chunks at a time, the ring normally needs a closed cycle.

A line is not a ring.

```text
Line:
  A <-> B <-> C

Ring:
  A <-> B <-> C <-> A
```

If no `C <-> A` path exists, the collective either needs:

- network-layer routing through B
- a different collective algorithm that explicitly supports the topology
- a custom communication plugin/runtime
- a different parallelism strategy

---

## 8. Pipeline parallelism is the first workaround to test

For a weak chain topology, pipeline parallelism is usually the first serving strategy to evaluate.

Example:

```bash
vllm serve <model> \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 3 \
  --distributed-executor-backend ray
```

The idea:

```text
A: early layers
B: middle layers
C: late layers
```

This reduces all-rank collectives compared with tensor parallelism.

Tradeoffs:

- better fit for chain topology
- lower intra-layer synchronization
- possible pipeline bubbles
- harder scheduling for short prompts or low concurrency
- still requires working cluster control-plane communication
- endpoint failure can stall the whole pipeline

Use pipeline parallelism when the model is too large for one node but the interconnect is not strong enough for heavy tensor parallel collectives.

---

## 9. Data parallel replicas avoid the chain problem

Another practical answer is not to split one model across all three nodes.

Instead:

```text
A: full model replica
B: full model replica
C: full model replica
```

Then route requests at the HTTP/load-balancer layer.

This is data parallel serving.

Pros:

- no cross-node GPU collectives for normal inference
- easiest operational model
- failure isolation is better
- latency is predictable

Cons:

- each node must fit the full model
- memory inefficient for very large models
- no single request uses all nodes

For production inference, replicas are often better than fragile model-parallel experiments unless the model size forces sharding.

---

## 10. NVLink clarification

NVLink is not a universal multi-node router.

There are several different situations:

```text
single server with direct NVLink:
  GPU-to-GPU peer path inside one NVLink domain

server with NVSwitch:
  many GPUs connected through switch fabric inside the system

multi-node NVIDIA systems:
  require specific NVLink Switch / networking architecture

ordinary Ethernet or InfiniBand cluster:
  inter-node communication normally uses NICs, RDMA, TCP, NCCL net plugins, UCX, etc.
```

Do not assume:

```text
"NVLink exists somewhere, therefore every node can peer-to-peer every other node."
```

Always inspect the actual topology.

For NVIDIA systems, start with:

```bash
nvidia-smi topo -m
```

For network reachability:

```bash
ip route
ping <peer>
nc -vz <peer> <port>
```

For RDMA:

```bash
ibv_devinfo
ib_write_bw
ib_read_bw
```

For NCCL:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING ...
```

---

## 11. Debugging ladder

Do not start with vLLM.

Start at the bottom.

### Step 1: physical and IP reachability

Check each pair:

```text
A -> B
B -> A
B -> C
C -> B
A -> C
C -> A
```

If `A <-> C` is expected to work through routing, prove it.

### Step 2: transport tests

For UCX:

```bash
ucx_info -d
ucx_perftest <peer> -t tag_bw
```

For RDMA:

```bash
ibv_devinfo
ib_write_bw <peer>
```

For TCP fallback:

```bash
iperf3 -s
iperf3 -c <peer>
```

### Step 3: collective tests

For NCCL:

```bash
all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

For PyTorch:

```bash
torchrun --nnodes=3 --nproc-per-node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=<head>:29500 \
  test_allreduce.py
```

### Step 4: vLLM small model

Only after collectives work, test vLLM with a small model.

Use the smallest possible topology first:

```text
1 node
2 nodes
3 nodes
```

Then compare:

```text
TP only
PP only
TP + PP
replicas
```

---

## 12. Minimal PyTorch all-reduce test

Use this before running a large model.

```python
# test_allreduce.py
import os
import socket

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend=os.environ.get("BACKEND", "nccl"))

    rank = dist.get_rank()
    world = dist.get_world_size()
    host = socket.gethostname()

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    x = torch.ones(1, device=device) * (rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    expected = world * (world + 1) / 2
    print(f"rank={rank} host={host} value={x.item()} expected={expected}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Run:

```bash
BACKEND=nccl torchrun \
  --nnodes=3 \
  --nproc-per-node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=<head-node-ip>:29500 \
  test_allreduce.py
```

If this hangs, vLLM is not the first problem.

---

## 13. How to choose a parallelism strategy

Use this decision table.

| Constraint | First Strategy to Try | Why |
|---|---|---|
| Model fits on one GPU/node | Data parallel replicas | Avoid distributed collectives |
| Model fits on one node but needs multiple GPUs | Tensor parallel within node | NVLink/NVSwitch/PCIe is easier than multi-node |
| Model does not fit on one node | Pipeline parallel across nodes | Lower communication pressure than cross-node TP |
| Strong full-mesh InfiniBand/RoCE | Tensor parallel or TP+PP | Fabric can support collectives |
| Chain topology only | Pipeline parallel or custom runtime | Avoid all-rank collectives where possible |
| Heterogeneous nodes | Replicas by capability class | Avoid synchronizing mismatched hardware |
| Researching custom interconnect | Start with microbenchmarks | Prove transport before model serving |

---

## 14. Common failure modes

### Failure: Ray sees nodes, vLLM still hangs

Ray placement is not enough.

Ray may launch workers successfully while NCCL/RCCL/UCC fails later during GPU communication.

Debug:

```text
Ray cluster status
  -> PyTorch all-reduce
  -> NCCL/RCCL logs
  -> vLLM
```

### Failure: TCP works but RDMA fails

This means basic IP reachability is not enough for the chosen transport.

Check:

- RDMA driver versions
- GID index for RoCE
- lossless Ethernet configuration
- MTU consistency
- firewall rules
- container device access
- GPU-NIC PCIe locality

### Failure: two nodes work, three nodes hang

This often indicates topology or rank-pair reachability mismatch.

Check every pair.

Do not assume:

```text
A <-> B works
B <-> C works
therefore A <-> C works
```

### Failure: performance is far below expected bandwidth

Likely causes:

- traffic routed through a bottleneck node
- wrong NIC selected
- no GPUDirect RDMA
- TCP fallback
- PCIe NUMA mismatch
- collectives using a poor ring/tree for the topology
- tensor parallel across slow links

---

## 15. Research project: topology-aware distributed inference

A good research project is:

> Build a topology-aware vLLM serving plan that chooses TP, PP, replicas, or custom communication based on measured link bandwidth and reachability.

Inputs:

```text
nodes
GPUs per node
GPU memory
GPU-GPU bandwidth matrix
NIC-NIC bandwidth matrix
reachability graph
model size
KV-cache budget
target latency
target throughput
```

Output:

```text
recommended serving layout
  data parallel replicas
  tensor parallel groups
  pipeline stages
  node placement
  expected bottleneck links
  required backend settings
```

Example graph:

```text
Nodes:
  A, B, C

Reachability:
  A-B: 100 Gb/s
  B-C: 100 Gb/s
  A-C: none

Recommendation:
  avoid TP group [A, B, C]
  test PP stages [A, B, C]
  if global runtime requires A-C reachability, add routing/switching or use replicas
```

---

## 16. Practical lab plan

### Lab 1: backend smoke test

Run the PyTorch all-reduce script with:

```text
world_size = 2
world_size = 3
backend = nccl
backend = gloo
backend = ucc, if built and available
```

Record:

- success/failure
- time to initialize
- time to all-reduce
- error messages
- selected interfaces

### Lab 2: vLLM placement experiment

Run a small model with:

```text
TP=1, PP=1
TP=2, PP=1
TP=1, PP=2
TP=1, PP=3
```

Record:

- startup success
- first-token latency
- output-token latency
- throughput under concurrency
- GPU utilization
- network throughput

### Lab 3: topology failure injection

Block direct traffic between two endpoint nodes.

Observe:

- Ray behavior
- PyTorch process group behavior
- collective behavior
- vLLM startup behavior
- whether errors are explicit or hang-like

The goal is not to make it fail.

The goal is to learn where the stack detects the topology mismatch.

---

## Key takeaways

- vLLM is the model serving layer, not the network router.
- Ray places workers, but GPU collectives still depend on the communication backend and fabric.
- PyTorch process groups call backend collectives; they assume the backend can reach required ranks.
- UCX moves bytes across transports such as RDMA, TCP, shared memory, CUDA, and ROCm.
- UCC expresses collective communication and can use multiple transport layers, but it does not magically fix unreachable peers.
- Tensor parallelism is much more topology-sensitive than pipeline parallelism.
- A chain topology can fit pipeline stages, but normal distributed runtimes may still need global reachability.
- Always prove reachability, transport, and collectives before debugging vLLM.

---

## References

- vLLM parallelism and scaling: [https://github.com/vllm-project/vllm/blob/main/docs/serving/parallelism_scaling.md](https://github.com/vllm-project/vllm/blob/main/docs/serving/parallelism_scaling.md)
- vLLM distributed API reference: [https://docs.vllm.ai/en/stable/api/vllm/distributed/](https://docs.vllm.ai/en/stable/api/vllm/distributed/)
- PyTorch distributed communication package: [https://docs.pytorch.org/docs/2.11/distributed.html](https://docs.pytorch.org/docs/2.11/distributed.html)
- UCX main features: [https://openucx.readthedocs.io/en/master/ucx_features.html](https://openucx.readthedocs.io/en/master/ucx_features.html)
- UCC API and documentation: [https://openucx.github.io/ucc/](https://openucx.github.io/ucc/)
- UCC specification: [https://openucx.github.io/ucc/api/v1.4.4/html/index.html](https://openucx.github.io/ucc/api/v1.4.4/html/index.html)
- UCC source repository: [https://github.com/openucx/ucc](https://github.com/openucx/ucc)
- NVIDIA NCCL documentation: [https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

