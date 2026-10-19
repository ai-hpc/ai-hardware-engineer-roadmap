# 03 — Place L40S replicas and model shards deliberately

**Outcome:** choose a feasible parallel layout from capacity, topology and service objectives. Source review: 2026-10-18. Deployment commands are hardware exercises, not measured results.

## Capacity comes before a latency prediction

A 12-card server's 576 GB nominal total is split into twelve 48 GB devices. A 70B BF16 model's approximately 140 GB weights cannot fit in TP=2; TP=4 leaves about 35 GB raw weights/rank before cache/workspace and imbalance. A four-bit artifact might use about 35 GB raw weights in total, plus scales and other tensors. Actual runtime support and peak memory decide the group size.

L40S has no NVLink. PCIe path, peer access, switches and NUMA determine communication. The link's Gen4 x16 theoretical one-way rate is roughly 32 GB/s, not a guaranteed P2P or AllReduce rate. Measure actual collectives rather than assigning every call “30–50 ms.” [L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/), [NCCL accounting](../NCCL-Deep-Dive/02-Algorithms-and-Bandwidth.md).

In a simple tensor-parallel transformer, forward reductions follow the attention **output projection** and MLP down projection. A column-sharded QKV projection can be local on replicated input; it is not inherently an AllReduce. Sequence parallelism, attention layout and backward change the schedule.

## Three layouts to compare

| Layout on 12 GPUs | Suitable hypothesis | What might invalidate it |
|---|---|---|
| 12 one-GPU replicas | Model and request working set fit on one card; independent requests dominate | CPU/network bottleneck, skewed queues, long requests exhausting cache |
| 6 two-GPU replicas | Supported quantized/medium model needs two cards | Unsupported head/weight partition, host-staged communication, inadequate capacity |
| 3 four-GPU replicas | A larger model needs four ranks | Communication and latency cost exceed benefit, uneven memory or cross-NUMA contention |

A single replica's group should use measured favorable connectivity. Do not assume adjacent numeric GPU IDs share a switch; inspect UUID/PCI bus/topology. More replicas do not guarantee linear throughput when they share host resources.

## Launch one replica, then scale the pattern

Set `MODEL` to a prepared local snapshot and check the installed runtime's support first. Device visibility selects physical devices; the process sees those selected devices renumbered locally. `--gpu-ids` is not the generic vLLM serve selector.

```bash
# One replica on physical GPU 0, local-only listening for the initial test.
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --served-model-name course-model \
    --max-model-len 4096 --max-num-seqs 8 --host 127.0.0.1 --port 8000

# A separate two-GPU replica, only if this model/format fits and TP=2 is supported.
CUDA_VISIBLE_DEVICES=2,3 vllm serve "$MODEL" --served-model-name course-model \
    --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 \
    --host 127.0.0.1 --port 8001
```

Run these in separate allocated services/terminals, or use a supervisor/scheduler with explicit resource requests. Bind a production endpoint through the deployment's configured access layer. Record ports, device UUIDs, runtime versions, health checks, admission limits and shutdown behavior. [vLLM parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/).

## Pipeline parallelism trades a different cost

Pipeline stages hold different layers. A boundary transfers activations forward and gradients backward during training; inference scheduling differs. For an illustrative BF16 activation `[4,2048,8192]`, the payload is 134,217,728 bytes. At a theoretical exclusive 32 GB/s path, a one-way payload lower bound is **4.194 ms** before overhead. This is arithmetic, not an L40S measurement.

Pipeline parallelism can avoid frequent within-layer collectives but adds stage scheduling and bubble/dependency costs. More microbatches can improve throughput while increasing memory or latency. Check model/runtime support and compare under the same request distribution; there is no unconditional “PP beats TP on PCIe” rule.

## Cross-node networking needs its own baseline

An HDR 200 Gb/s port offers a 25 GB/s one-way line-rate ceiling, so a claimed 20 GB/s bidirectional result cannot be compared directly with a 32 GB/s one-way PCIe ceiling. Record direction, payload scope and topology. Host staging, GPU-memory RDMA, multiple NICs and contention change application behavior.

Do not paste `NCCL_IB_GID_INDEX=3`, numeric GDR levels or a traffic class from another cluster. Modern NCCL can select a GID automatically; fabric policies and actual devices determine valid settings. Follow the [multi-node launch and diagnostics](../NCCL-Deep-Dive/05-Multi-Node-Clusters.md).

## Load balance work, not only request counts

A 10-token request and a 10,000-token request do not impose equal work. Measure queueing, prefill, decode and cache occupancy per group. Define bounded queues/admission, completion/error timeouts and retry policy; indiscriminate retries can duplicate work and worsen overload. Count successful responses that meet quality and deadline constraints.

**Worked comparison:** hypothetical group A produces 100 completions/s but only 70 satisfy the deadline; group B produces 90 completions/s with 85 on time. Under the same quality threshold, B has higher deadline-qualified goodput. These are invented arithmetic inputs, not GPU measurements.

**Exit artifact:** a physical rank map, per-rank capacity table, topology/collective evidence, and a controlled comparison of two feasible layouts using [chapter 5's method](05-Benchmarks.md).
