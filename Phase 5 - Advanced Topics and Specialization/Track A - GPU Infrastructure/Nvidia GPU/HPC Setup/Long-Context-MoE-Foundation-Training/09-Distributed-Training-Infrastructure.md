# Module 09 — Distributed Training Infrastructure

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Stand up the multi-node H200/B200 infrastructure that long-context MoE training depends on — interconnects, schedulers, NCCL tuning, checkpointing, and fault tolerance — and verify it with a real multi-node training run.

**Prerequisites:** Modules 02 (long-context attention), 05 (MoE systems), 08 (mesh design). The HPC Setup [NCCL Deep Dive](../NCCL-Deep-Dive/README.md), [8×H200 Training/Inference](../8x-H200-Training-Inference/README.md), and [GPUDirect Storage](../GPUDirect-Storage/README.md) modules.

**Artifact:** A working multi-node Megatron training launch, a checkpoint + resume cycle, and a fault-injection test that kills one node and proves the run recovers from the last checkpoint.

---

## Why it matters

A long-context MoE training run is a long-running, high-throughput job that touches every part of the cluster: GPU, NVLink, InfiniBand, NVMe, network filesystem, scheduler. Any one of them failing wastes hours of GPU time. This module is what stops a 100K-USD training job from becoming a 100K-USD outage.

---

## Mental model

### The physical stack

```
Host bare-metal: BIOS, IOMMU, NUMA topology
   └─ OS + drivers: kernel, NVIDIA driver, MOFED (InfiniBand)
       └─ Runtime: CUDA, NCCL, MPI
           └─ Containers: NGC base image, your training image
               └─ Scheduler: Slurm or Kubernetes + GPU operator
                   └─ Training framework: Megatron-LM / NeMo / DeepSpeed
                       └─ Your code + config
```

Each layer can break independently. If you do not own the layers below "your code," you must at least know whom to call when they break.

### Interconnect tiers

| Tier | Hardware | Effective BW | Latency | Used by |
|------|----------|--------------|---------|---------|
| GPU-GPU intra-node | NVLink 4 / NVSwitch (Hopper) | ~900 GB/s per GPU pair, ~3.6 TB/s aggregate per 8-GPU node | ~µs | TP, SP, EP, intra-node CP |
| GPU-GPU inter-node | InfiniBand HDR/NDR (NDR = 400 Gbps) | ~50 GB/s per port, ~200 GB/s with 4× NDR ports per node | ~5 µs | PP, DP, inter-node CP/EP (avoid) |
| GPU-Storage | GPUDirect Storage over IB + NVMe | ~30 GB/s per node sustained | ms | data loading, checkpoint I/O |
| Host-Host | Ethernet | irrelevant for training | - | management, log shipping |

The factor of ~20× between NVLink and InfiniBand is what dictates your mesh layout (Module 08). The factor of ~30× between NVLink and GPUDirect Storage is what makes checkpoint I/O a foreground concern.

### Scheduler

For most HPC clusters: Slurm. For most Kubernetes-native clusters: KubeRay or Volcano with the NVIDIA GPU operator.

The training job needs:

- **Rank-aware placement**: every node must launch the same number of processes (`--nproc-per-node 8` is universal); rank 0 is the launcher; `MASTER_ADDR` / `MASTER_PORT` are passed through.
- **NCCL environment**: `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_P2P_LEVEL=NVL`, `NCCL_NVLS_ENABLE=1`, `NCCL_DEBUG=WARN` (set to `INFO` only when debugging).
- **Topology hints**: `CUDA_VISIBLE_DEVICES` for non-default GPU subsets; `NCCL_TOPO_FILE` for hand-tuned topology in NVSwitch boxes.
- **Resilience**: `--max-restarts` for torchrun; checkpoint-and-resume integration with the scheduler's restart hook.

The Slurm launcher template you want to keep around:

```bash
#!/bin/bash
#SBATCH --nodes=8 --ntasks-per-node=1 --gres=gpu:8
#SBATCH --time=24:00:00 --signal=USR1@180

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_P2P_LEVEL=NVL
export NCCL_NVLS_ENABLE=1
export NCCL_DEBUG=WARN

srun --container-image=$IMAGE --container-mounts=$PWD:/workspace \
     torchrun --nnodes=$SLURM_NNODES --nproc-per-node=8 \
              --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d \
              --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
              pretrain_gpt.py \
              [training args]
```

### NCCL tuning that actually matters

- `NCCL_P2P_LEVEL=NVL` — force NVLink-only peer access where available.
- `NCCL_NVLS_ENABLE=1` — enables NVLink Sharp for accelerated reductions on NVSwitch boxes (H100/H200 SXM).
- `NCCL_IB_HCA` — specify which HCAs to use; misnaming this falls back to TCP, which destroys performance.
- `NCCL_IB_GID_INDEX` — only relevant on RoCE; pick the right RoCE v2 GID.
- `NCCL_BUFFSIZE=8388608` — 8 MB per-channel buffer; useful for very long messages, careful with memory.
- `NCCL_ALGO=Tree,Ring` — let NCCL choose between tree (latency-bound) and ring (bandwidth-bound).

NCCL prints its chosen algorithms on init when `NCCL_DEBUG=INFO`. The first run with a new layout, capture that log and confirm it picked something sensible.

### Checkpointing

Long-context MoE training checkpoints are huge: a 70B-parameter model's optimizer state + parameters in fp32 is ~840 GB. Sharded checkpoints (one shard per DP × PP × EP × TP rank) are the only practical option.

The checkpoint layer:

- **Megatron's distributed checkpoint**: each rank writes its own shard; metadata file maps the mesh.
- **Torch DCP** (distributed checkpoint): newer, handles mesh changes between save and load.
- **Async checkpointing** (Megatron `--async-save`): write checkpoint in background, training continues; protects against the checkpoint-induced step-time spike.

I/O target: a 64×H200 cluster should checkpoint a 70B model in under 60 seconds end-to-end. If it takes longer, the storage tier is the bottleneck — see GPUDirect Storage module.

### Fault tolerance

In a real run, hardware *will* fail: a GPU ECC error, a switch hiccup, a node lost to a kernel panic. The job must:

1. Detect the failure (NCCL hang detection, watchdog).
2. Tear down cleanly (avoid leaving zombie processes).
3. Relaunch from the last checkpoint, possibly on different physical nodes.
4. Log the incident.

Torchrun's `--max-restarts` handles the relaunch; `c10d` rendezvous handles re-membership. The training script must be deterministic enough that resuming from step N reproduces step N+1's behavior (or close enough that loss curves stay sensible).

A practical pattern: a Slurm signal handler on `USR1` triggers a graceful checkpoint, then the scheduler resubmits the job. The job picks up the latest checkpoint automatically.

### Monitoring

Minimum dashboards:

- **Per-step time** + variance.
- **Per-rank loss** (must stay tight; divergence across ranks is a routing or numerics bug).
- **NCCL communication time** by collective.
- **GPU utilization, memory, temperature, power** per rank.
- **Network**: per-NIC throughput, retransmissions, link errors.
- **Storage**: write throughput during checkpoint windows.

Standard stack: NVIDIA DCGM exporter → Prometheus → Grafana. PyTorch profiler dumps for the deep-dive moments.

---

## Build it

### 1. Multi-node baseline

Use the Slurm template above (or your scheduler's equivalent). Launch a 2-node × 8-GPU training run with a small Megatron config:

```bash
sbatch train_2node.sbatch
# Inside the sbatch, pretrain_gpt.py with --num-layers 12 --hidden-size 2048
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2
# --num-experts 8 --moe-router-topk 2 --expert-model-parallel-size 2
# --seq-length 8192 --use-flash-attn --bf16
# --train-iters 200 --save /shared/checkpoints/v1 --save-interval 100
```

Confirm: NCCL chose NVLink intra-node, IB inter-node, no fallbacks. Per-step time stable. Loss decreasing.

### 2. Checkpoint and resume

After step 100, the job saved a checkpoint. Kill it. Restart with `--load /shared/checkpoints/v1`. Confirm the run picks up exactly at step 100 with a loss in line with the pre-kill trajectory.

### 3. Fault injection

While the run is going, log into one of the worker nodes and `kill -9` the training process on that node only. The scheduler should detect the missing rank, the job should tear down, and a relaunch (via `--max-restarts` or a Slurm requeue) should pick up from the last checkpoint. The total downtime: ideally under 5 minutes for a 2-node setup.

If the relaunch produces a different loss curve from what the pre-kill run was on track for, you have a determinism problem to fix before scaling up.

### 4. NCCL profiling

Rerun with `NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log`. Inspect the logs:

- Did NCCL pick the expected algorithm (tree vs ring vs hierarchical)?
- Are there warnings about fallbacks to TCP or to a slower path?
- Is the per-collective time consistent across ranks?

Persist this profile alongside your training log; it is the first thing you check when a future run regresses.

---

## Use it in the real stack

Real research clusters tend to live on top of:

- **NVIDIA Base Command Manager + DGX SuperPOD reference architecture** — the canonical "we sell you a long-context training setup" deployment.
- **Slurm + Pyxis + Enroot** for container-aware scheduling on bare-metal clusters.
- **Kubernetes + KubeRay + GPU Operator** for cloud-native deployments (less common for largest training jobs because of pod-restart latency).
- **CoreWeave / Lambda / Crusoe** managed offerings — provide a ready-made Slurm + IB + storage stack and hand you a ready-to-use launcher.

The MoE long-context skill page from NVIDIA assumes a Slurm-style launcher. Whatever stack you're on, your launcher must produce equivalent environment variables and process placement.

---

## Measure it

For the runs above:

- **Per-step time** average and p99.
- **NCCL time as fraction of step time** (should be ≤ 30% at the configurations you chose).
- **Checkpoint write time** (target: ≤ 60s for 70B-class).
- **Checkpoint resume time** (target: ≤ 2 min from job submission to first new step).
- **Fault-recovery total downtime** (target: ≤ 5 min).

Record these as a SLO for the cluster. Future regressions are compared against this baseline.

---

## Ship it

Drop into `lcm-course/`:

1. `train_2node.sbatch` — your scheduler launcher template.
2. `nccl_env.sh` — the NCCL environment variables you settled on.
3. `multi_node_run.log` — abbreviated log showing per-step time, NCCL inits, checkpoint and fault-injection cycles.
4. `infra_slo.md` — your cluster SLOs in measurement-ready form (numbers above with the values you actually measured).

---

## Related pages

- [Module 05 — MoE systems and infrastructure](05-MoE-Systems-Infrastructure.md)
- [Module 08 — Combining long-context and MoE](08-Combining-LongContext-and-MoE.md)
- [NCCL Deep Dive](../NCCL-Deep-Dive/README.md)
- [8×H200 Training/Inference](../8x-H200-Training-Inference/README.md)
- [GPUDirect Storage](../GPUDirect-Storage/README.md)
- Megatron-LM distributed checkpointing: <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/dist_checkpointing/README.md>
- NCCL tuning: <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>
