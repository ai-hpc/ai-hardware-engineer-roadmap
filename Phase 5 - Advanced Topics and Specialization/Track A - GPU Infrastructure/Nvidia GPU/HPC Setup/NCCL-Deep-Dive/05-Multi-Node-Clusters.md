# 05 — Multi-node collectives without rank mistakes

**Outcome:** draw the physical paths, launch exactly one worker per GPU, and distinguish per-rank from per-node traffic. Prerequisite: [bandwidth accounting](02-Algorithms-and-Bandwidth.md). Source review: 2026-10-18. This chapter's GPU, network and scheduler exercises require an allocated cluster and were not executed here.

## Name the topology before choosing parallelism

An HGX H200 node has a fast switched GPU fabric; external NICs connect nodes. The advertised 900 GB/s NVLink figure is aggregate **bidirectional per GPU**. HDR 200 Gb/s means up to **25 GB/s one-way per port** before overhead; NDR 400 Gb/s means up to 50 GB/s. Number of NICs, GPU–NIC locality, fabric oversubscription and competing traffic determine usable node bandwidth. Adding ports does not guarantee a proportional application speedup.

Record `nvidia-smi topo -m`, `ibstat`, `ibv_devinfo` and network/fabric configuration. ConnectX-7 is an NDR-capable generation, not inherently an HDR-only adapter. Link rate is the negotiated rate across adapter, switch and cable. Use the system supplier's supported driver/firmware installation procedure rather than a truncated historical download URL.

## Hierarchy moves work; it does not erase node traffic

One possible hierarchy is local reduce-scatter, cross-node AllReduce of corresponding shards, then local all-gather. NCCL's actual algorithm is version/topology dependent; this is a useful model, not a promised trace.

For **two nodes with eight GPUs each**, every rank starts with S bytes. A local reduce-scatter leaves eight S/8 shards per node. Each corresponding pair exchanges its S/8 partial result. Each participating rank sends S/8 across the network in the ideal two-node reduction; the eight ranks together send **S per node**, not S/8. Eight independent NICs may parallelize this; one shared NIC remains a node-level bottleneck. Generalize the cross-node ring volume to `2(K−1)S/(K×G)` per rank for K nodes and G local shards, assuming that particular hierarchy.

**Check:** with S=1 GB and eight shards, each rank exchanges 125 MB; all ranks in a node exchange 1 GB outgoing total. A single 25 GB/s NIC needs at least 40 ms for that outgoing data, ignoring overhead and local collectives.

## GPUDirect RDMA and SHARP are capabilities to verify

GPUDirect RDMA allows a compatible NIC to access GPU memory without staging the payload through host memory. The CPU still initializes, registers and controls work, and the data still follows the actual PCIe/NVLink topology. A nonzero `ibv_devinfo dm_size` does **not** prove GPU-memory RDMA. Kernel peer-memory or supported DMA-BUF paths, driver versions, container access and topology matter. Save NCCL transport logs and run a supported GPU-memory RDMA test. [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/), [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).

SHARP can perform supported reductions in the network; NVLink reduction hardware is a related but different local-fabric path. Selecting `NCCL_ALGO=CollNetDirect` alone does not install a network plugin, configure a SHARP manager or provision compatible switches. Establish a working default baseline, then compare eligible offload configurations with the same operation and data. Do not promise a fixed 50% reduction in time. [NVIDIA SHARP documentation](https://docs.nvidia.com/networking/display/sharpv3150).

## Launch with torchrun

Example: two nodes, eight GPUs each. Both run the same script from [chapter 4](04-Configuration-and-Tuning.md), which selects CUDA device using `LOCAL_RANK`. Replace `node0.example` with an address reachable by both nodes.

```bash
# On node 0:
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
    --master-addr=node0.example --master-port=29500 nccl_smoke.py

# On node 1:
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=1 \
    --master-addr=node0.example --master-port=29500 nccl_smoke.py
```

There are 16 global ranks, local ranks 0–7 on each node, and one worker per GPU. The correct SUM output is `16×17/2=136` everywhere. Port reachability proves rendezvous connectivity only; inspect the payload transport separately.

## Slurm: one launcher per node

A common error launches eight Slurm tasks **each spawning eight torchrun children**, producing 64 workers per node. Use one Slurm task per node when torchrun spawns the local GPU workers. The partition, GPU resource syntax and CPU allocation are site-specific.

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
set -euo pipefail
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c '
    exec torchrun --nnodes="$SLURM_NNODES" --nproc-per-node=8 \
        --node-rank="$SLURM_NODEID" --master-addr="$MASTER_ADDR" \
        --master-port="$MASTER_PORT" nccl_smoke.py
'
```

The single-quoted command expands `SLURM_NODEID` inside each remote task. Choose a job-unique rendezvous endpoint according to site policy when nodes are shared. Follow site GPU binding/visibility rules. [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html), [Slurm srun](https://slurm.schedmd.com/srun.html).

## MPI nccl-tests uses a different launcher

Build nccl-tests with `MPI=1` and a compatible MPI development installation. This example starts 16 MPI ranks, each controlling one GPU (`-g 1`). A standalone `-g 1` without MPI measures only one participant and is not an inter-node test.

```bash
make -j MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
mpirun -np 16 --host node0:8,node1:8 -x NCCL_DEBUG=INFO \
    ./build/all_reduce_perf -b 8M -e 256M -f 2 -g 1 -n 20 -w 5
```

Use your cluster's MPI launcher, CPU binding and environment policy; record them. Ensure matching binaries/libraries and device permissions on all nodes. Increase message size only after correctness and memory checks. [nccl-tests build and launch options](https://github.com/NVIDIA/nccl-tests).

## Make communication savings explicit

Tensor parallelism usually has frequent intra-layer communication; keeping its group within a fast local fabric is a useful starting hypothesis. Pipeline parallelism transfers activations and backward gradients at stage boundaries; microbatch count, bubble time and activation size matter. Data parallelism synchronizes gradient buckets, not necessarily a single call. Experts add routing/dispatch traffic. Draw the groups and calculate bytes before assigning ranks.

Gradient accumulation can reduce synchronization frequency. DDP's `no_sync()` must enclose the **forward and backward** of unsynchronized microbatches. Synchronize the final microbatch, scale the total loss by the actual sample/token count, then step once. A short final accumulation group must not be divided by the nominal full group size. Different numbers of steps across ranks require an explicit uneven-input strategy; do not silently skip collectives. The [CPU lab](CPU-Collectives-Lab/Guide.md) demonstrates the related unequal-batch normalization error.

PowerSGD approximates gradients and can change training. For an 8192×8192 FP32 matrix: dense storage is 268.4 MB; rank-16 factors contain 262,144 numbers = 1.049 MB, a 256× **factor-storage** ratio. Error feedback, residual storage, additional reductions and convergence affect actual savings. A storage ratio is not a guaranteed communication or time speedup.

**Assessment:** produce a 2-node rank map, show why the Slurm script creates 16 rather than 128 workers, calculate both per-rank and per-node traffic, and submit logs proving which network transport the real job used. A CPU-only submission can complete the algebra and Gloo correctness lab while marking cluster measurements unexecuted.
