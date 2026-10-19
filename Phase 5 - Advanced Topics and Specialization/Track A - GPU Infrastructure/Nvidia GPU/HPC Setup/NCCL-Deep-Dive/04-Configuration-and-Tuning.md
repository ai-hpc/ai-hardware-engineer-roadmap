# 04 — Configure NCCL by testing one hypothesis

**Outcome:** identify a transport, select the correct process-local device, and test one configuration change. Checked against NCCL **2.32.3** documentation on 2026-10-18. A framework wheel may bundle an older NCCL: record its actual version before copying settings. GPU commands below are unexecuted hardware exercises.

## Begin with the default configuration

NCCL has environment settings and communicator configuration APIs; environment variables are not its only interface. Tuning overrides can prevent a newer runtime from choosing a better algorithm. Record inherited `NCCL_*` and `TORCH_NCCL_*` variables, then compare an explicit clean baseline inside your allocated job. Do not modify shared cluster configuration.

```bash
printenv | sort | grep -E '^(NCCL|TORCH_NCCL)_'
nvidia-smi topo -m
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
```

Debug a small reproducible run with `NCCL_DEBUG=INFO` and `NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log`. `%h` and `%p` prevent hosts/processes overwriting each other's logs. Preserve all ranks' logs. `NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET` narrows output; exact messages and channel counts vary by version. Restore ordinary verbosity after diagnosis.

## A complete CUDA smoke test

Save as `nccl_smoke.py`, then launch **one process per GPU**. Global rank can exceed the number of local GPUs; device selection uses `LOCAL_RANK`.

```python
import os
from datetime import timedelta
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=60))
try:
    rank, world = dist.get_rank(), dist.get_world_size()
    value = torch.tensor([rank + 1.], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    torch.cuda.synchronize(local_rank)
    assert value.item() == world * (world + 1) / 2
    print(f"rank={rank} local={local_rank} sum={value.item()}", flush=True)
finally:
    dist.destroy_process_group()
```

```bash
NCCL_DEBUG=INFO torchrun --standalone --nproc-per-node=8 nccl_smoke.py
```

`torchrun` takes a script path here; `torchrun ... python -c ...` mistakenly treats `python` as the script. A collective forces actual communicator use; initialization alone may defer part of setup. For a CPU-only machine run the [executed Gloo lab](CPU-Collectives-Lab/Guide.md).

## What the important controls mean

| Setting | Meaning and common mistake |
|---|---|
| `NCCL_SOCKET_IFNAME='=eth0'` | Exact IP interface match; plain `eth` is a prefix. Bootstrap/interface selection does not prove RDMA data transport. |
| `NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1'` | Exact HCA names and ports. Multiple HCAs are eligible paths, not automatic OS bonding. Discover actual devices first. |
| `NCCL_IB_DISABLE=1` | Disable NCCL's IB/RoCE verbs transport for a diagnostic comparison; socket Ethernet and RoCE are different transports. |
| `NCCL_P2P_LEVEL=LOC` | Disable P2P; it does **not** mean “same CPU socket.” |
| `NCCL_P2P_LEVEL=NVL/PIX/PXB/PHB/SYS` | Increasing permitted topology distance. These are alternatives, not a literal slash-separated value. A setting cannot make unsupported peer access work. |
| `NCCL_NET_GDR_LEVEL` | Controls permitted GPU–NIC distance for GPUDirect RDMA. Prefer documented string values, not changing legacy integers. Leave automatic selection unless testing evidence. |
| `NCCL_IB_GID_INDEX` | Device-specific RoCE GID index. Index 3 does not universally mean RoCEv2. Modern NCCL can select it automatically. |
| `NCCL_IB_TC`, `NCCL_IB_SL` | Match the fabric administrator's QoS policy; an arbitrary value such as 106 does not create a correctly configured lossless fabric. |
| `NCCL_IB_TIMEOUT` | Verbs exponent: 4.096 µs × 2^value. Value 23 gives about **34.36 seconds**, before retry behavior. Not a training step watchdog. |
| `NCCL_ALGO`, `NCCL_PROTO` | Diagnostic restrictions on supported algorithms/protocols. Removing NVLS or another eligible algorithm may regress performance. No universal message-size thresholds. |
| `NCCL_MIN_CTAS`, `NCCL_MAX_CTAS` | Bounds on communication GPU work. More can consume compute resources; names/availability depend on release. Older channel controls are legacy. |

The official reference warns against retaining debugging overrides in production. [NCCL environment reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

## Timeouts belong to the correct layer

`NCCL_TIMEOUT` and `NCCL_SOCKET_TIMEOUT` are not documented NCCL controls in this reviewed release. Set the PyTorch process-group `timeout` explicitly, as above. PyTorch's `TORCH_NCCL_ASYNC_ERROR_HANDLING` and watchdog/flight-recorder settings belong to **ProcessGroupNCCL**, not generic NCCL. Network retry settings and elastic rendezvous timeouts affect other layers. Increasing every timeout can conceal a missing rank rather than fix it. [PyTorch NCCL variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html).

## A controlled tuning experiment

1. Save baseline raw nccl-tests results, correctness checks, topology and version manifest.
2. State a prediction: for example, “disabling RDMA will reduce large-message bandwidth across these nodes.”
3. Change one variable for one job, repeat the same sweep, and remove it afterward.
4. Inspect transport logs as well as timing. An unexpected no-change result may mean RDMA was already unavailable.
5. Test the real application: a faster isolated collective may occupy SMs needed for useful compute.

For DDP, a larger `bucket_cap_mb` can reduce launch count but delay communication until more gradients are ready. A smaller bucket can begin earlier but add overhead. `static_graph=True` requires stable graph/use of parameters; it is not a generic speed switch. Use a trace of **backward computation overlapping ready gradient buckets**, followed by completion before the optimizer step.

```bash
nsys profile --trace=cuda,nvtx --stats=true --output=nccl_profile \
    torchrun --standalone --nproc-per-node=8 train.py
```

Check your installed Nsight Systems trace options before adding NCCL-specific tracing. A CUDA event or synchronization used for isolated timing can remove overlap; keep a separate end-to-end trace.

**Assessment:** explain why forcing `P2P_LEVEL=NVL` is unsuitable as a universal recipe for L40S (it has no NVLink), why `LOCAL_RANK` selects the device, and why 8 channels is not guaranteed on an 8-GPU node. Submit the baseline, one controlled variation and the observed transport evidence.
