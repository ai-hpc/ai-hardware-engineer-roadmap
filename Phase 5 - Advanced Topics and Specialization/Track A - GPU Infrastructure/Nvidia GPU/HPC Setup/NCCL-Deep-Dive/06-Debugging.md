# 06 — Diagnose distributed failures from evidence

**Outcome:** reduce a failure to a small reproducible cause without damaging other jobs. Source review: 2026-10-18. Hardware troubleshooting below is source-reviewed, not executed on a GPU cluster.

## First distinguish three observations

- **Stall:** a rank has not reached or completed the expected collective. Missing workers, different control flow, network loss and stalled earlier CUDA work are competing explanations.
- **Reported error:** preserve the first failing rank's exception and all other rank logs; later peer failures can be consequences.
- **Wrong result:** begin with operation, shape, dtype, rank ordering and normalization. Numerical overflow or application bugs are more direct hypotheses than unexplained “silent hardware corruption.”

A process-group timeout bounds part of the wait. `NCCL_TIMEOUT` is not a documented general NCCL watchdog setting. Use [chapter 4](04-Configuration-and-Tuning.md) for layer-specific controls.

## Reduce the problem

1. Save version manifest, launcher command, topology, per-rank logs and the smallest input reproducing the issue.
2. Run one GPU without distributed execution. Then two GPUs on one node. Then the same small collective across two nodes.
3. Compare the first failing boundary. A correct local test and failing cross-node test narrows the transport/launch hypotheses, but does not prove hardware failure.
4. Check return codes and `#wrong` in nccl-tests before interpreting timings.

Use an allocated job's own processes and temporary resources. Do not run broad `pkill` patterns or remove `/dev/shm/nccl-*` while other jobs may own those objects. Stop the specific job through its scheduler, preserve evidence, and let the site administrator handle shared resources.

## Check rank contracts before payload communication

For a same-shaped SUM AllReduce, ranks must agree on dtype, shape, operation and ordering. A local assertion does not validate other ranks; a coordinated preflight can gather small trusted metadata first. The [CPU lab](CPU-Collectives-Lab/Guide.md) intentionally mismatches one rank's shape and confirms that both ranks reject it **before** entering the payload collective.

```python
# Wrong: only some ranks may enter the collective.
if local_loss > threshold:
    dist.all_reduce(tensor)
```

```python
# Coordinated decision; all ranks enter the same preliminary collective.
flag = torch.tensor([int(local_loss > threshold)], dtype=torch.int64, device=device)
dist.all_reduce(flag, op=dist.ReduceOp.MAX)
if flag.item():
    dist.all_reduce(tensor)
```

Even this cannot recover a rank that already crashed. All participants must execute the same sequence on the same group. Unequal dataloader exhaustion and exception handling are common causes. The preflight object gather in the local lab uses pickle; it is for trusted peers, not untrusted network input.

## Capture stacks correctly

`faulthandler.enable()` handles fault signals; it does not by itself register SIGUSR1 or schedule a timeout. Add the following to a Python process you control on a platform with SIGUSR1:

```python
import faulthandler
import signal
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
faulthandler.dump_traceback_later(60, repeat=True)
# Later, when the diagnostic window ends:
# faulthandler.cancel_dump_traceback_later()
```

Signal only that known process after registration. A Python stack may show a wait without revealing the earlier failing CUDA kernel. `CUDA_LAUNCH_BLOCKING=1` is a diagnostic run that changes scheduling, not a production optimization. `torch.cuda.check_error(torch.cuda.current_device())` is incorrect: a device index is not a CUDA error code. [Python faulthandler](https://docs.python.org/3/library/faulthandler.html).

## Separate health indicators from conclusions

```bash
nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv
nvidia-smi topo -m
nvidia-smi nvlink --status -i 0
dmesg | grep -E 'NVRM|Xid|PCIe|AER'
```

Counters may be cumulative, unsupported, historical or corrected. Compare changes during the failing run and correlate with vendor diagnostics. A single nonzero counter is not a sufficient replacement diagnosis. Some commands require site permissions; capture what is available.

| Xid | Meaning to investigate |
|---|---|
| 43 | GPU stopped processing; often application-induced, not a synonym for NVLink failure |
| 48 | Double-bit ECC error; follow the model-specific recovery procedure |
| 79 | GPU fallen off the bus; investigate power/PCIe/system evidence |
| 94 | Contained ECC error; not “container isolation violated” |
| 119 | GSP RPC timeout |

Use the exact GPU/driver generation and the [official Xid catalog](https://docs.nvidia.com/deploy/xid-errors/index.html) for recovery actions. Resetting devices, changing ECC, unloading a driver or rebooting affects other jobs and belongs to the cluster's maintenance process.

## Network isolation

Verify both endpoints and the exact HCA port. `ib_write_bw` needs a server process and a matching client; a single command alone does not test a fabric. A successful host-memory RDMA test does not prove GPU-memory RDMA. Record link rate, GID/transport, MTU and topology, then check GPU-memory transfer and NCCL separately. Do not hard-code a RoCE GID index or traffic class copied from another cluster. [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).

## Recovery is a training-state problem

A launcher restart recreates processes; it does not restore useful progress automatically. A resumable checkpoint must bind model parameters, optimizer, scaler/scheduler, step, RNG state, data position and relevant partition metadata. Sharded jobs need a consistent distributed save protocol. Publish a checkpoint only after all required files are durably complete. A changed world size may require supported resharding and a decision about effective batch size.

Test recovery by saving, completing one more deterministic step, restoring into new processes and comparing that step with the uninterrupted reference. Weight-only reload is insufficient for Adam equivalence. After a failed CUDA/NCCL operation, terminate the affected job and restart from a known complete checkpoint rather than continuing with potentially inconsistent state. [torchrun restart behavior](https://docs.pytorch.org/docs/stable/elastic/run.html).

**Exit exercise:** explain a case where doubling the timeout helps (a valid but slow operation) and one where it only delays failure (one rank never enters). Submit a minimal reproducer, a hypothesis with disconfirming evidence, and the smallest change that fixes the actual failure.
