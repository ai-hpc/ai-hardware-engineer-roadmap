# Module 09 — Make distributed training recover correctly

A useful training system must preserve the intended update when it distributes work, and restore the intended experiment after interruption. Starting replacement processes is only one part of recovery.

**Before this lesson:** [mesh and batch accounting](08-Combining-LongContext-and-MoE.md). **Outcome:** run an existing two-process CPU correctness lab, calculate a checkpoint storage budget, and design a falsifiable resume test. Multi-node GPU training, scheduler recovery, and large distributed checkpoints are integration work not executed by this lesson.

## 1. Begin with a real distributed update on CPU

Complete [CPU collectives: two robots, one correct update](../NCCL-Deep-Dive/CPU-Collectives-Lab/Guide.md). Its included data, dependency lock, commands, serial oracle, and retained results run two local processes with PyTorch/Gloo. It checks AllReduce/AllGather and an actual gradient update against a serial reference, then deliberately rejects incompatible shapes.

The unequal-shard case has two samples on one rank and three on the other. Averaging local mean losses gives the smaller shard disproportionate weight. For default DDP gradient averaging, a global sample-mean objective requires each rank to use:

`local_loss_sum × world_size / global_valid_count`.

For language models, count valid predicted tokens rather than padded sequence slots. Accumulation over microbatches with unequal valid-token counts also needs the correct total denominator; averaging microbatch means has the same problem. Derive scaling from the intended objective and the actual reducer.

This CPU experiment proves its specified collective/update semantics. It does not measure NCCL, NVLink, RDMA, or a multi-node fabric. Move to the target GPU stack only after the small numerical comparison is understood.

## 2. Separate four kinds of failure

| Failure | Responsible mechanism | What it does not automatically restore |
|---|---|---|
| One worker process exits | Process launcher and process-group timeout | Model, optimizer, and data progress |
| A node leaves the allocation | Scheduler/resource manager and job policy | A replacement allocation or compatible mesh |
| A checkpoint write is incomplete | Checkpoint commit/integrity protocol | Missing shards or a coherent snapshot |
| A restarted job reads a different sample order | Data/state replay contract | The original training trajectory |

`torchrun` can restart worker groups within its configured policy. Ranks may change after restart, and elastic membership can change world size. It does not make application state durable or obtain a fresh scheduler allocation by itself. Recovery must explicitly reload state and recompute compatible batch/mesh assumptions. [PyTorch 2.14 launcher semantics](https://docs.pytorch.org/docs/2.14/elastic/run.html).

For Slurm or another scheduler, document the site's launch method, GPU allocation, rendezvous, network interface selection, shared storage, and resubmission policy. Interface names and MPI/PMI plugins are site-specific. A command containing missing training arguments or invented network device names is a template, not an executable course step.

## 3. Define the checkpoint boundary

A checkpoint should describe one coherent training boundary, such as **after optimizer step N and before selecting the next batch**. If saving in the middle of accumulation, either preserve accumulated gradients and the microstep state or explicitly restart from a prior completed update.

| State | Failure if omitted |
|---|---|
| Model parameters and persistent buffers | Wrong weights or missing running/router state |
| Optimizer tensors and step counters | Different next update despite identical model weights |
| Scheduler and mixed-precision scaler when used | Wrong learning rate or overflow behavior |
| Relevant CPU/device RNG states | Different dropout, augmentation, or sampling |
| Data cursor, shuffle/sampler state, packing buffer, mixture state | Repeated or omitted training examples |
| Global consumed valid tokens, accumulation position | Wrong schedule/batch interpretation |
| Code, model, tokenizer, data-manifest and mesh revisions | A loadable checkpoint with a changed experiment |

A raw rank number is not a stable identity for all of this state. Resharding must map model/optimizer tensors through the framework's supported layout, and data replay must use its stated logical sample identity. Record any approximate rather than exact resume policy.

PyTorch Distributed Checkpoint supports sharded state and resharding for supported state representations/planners. This is not a guarantee that arbitrary model changes, optimizer formats, or TP/EP/PP layouts can be exchanged. Checkpoint compatibility must be tested with the actual framework revision. Its asynchronous save has a staging boundary and a completion result: the call returning is not equivalent to a durable completed checkpoint. With custom staging, synchronization must prevent later training writes from changing the snapshot. [PyTorch 2.14 checkpoint API](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html).

Publish a checkpoint as resumable only after its expected shards and metadata are complete under the chosen storage backend. Keep the previous completed checkpoint while writing the next. A directory existing, or one rank reporting success, is insufficient evidence of global completion. Hashes detect accidental changes only when compared with a trusted manifest; they do not make an untrusted checkpoint authentic.

## 4. Calculate storage instead of quoting a universal save time

For an illustrative 70-billion-parameter model, storing FP32 parameters plus two FP32 Adam moments takes:

`70e9 × (4 + 4 + 4) bytes = 840 GB` in decimal units.

That accounting excludes buffers, metadata, and any additional saved state. A format that separately stores BF16 parameters, FP32 master parameters, and two FP32 moments uses 14 bytes per parameter, **980 GB**, before those additions. Implementations may avoid duplicated representations or reconstruct some state. Count what the chosen serializer actually writes; do not multiply replicated optimizer/model copies into unique checkpoint size unless it really serializes them all.

Writing 840 GB in 60 seconds needs at least **14 GB/s of aggregate payload throughput**. This is a lower bound, not a promise: host staging, metadata, contention, synchronization, and commit overhead remain. Per-GPU HBM bandwidth and nominal per-port network bandwidth are not shared-filesystem write throughput.

Measure checkpoint staging pause, total completion time, bytes written, peak host/device memory, and recovery time separately. Asynchronous I/O may reduce a training pause while increasing host memory and competing with data loading. A storage optimization is successful only if the resulting checkpoint still passes recovery tests.

Choose the save interval from measured save cost, interruption frequency, and acceptable lost work. If a failure can occur uniformly between completed saves spaced 30 minutes apart, the average lost completed progress is about 15 minutes under that simplified assumption. This excludes restart delay and failures during saving; use observed behavior for an operational policy.

## 5. Run a resume comparison that can expose missing state

Start with a small disposable job and deterministic data IDs. The comparison design is:

1. Run an uninterrupted reference for N+K updates; retain batch IDs, learning rates, losses, optimizer steps, and selected tensor values.
2. Run the same initial state to update N and complete a checkpoint.
3. Restart fresh processes from that checkpoint, then run K more updates.
4. Compare the next batch IDs and the full state/update progression with the uninterrupted reference under declared numeric tolerances.
5. Deliberately omit optimizer state or advance the data cursor in a separate negative-control copy. The test should detect both mistakes.

For a hand calculation, let scalar SGD with momentum use `v = 0.9v + g` and `w = w − 0.1v`. At the checkpoint, w=1 and v=2; the next gradient is g=3. Correct resume gives v=4.8 and w=0.52. Resetting momentum to zero gives v=3 and w=0.70. Identical restored weights did not restore the training process.

This design is a **planned recovery test** until it runs against the intended implementation. The course's CPU collectives lab executes distributed update correctness, not this checkpoint fault-injection protocol. A single matching loss is insufficient: different state can produce the same scalar loss. Conversely, permitted floating-point reduction differences can change last digits without changing the intended objective; specify the tolerance and investigate large divergence.

Only after same-layout recovery passes should you test a supported mesh change. Verify data IDs and effective global batch again. Test worker interruption inside a disposable allocated job, rather than treating a production node failure as a teaching exercise. Keep failure artifacts and launcher logs with the recovery report.

## 6. Observe the right signals

Monitor valid tokens per second, step-time tails, finite losses/gradients, memory peaks, data wait, expert assignment/drop counts, and completed checkpoint age. Rank-local losses need not be equal because ranks can receive different data and routed experts. Compare the correct globally weighted objective and invariants for replicated state instead of imposing equal rank-local losses.

Before interpreting a stall, identify which ranks entered which operation and which dependency failed. A timeout should surface the fault and preserve useful logs; arbitrarily lengthening it can hide a missing participant. The CPU lab's incompatible-shape preflight works only while all peers are alive to participate.

## Exercise and answer check

A job restarts successfully and resumes from a model-only file. Its first loss matches the pre-interruption loss, but optimizer moments and the packing buffer were reset. **Is recovery validated?** No. Show the momentum calculation above and explain how resetting the buffer can duplicate or omit tokens. Submit the CPU collectives evidence, your checkpoint-state inventory, actual serializer byte accounting, and the planned or measured resume comparison with its status clearly labeled.

**Continue:** [Module 10 — Make a defensible capability claim](10-General-Purpose-Model.md).
