# Lecture 18 — Training Infrastructure, Distributed Execution, and Reproducibility

Build a training run whose resource use, numerical behavior, and restart state are understandable before scaling to expensive VLA experiments.

**Prerequisites:** Lectures 14–17; GPU memory, optimizers, and Python training loops. **Effort:** setup 1 hour, lab 4 hours, analysis 2 hours. **Hardware:** CPU for resume and accumulation checks; one GPU for profiling; two or more GPUs for the optional distributed extension. **Stack/roles:** compute hardware, communication, runtime, and model training; ML infrastructure and AI hardware performance engineers. **Artifact:** a profiled training run, resource model, restart test, and reproducibility manifest.

## Count the memory that actually exists

Parameter storage is only one term:

$$
M_{peak}\approx M_{weights}+M_{gradients}+M_{optimizer}
+M_{activations}+M_{temporary}+M_{communication}.
$$

For an illustrative full-tuning recipe with BF16 weights and gradients, FP32 master weights, and two FP32 Adam moments, persistent storage is approximately \(2+2+4+8=16\) bytes per parameter. A 3-billion-parameter example therefore needs about 48 GB in decimal units for those components alone. Actual frameworks can omit the master copy, store gradients differently, or use other optimizers; this is a stated budget model, not a measured requirement.

Activations depend on image count/resolution, token length, action horizon, hidden sizes, batch size, and checkpointing. Padding three cameras when one is present may still cost computation unless the implementation avoids it. Frozen parameters can reduce optimizer states while leaving substantial activation or forward-pass cost.

## Data parallelism and sharding solve different problems

Ordinary data parallelism replicates parameters across workers and aggregates gradients. It increases throughput if communication and loading permit, but each worker still needs the replicated model. Sharding optimizer state, gradients, and parameters reduces persistent replication, at the price of communication and temporary materialization. ZeRO explains this progression; PyTorch's FSDP tutorial documents a concrete implementation. [ZeRO, 2019](https://arxiv.org/abs/1910.02054), [official FSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

Do not divide total memory by GPU count and declare the model fits. The largest gathered module, communication buffers, activations, and checkpoint-save behavior can dominate peaks. Likewise, all-gather latency over PCIe or Ethernet differs from an interconnect with higher bandwidth. Record topology and measure the collective critical path.

Tensor parallelism splits operations within layers; pipeline parallelism distributes layers; data parallelism distributes examples. These can be combined, but each adds partitioning constraints and communication. Start with the least complex layout that fits and meets the experimental budget. A toy model is often slower when distributed because launch and communication overhead exceed compute savings.

## Accumulation is a mathematical operation

With \(W\) workers, per-worker microbatch \(b\), and \(G\) accumulation steps, nominal global batch is \(WbG\). This equality does not guarantee identical optimization if examples have different numbers of valid action coordinates.

The correct global masked mean is total loss numerator divided by total valid denominator. Averaging microbatch means gives equal weight to microbatches, even when one contains twice as many valid targets. Under gradient-averaging distributed runtimes, scale local summed losses consistently with the globally reduced denominator and the runtime's averaging convention. Test against a single-process reference before scaling.

Mixed precision adds another sequence constraint: accumulation, loss scaling if used, gradient unscaling, clipping, optimizer step, and scaler update must occur in the intended order. Do not clip scaled gradients as though they were physical gradient magnitudes. The framework's official AMP examples provide API guidance; pin the version rather than assuming the moving documentation matches an installed runtime. [PyTorch AMP examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html).

## Worked example: an innocent mean changes the gradient

Microbatch A has ten valid errors, each 1; B has one valid error of 9. Averaging the two means gives \((1+9)/2=5\). The valid-coordinate mean is \((10+9)/11\approx1.727\). The discrepancy changes both reported loss and relative gradient weighting. Long episodes, missing joints, and padded horizons make this a realistic VLA concern.

For throughput, suppose a measured training step spends 40 ms decoding data, 60 ms computing, and 20 ms communicating with no overlap. Its duration is 120 ms. Doubling compute speed reduces the step to 90 ms, a 1.33× speedup, not 2×. Real overlap must be read from a trace; adding stage averages does not always equal the critical path.

## Reproducibility requires state, not only seeds

A resumable checkpoint includes parameters, optimizer moments, scheduler, precision scaler if used, random-generator states, step counters, sampler/streaming position, and any curriculum or data-mixture state. Save normalizers, preprocessing, tokenizer, action contract, code revision, dependency lock, dataset manifest, and model origin alongside it.

Seeds do not promise identical results across libraries, releases, GPU architectures, or nondeterministic kernels. Distinguish bitwise restart parity in a controlled environment from statistical reproducibility across environments. Framework documentation describes these limitations and deterministic options. [PyTorch reproducibility notes](https://docs.pytorch.org/docs/main/notes/randomness.html).

For claims, predefine the hypothesis, primary metric, holdout, compute/data budget, seed count, and stopping rule. Tune on development data; report final tests once the choice is fixed. Save failed runs and reasons for exclusion. A model chosen from many seeds and reported only at its best is a selection procedure that must be disclosed.

## Build, interrupt, and profile

**Exercise / pseudocode.** Extend a small learner from the [shared lab guide](Lab-Guide.md); the following distributed and mixed-precision work is a learner implementation exercise, not an already executed VLA training run.

1. Write an analytical memory table for weights, gradients, optimizer, and estimated activations. Compare with measured process/device peaks where available; explain missing terms.
2. Train for a fixed 200 optimizer steps with saved sample IDs. Compare a continuous run with a run stopped and resumed at step 100. Verify next-batch identity, learning rate, and parameter differences.
3. Compare a single large batch against accumulated microbatches using an intentionally uneven validity mask. Require gradient agreement within a declared numeric tolerance on a deterministic tiny model.
4. Profile loading, host-to-device copy, forward, backward, communication, optimizer, and checkpoint write. Report both startup and steady state; avoid including a one-time compile cost in every step estimate.
5. Change one factor: decode workers, activation checkpointing, precision, or sharding. Measure its memory and throughput effect while checking held-out output/loss agreement.
6. Optional: use two GPUs with a pinned, documented framework recipe. Verify distributed sample coverage and gradient behavior against the single-process reference before attempting a larger model.

Stop after the planned 200-step runs and one measured optimization. **Acceptance criteria:** resume restores sample and optimizer progression; masked accumulation matches the reference; measured peaks and throughput include methodology; manifests identify immutable artifacts; the optimization preserves required numerical behavior. CPU checks cannot establish multi-GPU performance or accelerator kernel support.

## Failure clinic and self-check

**Failure clinic:** restart loss spikes suggest missing optimizer or sampler state; speedup plateaus suggest input or communication bottlenecks; late OOMs can occur during checkpoint gathering; changing batch layout changing training suggests reduction or masking errors; one worker hanging suggests collective-order or uneven-input problems.

1. **Does sharding remove all memory replication?** No; temporary gathers, activations, and buffers remain.
2. **Why log sample IDs?** They reveal duplicate, skipped, or reordered data after sharding and restart.
3. **What makes a throughput comparison fair?** Same useful examples, masks, precision contract, warmup policy, and hardware configuration, with correctness checked.
4. **When are you ready for inference engineering?** The model package, action interface, held-out quality baseline, and reproducibility record are complete.

[Previous: RL post-training](Lecture-17.md) · [Course index](README.md) · [Next: closed-loop evaluation](Lecture-19.md)
