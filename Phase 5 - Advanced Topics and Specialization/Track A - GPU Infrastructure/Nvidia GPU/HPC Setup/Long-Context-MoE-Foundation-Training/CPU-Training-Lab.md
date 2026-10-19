# CPU Lab — From a Cost Ledger to a Trained MoE and Real Dispatch

**Time:** three sessions of 45–60 minutes. **Requires:** Python 3.12 for the ledger; an isolated CPU PyTorch environment for training and Gloo. No GPU, downloaded model weights or private dataset is needed. **Scope:** explicit arithmetic, a small trained regression network, and real two-process dispatch; this does not train or benchmark a long-context language model.

Start with [Module 01](01-Long-Context-Bottlenecks.md), then read [routing](04-MoE-Fundamentals.md) and [dispatch](05-MoE-Systems-Infrastructure.md) alongside the experiments. All commands below run from **this course directory**, the folder containing this page and `labs/`.

## Session 1 — Predict and check the ledger

```bash
python3 labs/ledger.py > /tmp/long-context-ledger.json
python3 -m unittest discover -s labs -p 'test_ledger.py' -v
```

There are ten standard-library tests. For three tokens, hidden width four, two query heads, one KV head, head width two and FFN width eight, predict:

- Six causal query-key pairs, giving `4*6*4 = 96` pair FLOPs.
- Q/K/V/output projections totaling `288` FLOPs.
- Nongated FFN totaling `384` FLOPs.
- BF16 Q/K/V/output tensors totaling `72` bytes; one FFN intermediate is `48` bytes.

The tests check those hand calculations, causal/dense differences, GQA, exact integer capacity, the three drop denominators and current dense/expert grid cardinalities. They reject malformed shapes and invalid divisions. Capacity uses exact arithmetic on the supplied factor's decimal representation: even a positive factor of `5e-324` admits a capacity of at least one, and token counts beyond `2**53` do not silently round down. The printed large-sequence table lists tensor sizes and useful forward arithmetic, not peak memory or GPU execution.

**Variation:** double the KV-head count. Predict which projection/storage terms change and which attention terms stay fixed. Then explain why a global batch of 1,024 does not imply 1,024 simultaneous copies of every activation on each pipeline stage.

## Session 2 — Actually train the experts

Install once, or reuse an environment that already contains the same pinned CPU framework:

```bash
python3.12 -m venv /tmp/long-context-course-venv
/tmp/long-context-course-venv/bin/python -m pip install -r labs/requirements-torch.txt
/tmp/long-context-course-venv/bin/python -m unittest discover -s labs -p 'test_*.py' -v
/tmp/long-context-course-venv/bin/python labs/tiny_moe.py --output /tmp/long-context-moe
```

The framework pin is **PyTorch 2.14.0+cpu**. The complete local suite runs 19 tests: ten ledger checks, eight PyTorch checks and one integration test that launches four two-process Gloo jobs. Without PyTorch, the nine framework/integration tests skip explicitly; Gloo also requires a supported distributed CPU build. A skipped test is not executed validation.

Open [tiny_moe.py](labs/tiny_moe.py). Four experts each map four inputs through a 16-unit GELU layer back to four outputs. The router chooses two experts per sample. The synthetic target is `[sin(x0), .3*x1², x2*x3, tanh(x0+x3)]`. Training uses 512 generated samples with seed 41; validation uses 256 independent samples with seed 42. Initialization is seed seven. The target formula is intentionally available so the experiment is inspectable.

The script performs 120 Adam updates for each auxiliary weight `0`, `.01` and `.5`, writes training history and saves actual model state files. It reloads each saved state into a new module and compares predictions on a third probe set. Use only locally created/trusted artifacts with the specified `weights_only=True` loader.

The [recorded run](labs/training-run.json) measured:

| Auxiliary weight | Initial validation MSE | Final validation MSE | Reload max error |
|---:|---:|---:|---:|
| 0 | 0.638232 | 0.017737 | 0 |
| .01 | 0.638232 | 0.017776 | 0 |
| .5 | 0.638232 | 0.019364 | 0 |

The zero-auxiliary-loss run did not collapse; it assigned roughly 22–28% of choices to each expert. This small observation does not establish that balancing losses are unnecessary in large models. It disproves a claim that collapse must occur in this exercise.

The tests compare sparse forward **and backward** against an independently structured all-expert oracle. A finite difference checks a router derivative away from selection boundaries. A separate top-one case demonstrates that normalizing its sole gate to one removes the ordinary task-loss router gradient. Training validation checks a substantial held-out loss reduction rather than merely checking that a loop returned. A hand-calculated two-expert case independently checks highest-gate admission and retained weights; malformed dimensions, invalid coefficients and nonfinite intermediate values fail explicitly.

### Add capacity pressure

```bash
/tmp/long-context-course-venv/bin/python labs/tiny_moe.py \
    --capacity-factor 0.5 --output /tmp/long-context-moe-capacity
```

With 256 validation tokens, top-two, four experts and factor `.5`, capacity is 64 per expert. The recorded `.01` run drops **50% of assignments**, affects **91.796875% of tokens**, and leaves **8.203125% of tokens with no admitted expert**. Validation MSE becomes approximately `0.068784`. See [the complete capacity run](labs/capacity-run.json).

The output combines remaining weighted expert contributions without renormalizing them. Admission compares samples within the current batch, so capacity-limited predictions can depend on batch composition. Keep the validation batch definition fixed and report it. This is a teaching experiment about the drop policy, not an endorsed production tolerance.

**Variation:** predict the capacity at factor one before running it. Repeat a new initialization and keep all results; do not tune on a final test set or silently select the best seed. A routing-balance improvement is useful only with the stated task and system tradeoffs.

## Session 3 — Exchange actual assignments between processes

```bash
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 /tmp/long-context-course-venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    labs/dispatch_gloo.py --output /tmp/long-context-dispatch.json

GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 /tmp/long-context-course-venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    labs/dispatch_gloo.py --skew --output /tmp/long-context-dispatch-skew.json
```

These jobs use loopback on one Linux host, with a 20-second process-group timeout. They use simple affine experts and predetermined routes to make the expected answer independent of the learned network. **The trained network above is not distributed by this script.**

Inspect [dispatch_gloo.py](labs/dispatch_gloo.py): first exchange counts, then inputs, then return computed outputs with reversed split vectors. Finally use the retained source permutation to combine contributions. The top-two case has different source batch sizes, and the skew case includes a zero-sized partition. All cases compare every result against a serial calculation, exactly for these representable values.

Two additional cases exercise a whole idle expert and a whole empty source batch:

```bash
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 /tmp/long-context-course-venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    labs/dispatch_gloo.py --empty-expert --output /tmp/long-context-empty-expert.json
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 /tmp/long-context-course-venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    labs/dispatch_gloo.py --empty-source --output /tmp/long-context-empty-source.json
```

An empty source still participates in every collective and can own an expert that receives work. An idle expert returns an empty tensor with the correct feature dimension; it does not skip the collective. The tests verify both split totals and pairwise send/receive counts.

[Top-two record](labs/dispatch-run.json) · [empty-partition record](labs/dispatch-skew-run.json) · [idle-expert record](labs/dispatch-empty-expert-run.json) · [empty-source record](labs/dispatch-empty-source-run.json). All four executed on CPU/Gloo and returned zero error. They do not establish GPU throughput, distributed backpropagation or failure recovery. Continue to the [real distributed-gradient lab](../NCCL-Deep-Dive/CPU-Collectives-Lab/Guide.md) to study rank-local versus globally weighted losses.

## Submit evidence you can explain

Keep the ledger, test output, all three training histories, saved/reloaded models, capacity variation and all four dispatch reports. Explain one calculation by hand, one unexpected observation, and one failure the tests catch. Add a diagram of source token → expert owner → original token. The [validation manifest](labs/validation.json) records the executed environment, file hashes and the boundary between these CPU experiments and unexecuted cluster training.

[Course overview](README.md) · [Data and evaluation](06-Adaptive-Data-Pipelines.md) · [Infrastructure](09-Distributed-Training-Infrastructure.md)
