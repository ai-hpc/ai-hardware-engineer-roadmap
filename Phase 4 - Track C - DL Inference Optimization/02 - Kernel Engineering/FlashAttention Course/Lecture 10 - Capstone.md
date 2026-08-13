# Lecture 10 — Capstone

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Pick one focused kernel-path change, implement it, prove it correct, benchmark it, and write the report a hiring manager (or your future self) can actually read.

**Prerequisites:** Lectures 1–9, including the correctness harness from Lecture 7 and the benchmarks from Lectures 5/6/8/9.

**Artifact:** A patch on top of `flash-attention`, a correctness report (harness output), a benchmark report (CSV + plot), and a 1–2 page write-up. Together: a single PR-style artifact you can show.

---

## Why it matters

The hardest skill to demonstrate as a kernel engineer is the loop of "find a real bottleneck, write a working change, prove it does not break anything, prove it is faster." This capstone is that loop in miniature on a codebase that matters.

Do not try to outperform Tri Dao on the main kernel — that is not the point. The point is to ship a credible, well-measured, well-documented change.

---

## Mental model: what counts as a good capstone

A good capstone is **focused**, **measurable**, and **honest**. It has:

1. A specific path you changed, with a short paragraph of why.
2. A correctness section that runs your Lecture 7 harness on the changed path.
3. A benchmark section that compares before-and-after on at least one shape range.
4. A "limitations" section. If the patch only wins for `d = 64`, say so. If it slows down small batches, say so.

A bad capstone reimplements FA2 from scratch and ends up at "I got within 4× of upstream." Skip that. Pick a small, real change instead.

---

## Capstone options (pick one)

### Option A — Hand-fused mask path

Pick a mask the upstream kernel does not handle in-fused form (custom block-sparse, document-block diagonal for packed-sequence training, or a sliding-window + global-tokens hybrid). Implement it as a Triton kernel in the style of `flash_attn/flash_attn_triton.py`. Compare correctness against a math reference. Compare speed against the upstream path that falls back to attention with `attn_bias`.

Why this is good: you exercise the masking machinery, you produce a kernel that has a real use case (long-context with global tokens, e.g. retrieval-augmented setups), and the benchmark gap is large.

### Option B — Decode-step paged-KV optimisation

In the style of the cacheon-sglang-miner project we worked on, take `flash_attn_with_kvcache` and produce a fused path that does in-kernel rotary plus an extra small optimisation — e.g. preloading the next layer's `K, V` while the current attention runs, or skipping a redundant H2D copy in the calling layer. Benchmark decode TPS at long context (16k–32k).

Why this is good: directly applicable to serving stacks; the harness is small (decode shape only); the win is end-to-end TPS, not micro.

### Option C — FA3 numerics audit at long context

Generate stress inputs (e.g. extreme score magnitudes that hit subnormal exp(-large)) and audit the bf16 / fp16 / fp8 forward outputs of FA3 against an fp64 reference for `N ∈ {8k, 16k, 32k, 64k}`. Build a per-row error CDF (not just a max-error number). Identify the shape where the dtype tolerance is exceeded; check whether `--rotary` / causal mask / sliding-window changes it.

Why this is good: numerics audits never get done and they always find something. Outputs are publication-quality plots.

### Option D — Triton port of one FA2 path

Take one upstream FA2 head-dim path (e.g. `d = 64, causal=True, no dropout`) and write the Triton equivalent (start from `flash_attn_triton.py` which is incomplete and outdated). Match upstream within ±20% on a sweep. Document where Triton's autotuner picks tile sizes vs upstream's hand-picked traits.

Why this is good: you learn the Triton expression of online softmax, and you produce a portable kernel for non-NVIDIA backends (Triton supports AMD).

### Option E — A real PR

If you have one *actually* small, *actually* useful patch (a doc fix, a missing varlen mask combo, a CMake clean-up, a numerics-tolerance bug surfaced by your harness), propose it upstream. The capstone artifact is the GitHub PR plus your correctness/benchmark addendum.

This option is the most useful but also the highest variance — your timing is in the hands of reviewers.

---

## Build it

For whichever option:

1. **State the change in one sentence.** "I am adding an in-kernel sliding-window-plus-global-tokens mask path for the FA2 forward at head_dim 128, causal=True."
2. **Implement.** Smallest possible diff. Do not refactor neighbouring code.
3. **Run the Lecture 7 harness.** Add at least one shape that exercises your change. All rows must pass at the documented tolerance.
4. **Run a benchmark.** At least three shapes spanning small / medium / large in your target dimension. Use median-of-N timings after warmup.
5. **Write the report** (template below).

---

## Report template

```
# <Capstone title>

## What
One paragraph. What did you change and where?

## Why
One paragraph. What workload or bottleneck motivated this?

## How
Two-to-five paragraphs of technical content:
- Algorithmic / scheduling change (mental model from Lecture 2 / 3 / 6 / 9)
- File-level diff summary (`path/to/file.cu`: added X, modified Y)
- Edge cases handled (masks, dtypes, head_dims you support and skip)

## Correctness
- Reference: PyTorch SDPA math backend OR the NumPy harness from Lecture 2.
- Harness output table (paste the JSON from Lecture 7).
- Tolerance bands used and why.

## Performance
- Hardware: GPU model, CUDA version, driver, peak bf16 TFLOPs.
- Sweep table: shape × {before, after, % delta, achieved TFLOPs, achieved HBM BW}.
- One plot.

## Limitations
- Shapes that regress.
- Shapes that are not supported by the new path.
- Numerics caveats at the edges.

## What I would do next
Two or three bullets. Honest about scope.
```

---

## Use it in the real stack

A capstone that integrates with a real serving stack is worth 10× one that just lives in a notebook. If you went with Option B, plug it into the cacheon-sglang-miner repo or a vLLM fork, run an SN14-style correctness sweep, and report end-to-end TPS not just kernel time. That is the artifact that gets you hired into MTS Kernels or DL Inference Optimization roles.

---

## Measure it

You already have all the measurement tools from earlier lectures:

- Lecture 1 roofline notebook → confirm your change moves the kernel in the expected direction (memory-bound → compute-bound or vice versa).
- Lecture 4 Nsight Compute metrics → confirm tensor-core utilisation, HBM traffic, occupancy.
- Lecture 7 harness → correctness.
- Lecture 8 paged-decode bench → if Option B.

Use all of them. Drop the raw outputs in an appendix.

---

## Ship it

Final deliverable in `flash-attn-course/capstone/`:

1. `README.md` — the report.
2. `patch.diff` — the actual code change as a git diff.
3. `correctness_report.json` — harness output.
4. `benchmark.csv` and `benchmark.png` — your sweep.
5. (Optional) `ncu_before.ncu-rep`, `ncu_after.ncu-rep` — profiling reports.

If you can hand someone that directory and they can understand within 10 minutes what you did and why it works, you have finished the course.

---

## What good looks like

- Honest scope. A 30-line patch with a clean benchmark beats a 1000-line patch with vague numbers.
- Real workload framing. "This helps decode TPS at 32k context on H200" beats "this kernel is 1.3× faster on synthetic inputs."
- Reproducibility. Anyone with your patch and your benchmark script must be able to reproduce your numbers within ~10%.
- A line in the limitations section. If your report has no limitations section, you have not measured enough.

---

## Where to go after this course

- **Kernel engineering depth:** CUTLASS examples, CUDA samples, NVIDIA HPC SDK, PTX manual, write a Hopper-specific GEMM from scratch.
- **Inference serving depth:** vLLM internals, SGLang's RadixAttention, TensorRT-LLM plugins, the cacheon-style production kernels.
- **Compiler depth:** Triton internals (`triton/python/triton/compiler/`), MLIR-based GPU compilers, the Mosaic / Pallas GPU backends.
- **ML systems research:** speculative decoding, MoE attention variants, multi-query rolling attention, FP4 numerics, infinite-context architectures.

The FA codebase will keep getting more interesting (FA4, FA-MoE, FA-paged-v2). You are now equipped to follow along.

---

## Related pages

- [Lecture 7 — Backward pass and validation](Lecture%2007%20-%20Backward%20Pass%20and%20Validation.md)
- [Lecture 9 — Hopper / FA3 / FA4](Lecture%2009%20-%20Hopper%20FA3%20FA4.md)
- [02 — Kernel Engineering](../Guide.md)
- [DL Inference Runtimes and Deployment](../../05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md)
