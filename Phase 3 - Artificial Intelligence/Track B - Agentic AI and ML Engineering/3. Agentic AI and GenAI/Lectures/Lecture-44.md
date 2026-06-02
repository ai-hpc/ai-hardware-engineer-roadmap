# Lecture 44 - TraceLens: Trace-Driven AI Performance Analysis

**Course:** [AI Agent Development 2026](../Guide.md) | **Previous:** [Lecture 43](Lecture-43.md) | **Next:** [Lecture 45](Lecture-45.md)

---

Performance optimization is **not a vibes problem**.

For AI systems, the hard part is often not collecting data. It is turning a huge profiler trace into a **specific diagnosis**:

```text
What was slow?
Where in the model did it happen?
Was the GPU computing, communicating, copying, or waiting?
Did the new kernel actually help?
Can we reproduce the slow op without the full model?
```

**TraceLens** is useful because it treats profiler traces as **structured data**, not screenshots for humans to manually inspect.

It consumes traces from frameworks such as PyTorch and JAX, then produces **summaries, comparisons, roofline-style metrics**, collective communication analysis, and minimal operation reproducers.

For this course, TraceLens is the missing evidence layer between:

```text
agent workload
  -> model server
  -> GPU kernels and collectives
  -> performance claim
  -> trace-backed diagnosis
```

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why raw AI profiler traces are hard to analyze manually.
2. Understand Trace2Tree as a hierarchical trace intermediate representation.
3. Use top-down performance breakdowns to locate bottlenecks.
4. Interpret GPU timeline, operator category, dispatch-level, and shape-level reports.
5. Understand TraceLens roofline metrics such as TFLOPS/s and TB/s.
6. Diagnose multi-GPU scaling with communication latency, bandwidth, and synchronization skew.
7. Use trace comparison to quantify regressions across hardware, drivers, kernels, or software versions.
8. Understand event replay as a way to produce minimal, IP-safe performance reproducers.
9. Apply TraceLens to OpenClaw-style agent serving and vLLM optimization workflows.

---

## 1. The profiler trace problem

Modern AI workloads generate large traces:

- Python module calls
- PyTorch or JAX framework operations
- CPU dispatches
- runtime launch events
- GPU kernels
- memory copies
- collective communication
- synchronization gaps

Tools such as **Perfetto** are useful for visual inspection, but the **manual workflow does not scale**.

The engineer ends up asking:

```text
Which kernels belong to this model layer?
Which aten op launched this slow kernel?
Is this memcpy from my code or from AMP/framework behavior?
Did communication overlap with compute?
Did a new software version shift time into a different op?
```

Raw traces are **too flat**.

They show events, but **not enough intent**.

TraceLens adds **structure**.

---

## 2. Trace2Tree: from flat trace to hierarchy

The central idea is **Trace2Tree**.

TraceLens converts **flat events into a tree**:

```text
Python module / function
  -> framework op
    -> CPU dispatch
      -> runtime launch
        -> GPU kernel
```

Example shape:

```text
nn.Module: Linear
  -> aten::linear
    -> aten::to
      -> aten::copy_
        -> elementwise kernel
    -> aten::addmm
      -> matrix multiply kernel
```

This matters because many expensive events are **hidden at the Python level**.

Common examples:

- automatic mixed precision casts
- implicit copies
- layout conversions
- unfused bias additions
- framework-generated elementwise kernels
- backend-specific convolution or attention kernels

Without a tree, a kernel is just a name and timestamp.

With a tree, the kernel has ownership:

```text
This exact module launched this exact framework op,
which launched this exact kernel.
```

That is the difference between **tracing and diagnosis**.

---

## 3. Top-down bottleneck analysis

TraceLens reports are designed to move from **coarse to precise**.

The useful progression is:

```text
GPU timeline
  -> operator category
  -> framework op
  -> unique input shape
  -> exact kernel or replay case
```

This is the right order.

Do not start with **kernel names**.

Start by asking **how the system spent time**.

---

## 4. GPU timeline: was the GPU doing useful work?

The GPU timeline report separates time into categories such as:

- computation
- exposed communication
- exposed memory copy
- busy time
- idle time
- total communication
- total memory copy

This answers the first diagnostic question:

```text
Is the GPU compute-bound, communication-bound, copy-bound, or idle?
```

Example interpretations:

```text
high computation time
  -> optimize kernels, shapes, dtype, fusion, algorithm

high exposed communication time
  -> improve overlap, collective strategy, tensor parallel layout, network path

high exposed memcpy time
  -> inspect dtype casts, host/device movement, layout conversions

high idle time
  -> inspect CPU dispatch, dataloader, synchronization, scheduling, dynamic shapes
```

The ROCm blog's Llama FSDP example shows a GPU timeline where the GPU is mostly busy, but **total communication** is still a large fraction of the run. That distinction matters: communication can be **present but hidden** under compute, or it can be **exposed and directly hurt** wall time.

---

## 5. Operator category view: what kind of work dominates?

After the timeline, group work by operation family:

- GEMM
- attention / SDPA
- convolution
- elementwise
- reduce
- Triton-generated kernels
- multi-tensor apply
- other backend kernels

This tells you **where to spend engineering effort**.

If GEMM and attention dominate most of the time, optimizing a small elementwise kernel will not move end-to-end performance unless it blocks fusion, causes copies, or sits on the critical path.

For transformer training and serving, the usual dominant categories are:

```text
GEMM
attention
communication
memory movement
```

The point is not that other ops are irrelevant.

The point is **prioritization**.

Performance work should **follow the trace**.

---

## 6. Dispatch-level view: stable names across backends

GPU kernel names change across:

- CUDA vs ROCm
- driver versions
- compiler versions
- library backends
- hardware generations
- autotuning choices

**Framework dispatch names** are often more stable.

Examples:

```text
aten::mm
aten::linear
aten::copy_
flash_attn::_flash_attn_forward
flash_attn::_flash_attn_backward
```

TraceLens uses this level to make comparisons more meaningful.

Instead of asking:

```text
Why did kernel igemm_fwd_gtcx3_... change?
```

you can ask:

```text
Did aten::convolution become slower?
Did aten::mm improve for this shape?
Did flash attention regress after the backend update?
```

That abstraction is important when comparing **CUDA and ROCm**, or comparing two software versions on the same platform.

---

## 7. Shape-level view: the real root cause

**Operator names are not enough.**

The same operation can have **very different performance** depending on:

- tensor dimensions
- strides
- dtype
- layout
- batch size
- sequence length
- head count
- head dimension
- padding
- transposition

TraceLens breaks down operations by unique input shape and arguments.

This is where **many real optimizations start**.

Example:

```text
aten::mm with shape A
  -> high TFLOPS/s

aten::mm with shape B
  -> poor TFLOPS/s
```

Possible causes:

- bad tile shape
- small batch dimension
- unfavorable alignment
- memory-bound shape
- layout conversion nearby
- low occupancy
- poor backend algorithm selection

For agent workloads, this is especially useful because **context length and output length** change request by request.

A model may be **fast at one sequence length and slow at another**.

---

## 8. Roofline-style compute modeling

Kernel duration answers:

```text
How long did it take?
```

It does not answer:

```text
Was that good?
```

TraceLens estimates **theoretical work** from operator arguments.

For GEMM:

```text
FLOPs = 2 * M * N * K
Bytes = (M*K + K*N + M*N) * element_size
```

Then it combines that with actual trace timing:

```text
TFLOPS/s = FLOPs / time
TB/s     = bytes / time
```

This gives a roofline-style view:

```text
high arithmetic intensity + low TFLOPS/s
  -> compute kernel may be underutilizing the GPU

low arithmetic intensity + high TB/s
  -> operation may be memory-bandwidth bound

unexpected bytes or copies nearby
  -> inspect layout, dtype, or framework-generated movement
```

Important distinction:

TraceLens estimates useful **theoretical work** from framework semantics.

Hardware profilers measure **what the GPU actually executed**.

Use both:

```text
TraceLens:
  "What work did the model intend?"

Hardware counters:
  "What did the GPU actually do?"
```

The gap between those two is often **where the optimization lives**.

---

## 9. Multi-GPU communication analysis

Distributed training and serving add another failure mode:

```text
total collective time != pure network time
```

Collective operations can include synchronization skew.

One rank may enter the collective late because:

- its compute was slower
- its batch was heavier
- it hit a local scheduling delay
- it had a CPU dispatch stall
- it waited on a prior dependency

If you only look at total collective duration, you may **blame the network for workload imbalance**.

TraceLens separates:

- payload size
- collective latency
- algorithmic bandwidth
- bus bandwidth
- synchronization skew

Useful diagnostic patterns:

```text
high communication latency, low skew
  -> likely network or collective implementation bottleneck

high skew, reasonable bandwidth
  -> likely imbalance or late-arriving ranks

high exposed communication time
  -> overlap is insufficient

high total communication but low exposed communication
  -> communication exists but is mostly hidden under compute
```

For hardware engineers, this matters because **AI scaling bottlenecks are often misdiagnosed**.

The right question is not:

```text
How much time did all-reduce take?
```

The better question is:

```text
How much pure communication time was exposed on the critical path,
and how much was rank skew?
```

---

## 10. Trace comparison: prove the delta

Most performance work is **comparative**:

```text
old kernel vs new kernel
CUDA vs ROCm
H100 vs MI300X
driver A vs driver B
BF16 vs FP8
FlashAttention backend vs FlashInfer backend
baseline vLLM vs optimized vLLM
```

TraceLens can compare reports and identify where time changed.

For simple cases, matching happens at the same operator level.

For more complex cases, TraceLens can use **morphological comparison**: it aligns trees and finds the lowest point where the call stacks diverge.

That is useful when two backends implement the same framework op differently.

Example:

```text
aten::_convolution
  -> ROCm backend subtree
  -> CUDA backend subtree
```

The framework-level intent is the same, but the backend subtree differs.

Trace comparison lets you say:

```text
This change improved convolution backward by X ms.
This change regressed aten::mm by Y ms.
This backend moved time from one subtree into another.
```

That is the **standard you should use** for optimization claims.

---

## 11. Event replay: isolate the slow operation

Finding a slow operation is **only half the job**.

You still need a **reproducer**.

Full model reproducers are often difficult to share because they include:

- proprietary model architecture
- private weights
- production input data
- distributed launch setup
- complex environment state

TraceLens **event replay** generates a minimal script from trace metadata:

- operation type
- tensor shapes
- dtypes
- strides
- relevant arguments

This produces a focused benchmark case.

Why this matters:

```text
model team finds a slow op
  -> event replay creates minimal reproducer
  -> kernel/compiler/vendor team debugs it
  -> fix is validated against original trace
```

This is how you turn model-level performance debugging into **systems engineering**.

---

## 12. Where TraceLens fits in the agent stack

Agent systems need performance evidence at multiple layers:

```text
application:
  user latency, tool latency, session throughput

agent runtime:
  planning time, tool-call count, retry count, context growth

model server:
  TTFT, ITL, output tokens/s, queueing, KV-cache memory

GPU:
  kernels, copies, collectives, idle time, roofline metrics
```

TraceLens focuses on the **lower layers**, but it should be connected to the higher layers.

For OpenClaw-style systems, a useful workflow is:

```text
run representative agent workload
  -> collect model-server trace
  -> generate TraceLens report
  -> identify bottleneck
  -> create event replay if needed
  -> change model/backend/kernel/config
  -> compare traces
  -> attach report to deployment decision
```

This pairs directly with **Lecture 43**.

If you enable FP8 KV-cache, do not only report:

```text
It feels faster.
```

Report:

```text
TTFT
ITL
output tokens/s
GPU memory
operator-level time
attention kernel time
copy time
communication exposure
accuracy
trace comparison delta
```

---

## 13. Practical workflow

Install TraceLens:

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

Generate a performance report from a PyTorch trace:

```bash
TraceLens_generate_perf_report_pytorch \
  --profile_json_path path/to/your/trace.json
```

Compare two reports:

```bash
TraceLens_compare_perf_reports_pytorch \
  baseline.xlsx candidate.xlsx \
  --names baseline candidate \
  --sheets all \
  -o comparison.xlsx
```

Generate a multi-rank collective report:

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
  --trace_dir /path/to/traces \
  --world_size 8
```

For ROCm `rocprofv3` Perfetto-style traces:

```bash
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --rccl-trace \
  --output-format pftrace \
  -d ./v3_traces \
  -- python3 your_app.py

TraceLens_generate_perf_report_pftrace_hip_activity \
  --trace_path sample.pftrace \
  --write_md
```

Use these commands as **starting points**, not a complete profiling policy.

For reliable comparisons, also pin:

- model revision
- input workload
- batch/concurrency
- sequence lengths
- dtype policy
- GPU clocks if relevant
- driver/runtime versions
- backend flags
- warmup count
- random seeds if accuracy is involved

---

## 14. Diagnostic playbook

Use TraceLens reports to **choose the next action**.

### Case: high idle time

Likely causes:

- CPU dataloader bottleneck
- Python overhead
- synchronization
- dynamic shape recompilation
- queue starvation
- model server scheduling gap

Next actions:

- inspect CPU dispatch timeline
- check batch construction
- reduce Python in the hot path
- verify warmup and compile behavior
- correlate with request-level telemetry

### Case: GEMM dominates but TFLOPS/s is low

Likely causes:

- small or awkward matrix shapes
- poor alignment
- bad layout
- backend algorithm choice
- unnecessary transposes
- low occupancy

Next actions:

- inspect shape-level rows
- compare against hardware peak and expected roofline
- generate event replay for the slow shape
- test alternative kernels or layouts

### Case: attention dominates long-context serving

Likely causes:

- KV-cache memory traffic
- head dimension issue
- backend selection
- context length distribution
- low-precision conversion overhead

Next actions:

- compare BF16 vs FP8 traces
- inspect attention kernel time
- test skip policies for sliding-window layers
- validate accuracy on real prompts
- connect to Lecture 43's TTFT/ITL benchmark plan

### Case: exposed communication is high

Likely causes:

- poor overlap
- collective algorithm issue
- rank imbalance
- network bottleneck
- tensor-parallel or FSDP layout mismatch

Next actions:

- inspect skew vs pure communication time
- compare algorithmic and bus bandwidth
- inspect per-rank timelines
- adjust sharding, bucket sizes, or overlap strategy

### Case: regression after a software update

Likely causes:

- backend kernel selection changed
- compiler changed generated code
- framework changed dispatch path
- dtype/layout behavior changed
- fusion changed

Next actions:

- run TraceLens comparison
- identify the largest positive deltas
- inspect lowest divergent subtree
- replay the slow op
- keep the trace report with the change review

---

## 15. How this connects to agent reliability

Earlier lectures focused on **agent skills, structured tools, and verification**.

TraceLens applies the **same idea to performance**:

```text
claim:
  "This optimization improves long-context serving."

required evidence:
  trace report
  before/after comparison
  workload description
  bottleneck explanation
  accuracy check
  deployment decision
```

This matters because agent systems invite **vague performance claims**:

- "The model is slow."
- "The GPU is the bottleneck."
- "Communication is killing scaling."
- "FP8 is faster."
- "The new backend regressed."

TraceLens helps turn those into **falsifiable statements**:

```text
On this workload, aten::mm for this shape regressed by X ms.
On this run, exposed communication is Y% and skew is Z us.
On this model, attention kernel time dropped but copy time increased.
On this backend, GPU idle time is dominated by CPU dispatch gaps.
```

That is the **engineering standard**.

---

## Mini-lab: Trace a model change

Choose one small but real workload:

- a PyTorch transformer block
- a vLLM benchmark
- a multi-GPU training step
- a custom CUDA/ROCm kernel path
- an OpenClaw agent workload that calls a local model server

Run two configurations:

```text
baseline
candidate
```

Examples:

```text
BF16 vs FP8
old backend vs new backend
CUDA vs ROCm
single GPU vs multi-GPU
eager vs compiled
with fusion vs without fusion
```

Collect traces, then generate TraceLens reports.

Write a short performance note:

```text
Workload:
Hardware:
Software versions:
Main bottleneck:
Biggest improvement:
Biggest regression:
Evidence:
Deployment decision:
Next experiment:
```

If you find one slow operation, generate an event replay and treat it as a kernel debugging task.

---

## Key takeaways

- Raw profiler traces are too large and flat for reliable manual analysis.
- TraceLens turns framework traces into hierarchical, queryable evidence.
- Trace2Tree connects Python modules, CPU dispatches, runtime launches, and GPU kernels.
- Start diagnosis at the GPU timeline, then drill into categories, ops, shapes, and kernels.
- Roofline-style metrics help distinguish "slow" from "inefficient."
- Multi-GPU collective time must be separated into pure communication and synchronization skew.
- Trace comparison is the right way to prove a performance change.
- Event replay turns trace metadata into minimal reproducers for kernel and backend debugging.
- For agent systems, performance claims should be backed by traces, comparisons, and workload descriptions.

---

## References

- AMD ROCm Blog, "TraceLens: Democratizing AI Performance Analysis": [https://rocm.blogs.amd.com/software-tools-optimization/tracelens/README.html](https://rocm.blogs.amd.com/software-tools-optimization/tracelens/README.html)
- TraceLens GitHub repository: [https://github.com/AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens)
- PyTorch profiler documentation: [https://docs.pytorch.org/docs/stable/profiler.html](https://docs.pytorch.org/docs/stable/profiler.html)
- Perfetto trace viewer: [https://ui.perfetto.dev](https://ui.perfetto.dev)
- Lecture 43 - FP8 KV-Cache in vLLM: [Lecture-43.md](Lecture-43.md)

---

*Next: [Lecture 45](Lecture-45.md)*
