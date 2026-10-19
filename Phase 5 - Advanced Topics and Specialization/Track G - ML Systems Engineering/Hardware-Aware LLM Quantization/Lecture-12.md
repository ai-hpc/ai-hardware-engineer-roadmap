# Module 12 — Research Methodology: Reproduce Before Comparing

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-11.md) | **Next:** [Next lesson](Lecture-13.md)

A benchmark needs an identity, a workload and a correctness gate. Quantization
changes numerical behavior and runtime costs together, so record both.

## 1. Record and control the machine

Log device name, driver, runtime/compiler, clocks, temperature, power policy,
concurrent work, host configuration and memory. Use steady-state warmup determined
from observed timings rather than assuming exactly twenty iterations is enough.

Read-only discovery examples, **not executed GPU evidence for this course**:

```bash
nvidia-smi --help-query-gpu
nvidia-smi -q -d SUPPORTED_CLOCKS
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv
```

Clock locking is optional and device/driver/privilege dependent. Consumer cards
may reject memory-clock controls. Query supported values and use only documented
controls on a machine you administer; record failures, actual clocks and reset
instructions. Do not copy a 14001 MHz memory setting across boards. Unlocked-clock
measurements can be useful with randomized/interleaved ordering and telemetry.
[NVIDIA SMI reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## 2. Freeze the variables that define the comparison

| Variable | Why it matters |
|---|---|
| Model and quantized artifact revisions | names and moving branches change |
| Runtime, kernel dispatch and build flags | identical format names can execute differently |
| Prompt IDs, tokenization and template | changes work and semantics |
| Input/output lengths and EOS rules | short output can look artificially fast |
| Batch/concurrency and scheduling | changes weight reuse and queues |
| KV/state policy and context | changes capacity, traffic and correctness |
| Sampling transform and seeds | changes distribution and variability |
| Target, drafter, K and counter definition | needed to interpret acceptance |

If changing target quantization, compare target behavior to the reference directly.
If changing drafter quantization, freeze the target and use exact correction.
Even a fixed drafter does not make acceptance a measurement of target-reference KL.

## 3. Pair runs and quantify uncertainty

Run A/B in randomized interleaved blocks on the same prompt IDs. Retain each raw
request's latency, lengths, errors and telemetry. Bootstrap requests or independent
sessions, not correlated tokens within one request. Report the effect size and
interval with the sample size; predefine a practically meaningful improvement.

A single seed or a mean over five timing loops cannot establish a tiny improvement
under variable clocks. Conversely, an unsupported fixed requirement such as “at
least 8,000 draft tokens” does not guarantee statistical power. Pilot variance,
the effect size of interest and independence determine sampling effort.

Measure time to first token, inter-token latency and full-request latency separately.
State whether initialization, loading, quantization and host/device transfers are
included. For asynchronous GPU work use a correct synchronization/event protocol;
wall-clock timing of an enqueue alone does not measure execution.

## 4. Check correctness before trusting a speedup

Test logits/task outputs, request lengths, cache reset and rollback. A mismatch can
come from an implementation bug, unsupported format, or real quantization error.
A surprisingly high speed is a reason to inspect work performed, not automatic
proof of a bug: the traffic model may overcount reads or omit cache reuse.

The [ledger](Lecture-04.md) estimates required traffic. Call `bytes_estimate/time`
an **estimated effective rate**, not a measured hardware bandwidth counter. Do
not require a guessed 92%-efficiency roofline to bound every valid measurement.

## 5. A report template without invented measurements

```yaml
status: planned  # change only after execution
model_revision: null
quant_artifact_sha256: null
runtime_revision: null
device: null
workload_hash: null
sampling: null
context_and_output_lengths: null
batch_and_concurrency: null
target_drafter_and_K: null
kv_and_recurrent_state_policy: null
calibration_ids_hash: null
quality_gate_and_mask: null
raw_results_path: null
latency_interval: null
quality_results: null
estimated_traffic_bytes: null
measured_dram_bytes: null
limitations: []
```

Unknown values stay unknown. A fabricated example row with realistic-looking
clocks, KL and throughput must never be presented as a completed run.

## Practice and completion

Use the CPU lab's frozen held-out and sensor-shift results to compare quantizers.
Explain why the lab measures numerical/calibration behavior but cannot claim
native FP4 speed. On a suitable GPU, choose one device experiment, retain its raw
measurements and a paired analysis, and check that it computes the same requested
work. An unsuccessful optimization with a sound explanation is a valid result.
