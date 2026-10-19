# Performance: calculate a bound, then measure a workload

This checkout has **no measured Jetson model throughput, temperature or energy result**. Its executable benchmark measures CPU parsing and sampling components. Keep that scope in every report.

## 1. Reproduce the available measurement

From this project directory:

```bash
./scripts/test_plan.sh /tmp/jllm-course
./scripts/bench.sh /tmp/jllm-course
```

The benchmark uses an 832-byte fixture, 100 parses per batch, 30 sampling operations per batch and seven batches. Synthetic logits have vocabulary size 32,000; sampling uses temperature 0.7, top-k 40, top-p 0.9 and seed 7.

The retained [example JSON](../labs/benchmark-example.json) reports median batch means of **14.20659 microseconds per parse** and **2904.6041 microseconds per sample** on the shared x86 review host. These are observations, not targets for your computer. The tiny file is cache-resident, so this does not measure storage bandwidth. Batch means cannot establish individual-operation p99 latency.

**Check:** is the reciprocal of sample time a model's tokens/s? No: there is no transformer forward pass, KV update or model in this measurement.

## 2. Count actual encoded bytes

A dense M-by-K matrix-vector operation performs approximately `2*M*K` operations. Weight bytes depend on the encoding, including metadata.

The local Q4_K structure stores 256 weights in **144 bytes**: 128 bytes of nibbles plus 16 bytes of scale/minimum metadata. This is 4.5 bits per weight. The FP16 payload for the same count is 512 bytes, giving a storage ratio of `512/144 ≈ 3.56`. Neither an exact 4x reduction nor a 3.56x speedup follows automatically.

Illustrative bound: assume 1.8 GB of weights must be read from DRAM per decode step and those reads sustain 60 GB/s. Reading alone requires at least `1.8/60 = 0.03 s`, an upper bound of about 33.3 steps/s before other work. Both numbers are assumptions, not local measurements.

Batching, cache reuse, sparsity, active experts, activation/KV traffic and decoding work change the model. Quantization also changes quality. Use actual active bytes rather than assuming every architecture reads every advertised parameter each step.

## 3. Keep the compute roof consistent

Do not divide FP32/FP16 operations by a sparse INT8 TOPS figure. Match arithmetic type, sparsity convention, accumulation and instruction path.

Orin Nano 8 GB has eight SMs, 1,024 CUDA cores, 32 Tensor Cores and no DLA. Query device properties and active clocks/power mode. Use the [platform course](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md) for a matched BSP/toolchain, rather than stale constants in an example header.

```text
arithmetic intensity = operations / bytes at the measured memory level
attainable rate <= min(compute roof, bandwidth roof * arithmetic intensity)
```

This ceiling does not identify the actual bottleneck. Dependencies, launch overhead, synchronization and insufficient parallelism can leave a workload below both roofs.

## 4. Define latency boundaries

| Metric | Boundary |
|---|---|
| Queue delay | Accepted request to processing start |
| Tokenization | Text to token IDs |
| Prefill | Full prompt processing under a stated cache convention |
| Time to first token | Request arrival to first emitted token |
| Decode intervals | Time between subsequent emitted tokens |
| End-to-end latency | Request arrival to completed response |

The experimental engine's prefill timer excludes the final prompt-token forward pass. It is therefore not a complete prefill or TTFT measurement. Fix the boundary before comparing with another engine.

## 5. Profile a validated executable

The optional wrapper accepts a workload that already works:

```bash
./scripts/profile.sh /tmp/jllm-profile-run1 -- /absolute/path/to/validated-program its-arguments
```

It needs Nsight Systems and a new output prefix. No trace was captured in this review. A trace explains waits and activity; it does not establish numerical correctness.

Report model/input hashes, prompt/output lengths, cache state, batch/concurrency, software versions, clocks, power mode, temperature, warmup, repetitions and dispersion. Compare identical inputs and quality thresholds before claiming an optimization. Integrate measured power over the task to obtain joules per completed task; watts alone are not energy.

**Finish:** reproduce and label the CPU JSON, derive the Q4_K storage ratio, and explain why neither is pretrained-model throughput. Use the [testing lesson](testing.md) to define the next experiment's correctness gate.
