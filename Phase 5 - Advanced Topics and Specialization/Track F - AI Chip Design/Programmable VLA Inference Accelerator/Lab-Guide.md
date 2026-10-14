# Laboratory guide — CPU design studies before hardware commitments

These labs reproduce the proposal's arithmetic and expose assumptions that change architecture decisions. They are **analytical fixtures**, not pretrained-policy, GPU, simulator, FPGA, or silicon benchmarks. Use Python 3.10+; no packages, network, model download, or privileged settings are required.

## Files and setup

From the repository root, in one Bash session:

```bash
VLA_DESIGN_LABS='Phase 5 - Advanced Topics and Specialization/Track F - AI Chip Design/Programmable VLA Inference Accelerator/labs'
python3 "$VLA_DESIGN_LABS/design_lab.py" --help
python3 -m unittest discover -s "$VLA_DESIGN_LABS" -p 'test_*.py'
```

Website readers can download [design_lab.py](labs/design_lab.py) and [test_design_lab.py](labs/test_design_lab.py) into the same directory and run them there. The test suite checks unit conversion, partial blocks, cold fills, reserve capacity, scheduling boundaries, overhead, malformed records, and failed-attempt energy.

**Shared notation:** GB is `10^9` bytes; MiB is `2^20` bytes. GB/s is a stated sustained rate in this model. `resident` means a validated subset whose lifetime spans the modeled iterations; it is not an arbitrary cache hit-rate assumption.

## Lab A — capacity and repeated traffic

**Connects to:** [workload tracing](Lecture-02.md) and [memory architecture](Lecture-04.md).

Predict these quantities for a 3B packed-4-bit backbone read once plus a 300M 8-bit expert read ten times: distinct weight capacity, per-evaluation traffic, and the bandwidth floor at 120 GB/s.

```bash
python3 "$VLA_DESIGN_LABS/design_lab.py" memory
python3 "$VLA_DESIGN_LABS/design_lab.py" memory --bandwidth-gbps 60
python3 "$VLA_DESIGN_LABS/design_lab.py" memory --nvfp4-metadata
```

The default outputs 1.8 GB capacity, 4.5 GB traffic, **37.5 ms** weight-transfer floor, and 90 GB/s required for those reads at 20 Hz. Halving bandwidth doubles this floor. It does not predict complete inference time.

The metadata option represents the entire synthetic backbone as **one** 1D NVFP4 tensor: `3e9/2 + 3e9/16 + 4 = 1,687,500,004` bytes. Real checkpoints have many tensors, unquantized components, padding, and possibly different layouts. This switch illustrates missing metadata; it is not a checkpoint conversion or an NVFP4 compute implementation.

**Negative control:** in your worksheet, multiply both stages by ten. Explain exactly which graph assumption that changes and why it no longer describes the fixture. Set bandwidth to zero and confirm the CLI rejects the input.

**Deliverable:** a ledger separating resident capacity, repeated traffic, metadata, and omitted costs.

## Lab B — partial residency and its limits

**Connects to:** [memory residency](Lecture-04.md) and [capstone design space](Lecture-10.md).

For weight size `W`, repeats `I`, and retained subset `R`:

```text
cold traffic = I * (W - R) + R
warm traffic = I * (W - R)
cold savings versus no residency = (I - 1) * R
```

Cold includes the first fill. Warm excludes a valid prior fill; a checkpoint/layout change or invalidated contents makes that assumption fail. For `I=1`, cold residency saves no weight reads.

```bash
python3 "$VLA_DESIGN_LABS/design_lab.py" memory --sram-mib 64 --reserve-mib 8
python3 "$VLA_DESIGN_LABS/design_lab.py" memory --sram-mib 64 --reserve-mib 8 --warm
python3 "$VLA_DESIGN_LABS/design_lab.py" sweep
```

Only 56 MiB remains for expert weights. Cold traffic becomes `4.5e9 - 9*56*2^20 = 3,971,517,696` bytes, a **33.096 ms** weight-transfer floor at 120 GB/s. The 300 MB expert still does not fit. The 8 MiB reserve is a toy assumption; derive the real activation, DMA, queue, accumulator, and conditioning reserve from live allocations.

The sweep compares 16/32/64 MiB total SRAM and 60/120/180 GB/s sustained bandwidth. It produces no area, power, or feasibility score. More bandwidth or SRAM is not free, and retained tensors may compete for banks or evict useful activations.

**Negative controls:** request a reserve larger than SRAM; invalidate resident weights between evaluations; compare a one-iteration workload. Explain which claimed savings survive.

**Deliverable:** a cold/warm sensitivity table and the trace evidence needed to justify retention.

## Lab C — shape utilization is not achieved performance

**Connects to:** [tensor/vector tiles](Lecture-03.md).

```bash
python3 "$VLA_DESIGN_LABS/design_lab.py" tiles
```

The model counts useful scalar products divided by all scalar products in fully padded tiles. For an `8×256` output with reduction size 256 and `64×64×64` tiles, the fraction is **12.5%**. A smaller row tile can improve this fraction but may change data reuse, control overhead, and implementation cost.

**Negative control:** use odd dimensions in all three axes, not just the output rows. The included test checks `17×19×23` against `16×16×16` tiles. A result of 100% based only on one dimension is a broken model.

**Deliverable:** two candidate tile organizations and the real operator shapes that could favor each. Do not call this fraction measured GPU occupancy or accelerator utilization.

## Lab D — legal scheduling boundaries

**Connects to:** [fresh-action scheduling](Lecture-07.md).

```bash
python3 "$VLA_DESIGN_LABS/design_lab.py" schedule
python3 "$VLA_DESIGN_LABS/design_lab.py" schedule --dispatch-ms 2
```

Whole-job FIFO completes action at 21 ms, after its 8 ms absolute deadline. With 3 ms legal tile boundaries and zero dispatch overhead, the EDF choice completes action at 6 ms. At 2 ms overhead per segment it completes at 10 ms and misses again.

This is one abstract server with ready independent jobs. There is no modeled GPU preemption, Linux scheduler, DMA overlap, dependency graph, bank conflict, or hard-real-time proof. The quantum represents a legal boundary already provided by an implementation, not the ability to interrupt any kernel at will.

Timing calculations use the decimal spelling of input values. This prevents binary floating-point rounding from delaying a release or reporting a deadline miss at exact equality. Limits are inclusive; strictly exceeding one counts as a violation. Real timestamp resolution and clock uncertainty still need an explicit measurement budget.

**Negative controls:** add a long noninterruptible transfer, overload the server, or delay action release. State which changes can be represented by this toy model and which require a richer simulator.

**Deliverable:** timelines with blocking and dispatch costs, plus a proposed hardware/software boundary to measure.

## Lab E — all-attempt energy and observation age

**Connects to:** [evaluation](Lecture-08.md).

```bash
python3 "$VLA_DESIGN_LABS/design_lab.py" evaluate
```

The three synthetic attempts consume 30 J in total; two succeed, so energy per success is 15 J. One action misses a 50 ms capture-to-usable deadline and one exceeds 70 ms observation age at consumption. The same attempt can trigger both counters.

This fixture has one instrumented action per attempt. A real episode needs a separate per-action table and per-episode energy/outcome table. All fixture timestamps share a clock. The nearest-rank p99 of three samples is only an arithmetic check, not a useful production-tail estimate.

**Negative controls:** all attempts fail, an episode ID is duplicated, energy is nonfinite, or timestamps run backward. The API either rejects invalid data or reports the no-success ratio as undefined (`null`), rather than manufacturing an efficiency result.

**Deliverable:** energy boundary, failed-attempt accounting, full-path latency and consumption age, and a paired quality-evaluation plan.

## Replace fixtures with measurements

Create a versioned trace with model/processor revisions, sensor settings, action iterations, shapes, layouts, actual operator dispatch, timestamps, allocation lifetimes, and power sampling. Compare the model with a measured subset before expanding it. Retain analytical estimates and measured columns side by side.

Use [Isaac engineering labs](../../Track%20D%20-%20Robotics/Isaac%20Sim%20and%20Isaac%20Lab/Lab-Guide.md) for closed-loop integration and [VLA evaluation labs](../../Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Lab-Guide.md) for paired outcomes. Continue to [prototype verification](Lecture-09.md) only after the relevant numerical and timing contracts pass.

**Completion:** reproduce all five exercises, explain an invalid assumption in each, and pass their artifacts to [the capstone](Lecture-10.md). Hardware access changes the evidence level; it does not change the requirement to label estimates and unexecuted work honestly.
