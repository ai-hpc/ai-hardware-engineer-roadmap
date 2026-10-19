# VLA laboratory guide

[Course overview](README.md) · [Training capstone](Lecture-31.md) · [Professional capstone](Lecture-32.md)

This course provides a **runnable CPU foundation** and separately scoped GPU/simulator extensions. The CPU code uses Python's standard library, downloads no model weights, and never sends robot commands. Advanced lecture exercises require learner implementation; they are not presented as already implemented experiments.

## What is implemented and tested here

| File | Implemented behavior | What it does not establish |
|---|---|---|
| [vla_lab.py](labs/vla_lab.py) | Generates RGB/language/state/action samples; trains nine attention logits with analytical SGD; evaluates correct, wrong, and absent language conditioning | Learned visual representations, pretrained-model adaptation, physical dynamics, or real-robot control |
| [contracts.py](labs/contracts.py) | Action denormalization checks, episode-safe chunks/masks, split-group overlap check, delay/expiry arithmetic, paired task bootstrap | A complete dataset validator, ROS scheduler, controller, or comprehensive statistics library |
| [episode_pipeline.py](labs/episode_pipeline.py) and [supplied episode file](labs/episode-fixture.json) | Causal image/state selection, frame/unit/rate conversion, split/normalizer provenance, masked loss, offline dispatch/reset replay | Actual RLDS/LeRobot loading, learned model behavior, physical controller, or RTC flow inpainting |
| [test_labs.py](labs/test_labs.py) | Finite-difference gradient, grounding intervention, numerical/input failures, reset boundaries, timing boundaries, and macro weighting tests | GPU performance, simulator correctness, or physical safety |

**Local validation:** the original exercises were executed on October 13, 2026 and the expanded 31-test suite was re-executed October 18. No Isaac installation, GPU model training, model API call, or physical robot experiment was run as part of this validation. Python 3.10+ is the teaching baseline; the scripts require no pip installation. The existing CI uses Python 3.12.

The separate [action-generation lab](Action-Generation-Lab.md) adds seven standard-library checks plus six optional PyTorch checks. The discovery command below finds them too: without PyTorch, the six framework tests explicitly skip. The original scripts still require no pip installation.

## Run from the repository root

These commands are **tested here**. The quoted directory handles the repository's spaces.

```bash
VLA_COURSE='Phase 5 - Advanced Topics and Specialization/Track D - Robotics/VLA Foundations to Frontier 2026'
python3 -m unittest discover -s "$VLA_COURSE/labs" -p 'test_*.py'
python3 "$VLA_COURSE/labs/vla_lab.py" --epochs 30 --seed 17 > /tmp/vla-grounding.json
cat /tmp/vla-grounding.json
```

The result is JSON containing sample counts, procedural seeds, optimization settings, training losses, three test errors, attention rows, and learned logits. Preserve the command, Python version, source commit or file hashes, and output when submitting an experiment. `/tmp` is convenient for a smoke test; copy your report into a persistent project directory when you want to retain it.

The recorded default run produced these rounded values:

| Metric | Locally measured synthetic result |
|---|---:|
| First-epoch online training loss | 0.085314 |
| Last-epoch online training loss | 0.000392 |
| Held-out MSE per coordinate | 0.000385 |
| Uniform-attention MSE | 0.284165 |
| Wrong-instruction MSE | 0.780895 |

First/last epoch losses average samples **while SGD updates the model**, not a separate evaluation of a fixed final checkpoint. The held-out metric evaluates the final parameters. Exact last digits can vary with floating-point implementation; tests use behavioral tolerances instead of requiring a byte-identical report.

## Lab A — See exactly what learns

**Preparation:** Lectures 07 and 09; 15 min setup, 45–90 min analysis. Any CPU laptop is sufficient.

Open `vla_lab.py` and trace one sample from generated image to action. The visual encoder computes channel centroids, the instruction selects a row of three learned logits, softmax forms attention over the centroids, and the current state is subtracted. All visual tokens are fixed features. The only learned association is instruction-to-color attention.

Predict the three attention rows before running training. After training, inspect which column dominates each row. Change the test instruction while keeping image and state fixed. Explain why that intervention is stronger evidence of language use than merely evaluating on a new random image.

The finite-difference test perturbs one parameter by \(\pm10^{-6}\) and compares its numerical loss derivative with the analytical gradient. Break the gradient deliberately in a local experiment and confirm that the test fails. Restore it before proceeding. An implementation that can memorize outputs but has an incorrect derivative is not a valid training reference.

**Acceptance:** held-out error is substantially lower than uniform attention and wrong-language error; the gradient test passes; the report identifies the fixed encoder and does not claim arbitrary language understanding. The test threshold of 0.02 belongs to this synthetic fixture, not to any robot deployment.

**Transfer:** add two objects of the same color and describe why one centroid cannot represent object identity. Propose a representation that preserves individual objects before increasing model size.

## Lab B — Action and episode contracts

**Preparation:** Lectures 06 and 14; 60–90 min. Run from the course directory to make the helper imports local:

```bash
cd "$VLA_COURSE"
python3 - <<'PYLAB'
from labs.contracts import assert_disjoint_groups, chunk_with_mask, denormalize
records = [
    {"episode": "sessionA/ep1", "time": 0.00, "action": [0.1, 0.2]},
    {"episode": "sessionA/ep1", "time": 0.02, "action": [0.3, 0.4]},
    {"episode": "sessionB/ep1", "time": 0.00, "action": [9.0, 9.0]},
]
print(chunk_with_mask(records, start=1, horizon=3))
print(denormalize([0.5, -1.0], center=[0.0, 1.0], scale=[0.02, 0.1]))
assert_disjoint_groups(["sessionA"], ["sessionB"])
PYLAB
```

Expected chunk is `[[0.3, 0.4], [0.3, 0.4], [0.3, 0.4]]` with mask `[1, 0, 0]`; the next episode must not leak into the target. Denormalized output is `[0.01, 0.9]` in the units assigned by your schema. Padding repeats the last action **only as a storage convention**; padded entries must not contribute to training loss or become executable targets.

Add a duplicate timestamp, a noncontiguous reused episode ID, an infinite action, a zero scale, and an overlapping session group. Explain the intended error for each. Split checks work only on the group IDs you supply; they cannot discover a duplicated video with a different filename. Grouping by episode may still leak a collection session, scene, or demonstrator.

**Acceptance:** a boundary-crossing target is rejected or masked, invalid numbers are rejected, and the report specifies action semantics and split grouping. This helper does not resample timestamps; real datasets need causal alignment before chunking.

## Lab C — Delayed action scheduling

**Preparation:** Lectures 21–22; 60–90 min. From the course directory:

```bash
python3 - <<'PYLAB'
from labs.contracts import usable_chunk_indices
for delay in [0.0, 0.02, 0.07, 0.14, 0.18]:
    print(delay, usable_chunk_indices(0, delay, delay, 0.02, 8, 0.2))
print("tight age bound:", usable_chunk_indices(0, 0.07, 0.07, 0.02, 8, 0.1))
PYLAB
```

The 70 ms case retains indices `[4, 5, 6, 7]` with a 200 ms age bound. Tightening it to 100 ms retains only `[4, 5]`, since later intended starts would use older observations. At exactly a scheduled start the command remains eligible. Revalidate age at actual dispatch if the scheduler wakes late.

All input timestamps must share one monotonically increasing clock and reset epoch. The helper checks numeric ordering, but cannot infer whether clocks are synchronized or epochs match. It omits commands scheduled in the past; it does not decide whether to hold, interpolate, or stop between future starts. Your controller contract must specify that behavior.

**Acceptance:** draw the intended action times and arrival time; correctly explain the retained indices and empty-chunk case. The supplied [extended lab](Dataset-and-Execution-Lab.md) implements an offline dispatcher with out-of-order replies and reset epochs. It is not a ROS node.

## Lab D — Paired evaluation and uncertainty

**Preparation:** Lecture 19; 60–120 min. From the course directory:

```bash
python3 - <<'PYLAB'
from labs.contracts import paired_task_bootstrap
rows = [
    {"task": "easy", "seed": i, "reference": 0, "candidate": 1}
    for i in range(9)
]
rows += [{"task": "contact", "seed": 0, "reference": 1, "candidate": 0}]
print(paired_task_bootstrap(rows, samples=2000, seed=17))
PYLAB
```

Macro delta is zero: the task means are +1 and -1. The episode-weighted delta is +0.8. The large task-bootstrap interval reflects the two very different observed task means, not uncertainty within each task. This intentionally extreme fixture teaches weighting; it is not a realistic benchmark.

The helper resamples paired **task means**, treating observed within-task rates as fixed. It does not estimate uncertainty from new episodes within each task. Implement hierarchical resampling as an extension and compare repeated synthetic datasets whose generating probabilities you know. More bootstrap draws improve numerical stability, not the amount of empirical information.

**Acceptance:** justify your estimand, preserve pairs, reject duplicates, report per-task denominators, and state the interval's scope. Do not claim noninferiority from a point estimate alone.

## Lab E — A complete recorded-data and dispatch path

Follow [From recorded episodes to timed actions](Dataset-and-Execution-Lab.md). The supplied JSON records and standard-library implementation turn the normalization and scheduling helpers into one traceable pipeline. Work through capture-versus-arrival timing, rate conversion, training-only statistics, terminal masks, result ordering, and reset isolation before attempting a real model adapter. This is an implemented offline lab; extending it to a physical controller remains separate work.

## GPU route — from public artifact to closed-loop evaluation

Use the [Isaac lab sequence](../Isaac%20Sim%20and%20Isaac%20Lab/Lab-Guide.md) for simulator setup and [Lecture 20](Lecture-20.md) for the adapter. Consult the [frontier register](Frontier-Register.md) for artifact availability. Prefer an upstream-supported task and fine-tuning recipe before creating a new embodiment mapping.

1. Pin software, model, data, assets, normalizer, and hardware configuration.
2. Validate scripted task behavior, demonstration replay, sensor transforms, and resets.
3. Run an upstream-documented model smoke test under its own environment.
4. Implement the explicit adapter; establish offline physical-action correctness.
5. Adapt using session-separated data and a held-out development set.
6. Evaluate paired closed-loop trials; then profile concurrent simulation and inference.
7. Run failure injections and present a report with uncertainty and limitations.

These are **learner milestones**, not locally executed results. RAM/VRAM requirements vary with model, renderer, cameras, batch size, and adaptation method. Use the selected simulator's documented hardware requirements and measure peak usage; reducing environment count does not guarantee unsupported hardware becomes supported. Set wall-time and GPU-hour limits before launching training.

## Required experiment record

For every lab, record the hypothesis, source revision, exact command, input generator/data revision, seeds, units, output, negative case, interpretation, and next experiment. Distinguish measured, synthetic, estimated, upstream-reported, and proposed evidence. Follow the [shared learning guide](../../../Learning-Guide.md) and retain at least one failed hypothesis with its diagnosis.
