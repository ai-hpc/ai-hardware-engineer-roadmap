# Isaac engineering lab sequence

Use this sequence after the introductory lectures. It adds deliberate debugging, task tests, and deployment evidence to the existing simulator and learning material.

**Prerequisites:** Python; vectors, coordinate transforms, and basic feedback control; Lectures 01–04 before scene work, 09–10 before training. Read the [shared learning guide](../../../Learning-Guide.md) for study rhythm and grading.

**Effort:** approximately 20–30 hours for the exercises below, excluding simulator installation and long training jobs. Begin with 16 environments, short runs, and state observations; scale only after correctness.

## Version and hardware contract

Reference documentation checked **2026-10-13**: [Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html) and [Isaac Lab v3.0.0-EA installation](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html). Use a resolved Isaac Lab commit and its lockfile, simulator version/image digest, driver, task/asset revision, physics backend, and renderer. A release branch can change after this course was written.

Full Isaac Sim workflows need a compatible RTX system. An 8 GB card is an experimental reduced path, below the documented 16 GB minimum, not a supported baseline for all labs. Kit-less training is a separate path with different features; passing it does not validate Isaac Sim rendering or ROS integration.

The existing Lecture 01 contains setup notes. Preserve your installation record and reconcile any version-specific command against the pinned source's help. Do not combine a 2.x tutorial's API with a 3.x task just because their class names look similar.

## Lab ladder

| Lecture | Main exercise | Evidence | Reduced path |
|---|---|---|---|
| [14 — Reproducible simulation setup](Lecture-14.md) | Pin, launch, zero/random-action smoke tests | Manifest and startup/steady-state logs | Inspect metadata and design smoke cases |
| [15 — Frames, units, and assets](Lecture-15.md) | Robot/scene contract and transform probes | Frame diagram and round-trip tests | Pure-Python quaternion fixture |
| [16 — Task and reset correctness](Lecture-16.md) | Partial resets, rewards, done flags, action mapping | Task test matrix | Reset-isolation fixture |
| [17 — Learning and evaluation design](Lecture-17.md) | Train baseline, held-out evaluation, ablation | Learning curves and outcome counts | Wilson-interval calculation |
| [18 — Sensor and simulation performance](Lecture-18.md) | Profile physics/render/policy and environment count | Throughput/memory curves | Workload and buffer arithmetic |
| [19 — ROS 2 timing and policy export](Lecture-19.md) | Observe-to-act timing and stale-command handling | Timing trace and policy contract | Offline command-age fixture |
| [20 — System identification and transfer](Lecture-20.md) | Perturb dynamics and delays, identify failure envelope | Paired evaluation and bounded assumptions | Offline response-data analysis plan |
| [21 — Language-guided robotics capstone](Lecture-21.md) | Asynchronous language planner plus bounded robot skills | Trace, rejection tests, closed-loop results | Recorded-event replay |

## CPU exercises supplied here

From this course directory:

```bash
python3 labs/robotics_contracts.py
python3 -m unittest discover -s labs -p 'test_*.py'
```

The [implementation](labs/robotics_contracts.py) and [tests](labs/test_contracts.py) need Python 3.10+ and no simulator. They cover coordinate convention, partial-reset isolation, rate arithmetic, confidence intervals, and command freshness. Their outputs are **synthetic checks**, not Isaac performance or robot validation.

Expected rate fixture: 16 environments, 100 policy steps, decimation 2, physics dt 1/120 s, 2 seconds wall time gives 800 transitions/s, 100 vector physics steps/s, and per-environment real-time factor 0.833333…. Aggregate transitions/s is not the clock speed of one robot.

## Simulator execution route

Inside the pinned Isaac Lab checkout, consult the [versioned task generator](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/developer-tools/template_generator.html) or existing course Lecture 10. Select the actual task ID printed by that checkout. The following **upstream-documented command pattern** uses a shell variable, not a literal placeholder task:

```bash
# Set TASK_ID to a task registered in your pinned project.
: "${TASK_ID:?Set the registered task ID}"
uv run isaaclab zero_agent --task "$TASK_ID" --num_envs 16
uv run isaaclab random_agent --task "$TASK_ID" --num_envs 16
uv run isaaclab benchmark runtime --task "$TASK_ID" --num_envs 16 --num_steps 1000
```

Record the selected backend and renderer from resolved configuration; defaults may select a kit-less backend. For the Isaac Sim path, use the pinned installation's documented extra and physics preset. First prove scene/reset correctness, then short training, then full runs. These simulator commands were sourced from upstream documentation and were not executed in this repository's CPU-only validation.

## Grading

A simulation milestone needs a launch log, task contract, negative test, raw measurements, and a reproducible configuration. A policy milestone additionally needs held-out episodes and a baseline. A transfer milestone requires evidence from the target environment; a simulation-only result remains simulation-only.

[Course overview](README.md) · [Start Lecture 14](Lecture-14.md)
