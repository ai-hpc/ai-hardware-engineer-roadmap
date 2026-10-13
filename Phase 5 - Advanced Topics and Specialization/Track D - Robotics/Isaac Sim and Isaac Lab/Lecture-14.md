# Lecture 14 — Reproducible simulation setup and smoke tests

**Prerequisites:** Lectures 01 and 09. **Time:** 2 hours after installation. **Outcome:** reproduce a task before changing its robot, reward, or backend.

## Why a manifest precedes training

Simulation behavior depends on more than a Python package version. A changed asset, solver preset, driver, timestep, or observation wrapper can alter results while the policy code stays identical.

Create a manifest with the Isaac Lab commit, lockfile hash, Isaac Sim version if loaded, GPU/driver, task ID, asset paths/revisions, backend, renderer, seed, physics dt, decimation, environment count, and observation/action schema. Save resolved configuration, not only CLI overrides.

## Versioned starting point

Follow the [versioned installation documentation](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html) for the course release. Resolve any tag/branch to a commit and keep it with the lockfile. The [Isaac Sim Compatibility Checker](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_workstation.html) checks a different path from a kit-less Isaac Lab task; use it for full simulator work.

A renderer-capable GPU and sufficient VRAM are requirements for the relevant sensor labs. A successful CUDA tensor operation does not test Vulkan/RTX or asset availability.

## Three smoke tests

| Test | What it exercises | What it does not establish |
|---|---|---|
| Startup / empty scene | Imports, backend initialization, renderer if selected | Correct robot dynamics |
| Zero-action rollout | Gravity, constraints, passive dynamics, resets | Learnability |
| Random-action rollout | Action shapes/ranges, finite observations, done/reset behavior | A useful control policy |

Use the commands in the [lab guide](Lab-Guide.md#simulator-execution-route). Stop smoke tests after a bounded number of steps or a short recorded interval; do not leave indefinite agent playback running as a benchmark.

## Worked failure diagnosis

Suppose the task works with state observations but fails after enabling an RTX camera. Possible causes include renderer support, render-product allocation, driver mismatch, and asset loading. Changing PPO learning rate cannot fix any of these.

Change one layer at a time: empty renderer → one static camera → one robot → one task → batched environments → policy. Save the first failing configuration and exact error. This gives a reproducible boundary instead of a list of guesses.

## Lab

1. Record a clean baseline manifest and run the three smoke tests.
2. Save cold-start duration separately from warmed simulation step timing.
3. Copy the configuration, change one parameter, and show the diff.
4. Repeat the baseline. Check finite state, matching shapes, and comparable behavior.
5. Introduce a recoverable wrong task ID or missing local asset path and record the diagnostic.
6. Restore the baseline and demonstrate recovery.

**Expected artifact:** manifest, resolved configuration, three logs, one failure/recovery pair, and an explicit hardware/feature matrix.

**Reduced path:** use the supplied CPU exercises and write the planned manifests. Mark simulator launch as not run.

## Self-check

- Why is a seed insufficient for reproducibility? Engine, asset, numerical, concurrency, and configuration differences remain.
- Why separate startup from step timing? Compilation, shader caches, and asset downloads can dominate only the first run.
- Is a moving release branch a pin? Resolve it to a commit for the experiment.

[Lab sequence](Lab-Guide.md) · [Next: frames and assets](Lecture-15.md)
