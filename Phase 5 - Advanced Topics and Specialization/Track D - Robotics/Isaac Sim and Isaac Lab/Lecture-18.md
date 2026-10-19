# Lecture 18 — Physics, sensors, and simulation performance

**Prerequisites:** Lectures 07, 12, and 17. **Time:** 3 hours. **Outcome:** optimize a simulator without changing the task unnoticed.

## Name each rate

For N environments, K policy steps, decimation d, physics timestep h, and wall time T:

```text
transitions/s = N * K / T
vector physics steps/s = K * d / T
aggregate environment physics steps/s = N * K * d / T
per-environment real-time factor = K * d * h / T
```

A thousand environments producing many transitions per second does not mean a single robot runs a thousand times faster than real time. Exclude setup/warmup consistently and state whether timing includes reset, policy inference, rendering, and learner updates.

## Memory before pixels

A simple stored RGB observation buffer costs:

```text
N * rollout_steps * height * width * channels * bytes_per_element
```

For an invented example, 64 environments × 32 steps × 96 × 96 × 3 channels × 4 bytes is 216 MiB before frame stacking, intermediate features, renderer buffers, or learning state. A viewport setting cannot reduce a rollout buffer allocated by the RL library.

Start state-only, then add one sensor. Headless mode removes the GUI but does not necessarily remove offscreen rendering. Verify the selected renderer and sensor update rate.

## Lab

1. Establish a correct state-only baseline at 16 environments.
2. Sweep environment count through a small feasible sequence.
3. Profile physics, observation construction, reward/reset, policy, rendering, and transfers.
4. Add one camera, then vary resolution and update frequency separately.
5. Measure whole-process/device memory as well as PyTorch allocations; the simulator may allocate outside PyTorch.
6. Re-run task correctness and evaluation after each performance change.

Use the upstream [benchmark command](https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/developer-tools/benchmarking/run_benchmarks.rst) pattern from the lab guide and save its resolved configuration. Its output is meaningful only with the actual task/backend settings.

## Failure clinic

| Change | Possible apparent speedup | Hidden change to check |
|---|---|---|
| Larger physics dt | Fewer physics steps | Contact/control accuracy and episode dynamics |
| Lower sensor rate | Less rendering | Observation age and partial observability |
| Simpler collision mesh | Faster contact | Reachability and collision behavior |
| More environments | More transitions/s | PPO batch size and time to target success |
| CPU scalar logging every step | Easier debugging | Device synchronization stalls |

**Independent task:** find a faster configuration with unchanged task definition, then a faster configuration that changes it. Label their evidence differently.

## Self-check

Why measure beyond the framework allocator? RTX/physics/runtime allocations may be outside it. Why report time-to-success? Maximum transition throughput can coexist with slower learning.

**Exit artifact:** throughput and peak-memory curves, a stage profile, and held-out outcomes before/after the chosen change. No speedup is accepted solely because simulated time advances with fewer physics steps.

[Previous](Lecture-17.md) · [Lab sequence](Lab-Guide.md) · [Next: ROS timing](Lecture-19.md)
