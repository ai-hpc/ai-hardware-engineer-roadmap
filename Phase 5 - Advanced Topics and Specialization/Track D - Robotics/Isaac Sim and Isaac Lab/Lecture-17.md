# Lecture 17 — Training experiments and held-out evaluation

**Prerequisites:** Lecture 16 and PPO fundamentals. **Time:** 3–4 hours of experiment work, plus a bounded training budget. **Outcome:** distinguish learning progress from evaluation evidence.

## Choose baselines before tuning

Evaluate zero-action, random-action, and a simple scripted controller where sensible. If a scripted baseline solves the task, it provides a valuable reference for reward design and latency. An RL result should explain its advantage or its intended learning objective.

Freeze training seeds, development seeds, and a final held-out seed/scenario set. Keep randomization ranges and task difficulty with each split. Seeds alone do not specify object geometry, initial conditions, friction, or sensor perturbations.

## Sample budget and wall time answer different questions

With N environments and H collected policy steps per update:

```text
transitions_per_update = N * H
policy_dt = physics_dt * decimation
```

Changing N while holding H fixed changes batch size. It can affect optimization as well as collection speed. Report learning curves versus both environment transitions and wall-clock time; avoid claiming sample efficiency from only a time plot.

## Worked uncertainty example

If 80 of 100 independent evaluation episodes succeed, the observed rate is 0.8. The supplied Wilson calculation gives a roughly 0.71–0.87 95% interval. It expresses binomial sampling uncertainty under an independence assumption; it does not cover distribution shift or variability between training runs.

Evaluate multiple trained seeds separately, then summarize between-seed variation. Hundreds of episodes from one policy are not hundreds of independent training runs. Repeated conditions or correlated trajectories further weaken naive intervals.

## Lab

1. Fix success, failure, timeout, and early-stop definitions.
2. Set a maximum training duration or transition budget.
3. Train at least three seeds if hardware permits; keep incomplete runs visible.
4. Evaluate each policy on the same held-out scenarios.
5. Log success counts, episode duration, failure category, and constraint violations.
6. Change one reward or randomization choice and rerun the paired evaluation.
7. Plot all seed curves; distinguish development selection from final reporting.

Use the [existing training lecture](Lecture-10.md) for task creation and the [reinforcement-learning API](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/api/lab_rl/isaaclab_rl.html) for the pinned command surface.

**Reduced path:** run the confidence-interval fixture and design the seed/split table. Do not label a calculation as a trained policy result.

## Self-check

- Why keep failed runs? Dropping them overstates reliability.
- Why compare equal transition budgets and equal wall time separately? They measure sample and system efficiency.
- Why can 100/100 successes still leave uncertainty? Finite samples do not establish a true success probability of one.

**Exit artifact:** evaluation specification, baseline results, learning curves, outcome counts with uncertainty, and an explanation of one failed episode.

[Previous](Lecture-16.md) · [Lab sequence](Lab-Guide.md) · [Next: performance](Lecture-18.md)
