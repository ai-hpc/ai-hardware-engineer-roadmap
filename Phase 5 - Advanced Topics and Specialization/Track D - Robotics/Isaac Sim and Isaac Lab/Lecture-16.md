# Lecture 16 — Task correctness, rewards, and partial resets

**Prerequisites:** Lectures 09–10 and 15. **Time:** 3 hours. **Outcome:** show that a vectorized environment implements the intended task before training.

## Treat the task as a program with invariants

Training cannot repair an incorrect action mapping, reward sign, or reset. Write the MDP contract: observations, actions, transition timestep, reward terms, success condition, termination, timeout, and randomization.

Distinguish task termination from a time-limit truncation. The learning wrapper's bootstrapping behavior must match that distinction. Consult the [manager-based environment tutorial](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/tutorials/03_envs/create_manager_rl_env.html) and the actual wrapper in your pinned release.

## Partial reset is the central vectorization test

For N environments, reset a strict subset such as {1, 3}. Check that only those environments' state, episode counters, command histories, recurrent state, and delayed-action buffers change. Setting every environment's last action to zero is a subtle cross-environment leak.

At reset boundaries, identify whether returned observations belong to the final state or the next episode. Do not pair a final reward with an initial observation and silently treat it as one uninterrupted trajectory. In the reviewed EA `DirectRLEnv`, `compute_final_obs=True` captures `extras["final_obs"]` before resetting completed rows. Consume it only under the current done mask: a retained extras entry is not a fresh terminal state for every row. Confirm how the chosen RL wrapper uses truncation and final observations.

The corrected [reach task](Lecture-10.md#43-the-environment-method-by-method) exposes previous action because its action-change reward depends on it, clamps targets to joint limits, clears only reset rows and requests a kinematics refresh. Its [actual CPU tensor check](labs/reach_tensor_cpu.py) tests those methods with a fake scene; the first real post-reset body/sensor pose remains a required simulator check.

## Work a reward example

Suppose a task rewards progress toward a goal and penalizes joint velocity:

```text
r = progress_reward - velocity_weight * sum(joint_velocity**2)
```

Specify whether terms are instantaneous rates integrated over policy dt or per-transition values. Changing decimation without checking this convention changes the learning problem. Success must be measured directly from task state; higher reward alone need not imply success.

Construct states where the robot is at goal, far away, violating a limit, and stationary. Predict reward and done flags before calling the implementation.

## Lab

1. Start at 4–16 environments with zero or scripted actions.
2. Check finite observations, action bounds, dimensions, and named joint mapping.
3. Unit-test reward components on known states.
4. Reset two environments and verify unchanged environments remain unchanged.
5. Force success, failure, and timeout independently.
6. Test observation history and delay buffers immediately after reset.
7. Repeat with a new seed and a second environment count.

Run the CPU reset-isolation fixture first, then implement the same invariant against real task tensors. The fixture does not test Isaac APIs.

**Independent extension:** inject an off-by-one done flag or fail to clear action history. Show the resulting test failure, fix it, and rerun the task smoke test.

## Self-check

- Why can reward improve while the task is broken? The policy can exploit reward or reset bugs.
- Why test an untouched environment? A full-reset test cannot detect accidental broadcast resets.
- Is a timeout automatically a physical failure? It depends on the task definition; preserve that distinction in evaluation.

**Exit artifact:** a task contract and test table with pass/fail evidence for every invariant above. Start expensive training only after these checks pass.

[Previous](Lecture-15.md) · [Lab sequence](Lab-Guide.md) · [Next: evaluation](Lecture-17.md)
