# CPU lab: get the return target right before training

Compute a tiny actor-critic rollout by hand, compare it with a runnable reference, and expose the two boundary bugs that can invalidate a much larger training run. This lab accompanies [Lecture 05](Lecture-05.md) and [Lecture 06](Lecture-06.md). It needs only Python 3.10+ and finishes in seconds.

**Tested here:** Python 3.12.3, 2026-10-18; all 16 standard-library tests pass. The result is synthetic arithmetic, not a learned policy or a simulator benchmark. Budget 15 minutes for the worked calculation, 20 minutes for the variation, and 15 minutes for the explanation.

## One rollout, two different endings

Use discount `gamma=0.9` and GAE decay `lambda=0.8`. Rows 0–1 belong to one episode; row 1 is cut by an external time limit. Row 2 starts another episode and reaches a true terminal state.

| Row | Reward | V(current) | V(true post-action state) | Terminated | Truncated |
|---|---:|---:|---:|---|---|
| 0 | 1 | 0.5 | 1 | false | false |
| 1 | 2 | 1 | 4 | false | true |
| 2 | 3 | 9 | 0 | true | false |

The **bootstrap mask** is zero only on true termination. The **trace mask** is zero at either termination or truncation, because the next stored row then belongs to another episode:

```text
delta[t] = reward[t] + gamma * (not terminated[t]) * next_value[t] - value[t]
adv[t] = delta[t] + gamma * lambda * (not (terminated[t] or truncated[t])) * adv[t+1]
target[t] = adv[t] + value[t]
```

At the end of the collected buffer, set the future advantage carry to zero. This is a rollout cut, not necessarily an MDP terminal state: its last `delta` still uses the true next-state value if the episode continues.

Work backward:

1. Row 2 terminates: `delta = 3 - 9 = -6`, `adv = -6`, target `3`.
2. Row 1 truncates: `delta = 2 + 0.9*4 - 1 = 4.6`. Stop the trace so row 2 cannot leak backward. Target is `5.6`.
3. Row 0 continues: `delta = 1 + 0.9*1 - 0.5 = 1.4`; `adv = 1.4 + 0.9*0.8*4.6 = 4.712`; target is `5.212`.

From the repository root, enter this course and run:

```bash
cd "Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Deep RL for Robot Learning"
python3 labs/returns.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

The expected advantages are `[4.712, 4.6, -6.0]`, and value targets `[5.212, 5.6, 3.0]`.

## Make two wrong implementations disagree

If row 1 is incorrectly treated as terminal, its target becomes `2`, losing `3.6` of bootstrap value. If a same-step auto-reset adapter incorrectly supplies the new episode's value `9` rather than the true final-state value `4`, the target becomes `2 + 0.9*9 = 10.1`. These are different bugs with opposite effects in this fixture.

Use the environment's documented reset mode. In a same-step mode, preserve the terminal observation from the information returned by the environment. In a next-step mode, the reset-only event must not enter the rollout as an action-caused transition. Names of final-observation fields differ by library/version; inspect the adapter rather than guessing. [Farama autoreset modes](https://farama.org/Vector-Autoreset-Mode)

A task with a true finite horizon needs that horizon represented in the state when it changes optimal decisions. An external training time limit on a continuing task is different. Label the task semantics first, then select masks. [Gymnasium termination/truncation explanation](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)

## Check PPO's sign before optimizing

PPO maximizes `min(ratio*A, clip(ratio,1-eps,1+eps)*A)`. It does not simply clip every ratio and use the result. For `eps=0.2`:

| Advantage | Ratio | Unclipped contribution | Clipped contribution | Chosen objective |
|---:|---:|---:|---:|---:|
| +2 | 1.5 | 3.0 | 2.4 | 2.4 |
| −2 | 0.5 | −1.0 | −1.6 | −1.6 |
| −2 | 1.5 | −3.0 | −2.4 | −3.0 |

The last row retains a penalty for increasing the probability of a bad action. A loss minimized by gradient descent negates this objective. Clipping removes incentives for some excessive changes; it does not enforce a hard probability-ratio constraint.

## Practice and answer guidance

**Guided variation:** change row 2's reward from 3 to 900. Rows 0–1 must not change, because row 1 ends the trace. If they do, your implementation leaks information across resets.

**Independent task:** build a complete three-step terminal episode with your own rewards and values. Set `lambda=1`, and compare every target against a direct discounted sum of rewards. Then cut the rollout before termination and supply an intentionally wrong bootstrap value: explain why the target now depends on that error. `lambda=1` does not remove bootstrap bias when the collected trajectory stops before the true end.

**Self-check:** Should both `terminated=True` and `truncated=True` bootstrap? No: true termination takes precedence. Does a successful GAE test prove training will improve? No: policy gradients, reward semantics, data collection, and optimization still need independent checks. Can a negative explained variance identify this bug uniquely? No; it flags poor predictions but does not by itself distinguish masks, observations, optimization, or an almost-constant target distribution.

**Artifact:** save the input table, manual calculation, program output, one detected bug, and a new complete-episode comparison. Carry the verified two-mask implementation into the GPU lab, where you must also validate the actual final-observation adapter.


## Preserve the objective when executing action chunks

A policy may choose two controls before observing again. With per-control discount gamma=0.9, rewards [1,2], and next-state value 10, the correct chunk target is:

`1 + 0.9×2 + 0.9²×10 = 10.9`.

If that chunk truly terminates the episode, the target is 2.8. If it is cut externally after only the first control, the target uses the **actual one-step duration**: `1 + 0.9×10 = 10`. The new `discounted_chunk_target` function and four tests check these boundaries. Shortening the number of decisions does not let you keep the old reward/discount units unchanged; [Lecture 13](Lecture-13.md#24-effective-horizon-and-why-chunks-help) explains the theoretical consequence.

## Optional: check the exact lecture functions with real CPU frameworks

The [optional tests](labs/optional/test_lesson_numerics.py) extract named function definitions directly from Lectures 06, 11, 13, and 14. They do not create a second implementation of the lesson. Independent checks use scalar log-density/gradient calculations, `torch.distributions.Normal`, exact enumeration of four Bernoulli outcomes, a known Wilson endpoint, a deliberately correlated bootstrap fixture, and hand-computed group weights that distinguish positive-advantage filtering from success-only imitation. Small temperatures must not overflow exponentiation.

Fifteen tests passed on 2026-10-18 using Python 3.12.3, PyTorch **2.14.0+cpu**, NumPy **2.3.5**, and SciPy **1.18.1**. The review reused an existing isolated environment with those versions; no simulator, pretrained weights, GPU execution, or complete RL trainer was involved. The [validation record](labs/numerical-validation.json) records source hashes and boundaries; [upstream source evidence](labs/rl-source-evidence.json) pins the inspected policy/framework code.

If you already have the shared course PyTorch CPU environment, reuse it and add the small SciPy requirements file. To create the documented Linux x86-64 / CPython3.12 environment from this **course directory**:

```bash
python3.12 -m venv /tmp/course-rl-cpu-venv
/tmp/course-rl-cpu-venv/bin/python -m pip install --require-hashes -r "../../../Phase 3 - Artificial Intelligence/2. Deep Learning Frameworks/PyTorch/CPU-Robot-Stop-Lab/labs/requirements-linux-cpu-py312.txt"
/tmp/course-rl-cpu-venv/bin/python -m pip install --require-hashes --no-deps -r labs/requirements-scipy-linux-py312.txt
/tmp/course-rl-cpu-venv/bin/python -m unittest discover -s labs/optional -v
```

The SciPy file is an addition to the complete shared CPU lock: its NumPy dependency is already installed by that lock. Other operating systems/Python versions need a corresponding resolved dependency set. Optional tests explicitly skip unavailable frameworks; skipped tests are not passed numerical validation.

Three results to explain:

1. With two independent Bernoulli samples at success probability 0.3, the logit return gradient is 0.21. The inclusive group-mean estimator has expectation **0.105**; leave-one-out recovers **0.21**. Std normalization and PPO clipping introduce further changes.
2. A zero-intermediate-noise flow sampler can still draw a random initial latent. Sampling with zero sigma is possible; the Gaussian transition log-density at zero sigma is undefined and rejected.
3. Two training runs score `[0,1]` and `[1,0]` on two tasks. Resampling whole shared runs preserves their mean 0.5; resampling task columns independently invents combinations and produces a different uncertainty interval. Choose the sampling unit from how the policies were trained.

Carry these numerical checks into a real integration only after validating episode IDs, final observations, masks, action units, sampler noise, and reference-policy state. A passing snippet test establishes its tested mathematics; it does not establish that PPO or VLA RL improves a robot task.
