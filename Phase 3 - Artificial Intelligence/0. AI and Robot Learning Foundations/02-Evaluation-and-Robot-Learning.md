# Evaluation that connects predictions to robot behavior

**Purpose:** design an evaluation that can reveal a model's failures before optimizing its runtime. **Before:** [math and training](01-Math-and-Training.md). **After:** [CPU policy lab](Lab-01-CPU-Policy.md).

## 1. Define the task before the model

“Build an AI robot” is too broad to evaluate. A useful task contract states observations, actions, conditions, and outcomes.

| Field | Our teaching fixture |
|---|---|
| Task | Bring a cart to position 0 m and settle |
| Observation | Goal error in metres; velocity in metres/second |
| Action | Acceleration in metres/second², clipped to `[-1, 1]` |
| Simulation update | `velocity += acceleration × dt`; then `position += velocity × dt` |
| Timing | `dt = 0.1 s`; one policy evaluation per simulation step |
| Episode | A fresh initial position/velocity, then 200 steps |
| Success | At the final step, `abs(position) < 0.1 m` and `abs(velocity) < 0.1 m/s` |
| Required comparisons | Zero-action baseline, learned policy, biased-observation policy |
| Limitations | No contacts, obstacles, actuator dynamics, real sensors, or hardware timing |

The final-step criterion is intentionally simple. A harder transfer exercise can require staying inside the goal region for several consecutive steps. State that change before comparing results.

For one concrete update, start at `position=0.5 m` and `velocity=0.2 m/s`.
The error is `-0.5 m`; the teacher requests `-0.52 m/s²`. With `dt=0.1 s`,
velocity becomes `0.148 m/s`, then position becomes `0.5148 m`. The cart is
braking but still moving away from the goal. This is why the success condition
checks velocity as well as position. The [action-interface bridge](../../Beginner-Reading-Aid.md#2-the-same-cart-has-two-different-action-interfaces)
connects this model to the first velocity-controlled cart.

## 2. Separate learning, selection, and reporting

Training data changes learned parameters. Validation data helps choose learning rate, architecture, thresholds, and stopping point. Test data estimates performance after those choices are fixed.

```text
Split episodes
  -> fit preprocessing on training data
  -> train candidates on training data
  -> select candidate using validation data
  -> evaluate frozen candidate on test data
```

Do not reuse the test score to repeatedly choose a better candidate. If you change the design after studying test failures, that set has become development evidence; reserve fresh episodes for a subsequent final claim. Fit transformations using training data and reuse them unchanged on other splits. This includes feature selection and imputation, not only scaling. [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) explains the same rule; reviewed 2026-10-18.

### A worked leakage example

You have two demonstrations. Episode A shows a red block on one table; episode B shows a blue block on another table. Each contains 100 adjacent camera frames. Randomly assigning 80 frames from each episode to training and the remainder to test lets the test share the same scenes and trajectories as training.

If the claim is “works on a new demonstration,” put whole episodes in separate sets. If the claim is “works with new objects,” group all episodes of an object together. If the claim is “works tomorrow,” use a time split and check whether repeated objects also leak across the boundary. A group split supports the grouping you chose; it does not automatically establish generalization to every new condition. See [grouped and time-aware validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

**Guided variation:** three episodes contain the same physical block. Which split supports a new-block claim? **Answer:** all three belong to one object group, so they must remain on the same side of the split.

## 3. Compare against something simple

For regression, compare against a constant predicted from training labels, as well as any relevant existing controller. For classification, compare against a majority-class or simple rule baseline. For a language agent, compare against a fixed workflow with the same tools and permissions.

Suppose a detector sees 10 hazards in 1,000 images and predicts “no hazard” for all of them. Accuracy is 990/1,000 = 99%; hazard recall is 0/10 = 0. A large accuracy number can describe a useless detector.

```text
precision = true positives / (true positives + false positives)
recall    = true positives / (true positives + false negatives)
```

Report how undefined denominators are handled. Choose thresholds using validation data and the task's error costs. For detection and segmentation, define matching/overlap rules too; a classification confusion matrix alone does not evaluate spatial localization.

For the cart, use action RMSE with units and per-episode values. The lab reports both pooled frame RMSE and the mean of episode RMSEs. These statistics can differ when episode lengths or difficulty differ. Do not silently switch the weighting between experiments.

## 4. Small evaluations have uncertainty

If a policy succeeds in 8 out of 10 episodes, the observed success rate is 80%. It is not evidence that exactly 80% of all future tasks will succeed. Use enough independently chosen episodes to reveal variation; report the count and conditions alongside the rate.

Repeated frames are not independent robot trials. For a paired comparison, run both policies on the same initial conditions and report which episodes changed outcome. In a larger study, bootstrap whole episodes or use an interval appropriate for independent binary outcomes. A larger sample does not fix a biased test distribution.

**Failure review:** keep the inputs, target, prediction, units, and episode identifier for large errors. Ask whether the cause is a unit error, missing observation, poor labels, limited model capacity, or a changed environment. “Train longer” addresses only some of these causes.

## 5. How the learning families connect

| Method | Training signal | Robot example | Main question to test |
|---|---|---|---|
| Supervised learning | Input and target pairs | Predict depth or object labels | Does the target represent the deployed task? |
| Self-supervised learning | Targets constructed from data | Predict masked image patches or future observations | Does the representation transfer to the task? |
| Behavior cloning | Expert observation/action pairs | Imitate a demonstration policy | What happens after the policy leaves expert states? |
| Reinforcement learning | Rewards from interaction or logged transitions | Learn a locomotion or manipulation policy | Does reward improvement correspond to actual task success? |
| Model-based control | Dynamics model and objective | Predict future trajectories and choose actions | Is the model accurate where the controller acts? |

A neural network is a model family used inside several of these methods. Behavior cloning is supervised learning applied to actions. Reinforcement learning learns from rewards, which differ from expert action labels. Offline RL learns from a fixed transition dataset; it does not automatically remove the difficulty of actions that lack data support.

## 6. Open-loop error and closed-loop success

**Open-loop evaluation:** feed a saved observation to the policy and compare its action with a saved target. The policy's action does not change the next saved observation.

**Closed-loop evaluation:** apply the action, update the environment, and feed the resulting observation back to the policy. Errors can change where the policy goes and what it sees next.

A small steering error may move a robot into a view absent from its demonstrations. This dependence between actions and future observations is a central reason behavior cloning needs rollout evaluation. DAgger addresses this by collecting expert labels on states visited by the learner; it requires an appropriate expert and interaction procedure. The foundational result is [Ross, Gordon, and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html).

The lab first measures action error, then rolls out the learned policy. It also adds a fixed bias to the observed position error. A policy can be a good fit to clean demonstrations and fail this changed sensor condition. This is a useful result, not a reason to conceal the failed run.

## 7. From a small policy to modern robot AI

A visual policy replaces or supplements low-dimensional observations with images. A vision-language-action (VLA) model additionally conditions on language and produces a specified action representation. “Action” might mean joint targets, relative end-effector motion, or a sequence of commands; these are not interchangeable formats.

Two demonstrations can take different valid paths around an obstacle. A model trained with squared error may predict their mean, potentially going into the obstacle. Richer action distributions and sequence models address multimodality. For example, [Diffusion Policy](https://arxiv.org/abs/2303.04137) learns action distributions through a conditional diffusion process. Its existence is not evidence that a particular deployment meets your success, memory, or timing limits. Published 2023; source checked 2026-10-18.

Follow the [VLA foundations course](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/README.md) after this foundation. Carry the same split manifest, normalization contract, and rollout metrics forward. For action chunks, record horizon, execution prefix, replanning period, and how stale observations are handled.

## 8. An agent is a different control loop

A language agent chooses tools and updates task state. A robot controller repeatedly turns observations into physical commands. A planning agent can propose a goal such as “inspect shelf A”; a typed interface, motion planner, and controller must translate that proposal into validated actions.

For a 50 Hz controller, one tick is `1/50 = 0.02 s = 20 ms`. A planner taking 500 ms spans 25 ticks. Keep the planning request asynchronous, define what the controller does while waiting, and reject stale results. Measure end-to-end timing; a model's token rate does not determine a robot's control frequency.

**Worked fault:** a planner returns a valid-looking target after the scene has changed. Correct JSON only establishes syntax. The receiver must check task identity, frame, units, validity time, and current execution permission. Continue with [agentic AI](../Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) for the application runtime.

## Exit questions

1. Your validation score improves but test performance falls after tuning on the test set. Which result is still a final held-out estimate?
2. Your behavior-cloned policy has half the action RMSE of another policy. Must it have higher task success?
3. A model has 40 ms inference latency. Can it directly provide a fresh action every 20 ms on one serial execution stream?

<details>
<summary>Answer guidance</summary>

1. The reused test set no longer provides an untouched final estimate. Freeze the next design and evaluate on a fresh, appropriate held-out set.
2. No. Error direction, sensitive states, feedback dynamics, and observation shift can change outcomes. Compare paired closed-loop episodes.
3. No under those assumptions. Change the controller/policy architecture, lower the policy update rate, or reduce measured latency. Pipelining may increase throughput, but it does not make each observation fresher; report observation age as well as completion rate.

</details>
