# AI and robot learning foundations

Learn enough mathematics, machine learning, and evaluation to explain a small robot policy before training a large model.

**Layer mapping:** L1 workload definition; L2/L5 implications for tensor operations and memory. **Roles:** robot learning, ML, edge inference, and accelerator engineers.

**Prerequisites:** write a Python function, use a list, and run a terminal command. No calculus course, GPU, robot, account, or paid API is required. If Python is new, first write a function that averages five sensor readings and rejects an empty input.

If the words in the diagnostic are unfamiliar, use the [beginner reading aid](../../Beginner-Reading-Aid.md)
and start with lesson 1. The diagnostic finds your starting point; it is not an
entry exam. The layer labels above are for the later
[hardware design route](../../Physical-AI-Design-Studio.md).

**Next:** [Neural Networks](../1.%20Neural%20Networks/Guide.md), then [Deep Learning Frameworks](../2.%20Deep%20Learning%20Frameworks/Guide.md). Robotics learners can then use the [robotics track](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Guide.md).

## Why begin with a tiny model?

A robot policy is a rule that maps observations to actions. A large network and a three-parameter linear model both need an input contract, training data, a loss, and an honest evaluation. With three parameters, you can inspect every calculation. Later, the neural network changes the model's capacity; it does not remove those responsibilities.

Our running example is a simulated cart moving along one line. Its inputs are signed goal error (`goal - position`) and velocity. Its output is acceleration. This is a mathematical teaching fixture, not a model of a particular robot.

If you came from [Start here](../../Start-Here.md), note the changed contract:
the goal is now `0 m`, and acceleration changes velocity before velocity changes
position. The [worked bridge](../../Beginner-Reading-Aid.md#2-the-same-cart-has-two-different-action-interfaces)
shows why a braking cart can initially keep moving away from the goal.

## A short entry diagnostic

Spend ten minutes without searching. Each question points to a specific lesson; missing one does not mean restarting the whole roadmap.

1. A batch contains 8 observations, each with 3 features. A linear layer produces 2 outputs per observation. What are the input, weight, bias, and output shapes?
2. A target acceleration is 0.6 m/s² and the prediction is 0.2 m/s². What are the squared error and its derivative with respect to the prediction?
3. You recorded 20 robot demonstrations with 100 neighboring frames each. Why can randomly splitting the 2,000 frames give a misleading test score?
4. Your policy predicts expert actions accurately on recorded frames. What must you measure when its own actions move the robot?
5. A classifier is 99% accurate on a dataset containing 99% empty scenes. What simple baseline should you compare against?

<details>
<summary>Answers and where to recover</summary>

1. `[8, 3] @ [3, 2] + [2] -> [8, 2]`, using this course's row-vector convention. There are 8 parameters: 6 weights and 2 biases. Study [math and training](01-Math-and-Training.md).
2. `(0.2 - 0.6)² = 0.16 (m/s²)²`; derivative `2(0.2 - 0.6) = -0.8` in the corresponding loss-per-output units. A small positive change in prediction reduces this loss. Study the worked gradient update.
3. Frames from the same demonstration share objects, lighting, and motion. Hold whole demonstrations out; if deployment needs new sites or objects, split at those levels too. Study [evaluation and robot learning](02-Evaluation-and-Robot-Learning.md).
4. Task success, failures, state trajectories, constraint violations, and observation-to-action timing over held-out episodes. The policy changes its future inputs.
5. Always predicting “empty” also achieves 99%. Inspect positive-class precision/recall and false negatives at a fixed threshold; accuracy alone cannot establish useful detection.

</details>

## Learning path

| Session | Learn and do | Evidence | Planning time |
|---|---|---|---|
| 1 | [Math and training](01-Math-and-Training.md): units, shapes, weighted sums, derivatives, gradient descent | One hand-worked forward pass and parameter update | 60–90 min |
| 2 | [Evaluation and robot learning](02-Evaluation-and-Robot-Learning.md): splits, baselines, uncertainty, behavior cloning, feedback | A task contract and evaluation plan | 60–90 min |
| 3 | [CPU policy lab](Lab-01-CPU-Policy.md): run, inspect, break, explain | JSON report, raw episode metrics, detected fault | Setup 5–15 min; lab 45–60 min; analysis 30 min |
| 4 | [Timed robot loop](Lab-02-Timed-Robot-Loop.md): reuse the trained policy under queue, delay and deadline changes | Paired trajectories and a capture-to-action timeline | 75–120 min |
| 5 (optional) | [Actual CPU physics](Lab-03-CPU-Physics.md): run the saved policy in MuJoCo and change mass/force/timestep | Matched trajectory, simulator warnings caught, controlled model mismatch | 60–90 min plus package setup |
| 6 | Independent transfer below | Your own reproducible experiment | 1–2 h |

Break a session into smaller pieces when needed. Return to the diagnostic after two days; explain the answers without opening this page.

## What you will be able to do

- Distinguish a model, its learned parameters, and the objective used to train it.
- Trace tensor shapes, attach units to inputs and outputs, and estimate parameter storage.
- Check a gradient numerically before using it to train a model.
- Fit preprocessing using training data and keep evaluation episodes separate.
- Compare a trained model with a simple baseline and report per-episode results.
- Explain why low prediction error and successful closed-loop behavior are different evidence.
- Hand a numerical reference, input/output contract, and evaluation set to the next course.

## Independent transfer and completion

Choose **one** variation after reproducing the CPU lab:

1. **Data:** compare evaluation-only Gaussian observation noise at **0.05 m and 0.10 m** standard deviation using the commands below. Training data, selected model, and initial episodes stay fixed; separate noise seeds 202/203 are recorded. Compare `test_observation_noise` and `rollout_observation_noise`. The clean baseline has zero observation noise; the dataset already has action-label noise, which this option does not alter. Predict RMSE and success changes, then explain why noisier observations need not change a thresholded success count on every small sample.
2. **Model:** add a nonlinear feature such as `error * abs(error)`. Compare the linear and extended models using validation data, then evaluate the selected model on a fresh held-out test partition. Count the additional operations and parameters.
3. **System:** delay observations by one or more simulation steps. Record the delay in seconds, plot or tabulate position over time, and compare task success against the unchanged reference. Do not label simulated delay as measured inference latency.

For the data variation, start at the **repository root**, as in the CPU policy
lab. Each output is a separate file:

```bash
python3 'Phase 3 - Artificial Intelligence/0. AI and Robot Learning Foundations/labs/policy_lab.py' --seed 7 --observation-noise-std-m 0.05 --output /tmp/ai-policy-noise-005.json
python3 'Phase 3 - Artificial Intelligence/0. AI and Robot Learning Foundations/labs/policy_lab.py' --seed 7 --observation-noise-std-m 0.10 --output /tmp/ai-policy-noise-010.json
```

Submit the code, exact command, script checksum, JSON output, prediction, one caught fault, and a paragraph stating what the fixture cannot prove. Use the [shared 10-point rubric](../../Learning-Guide.md#completion-rubric): at least 8/10, with full correctness and reproduction marks. A beautiful plot without a valid split does not pass.

**Ready to continue:** you can explain why the learned model needs its normalization statistics, why episodes are the evaluation unit, and why successful simulation does not demonstrate a real robot's safety or timing.
