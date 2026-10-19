# Lab: train and challenge a small robot policy on a CPU

**Purpose:** complete the train → select → evaluate → roll out cycle with code you can read. **Prerequisites:** [math](01-Math-and-Training.md) and [evaluation](02-Evaluation-and-Robot-Learning.md). **Artifact:** a reproducible JSON report with split identities, numerical checks, baseline comparisons, and episode outcomes.

**Access:** Python 3.10 or later and a CPU. Standard library only; after installing Python, no additional package, dataset download, simulator, GPU, API key, or robot is needed. **Effort:** setup 5–15 minutes; guided experiment 45–60 minutes; analysis 30 minutes. The default calculation takes seconds or less on a typical laptop. Stop and inspect the environment if this tiny job takes longer than a minute.

## 1. Predict before running

The synthetic teacher uses `acceleration = 0.8 × error - 0.6 × velocity`. The dataset adds Gaussian action-label noise with standard deviation `0.02 m/s²`. Training estimates three parameters: two weights and a bias. The cart dynamics are defined in [the task contract](02-Evaluation-and-Robot-Learning.md#1-define-the-task-before-the-model).

Write predictions for these three comparisons:

1. Will the learned policy beat a constant zero action on held-out demonstration frames?
2. Will its RMSE reach exactly zero even if the model family can represent the teacher?
3. What happens if the policy always observes goal error `0.6 m` larger than the true error?

The third change is an observation fault. Labels and the physical goal remain unchanged.

## 2. Run the reference

Run from the repository root: the folder containing `mkdocs.yml`, rather than
this lesson's folder. If your terminal is still inside Robotics Foundations,
open another terminal at the repository root. These commands were **tested
here** on Python 3.12.3, Linux x86-64, on 2026-10-18. No third-party Python
dependencies are used.

```bash
python3 "Phase 3 - Artificial Intelligence/0. AI and Robot Learning Foundations/labs/policy_lab.py" --output /tmp/ai-policy-report.json
python3 -m unittest discover -s "Phase 3 - Artificial Intelligence/0. AI and Robot Learning Foundations/labs" -p "test_policy_lab.py" -v
```

The first command prints a JSON report: nested named fields, lists, and numbers.
It also saves the report to the requested path. `/tmp` is temporary storage;
copy reports you want to keep into your own experiment folder. Use a different
output path for each variation. The second command runs this lab's ten checks,
including independent hand calculations, numerical gradients, split leakage,
aggregation weighting, and changed-sensor behavior. The later timed-policy lab
adds its own integration checks.

Read the report in this order:

| Report field | What to look for first |
|---|---|
| `split_episode_ids` | Three non-overlapping lists of demonstration identities |
| `max_gradient_absolute_error`, `leakage_fault_caught` | Numerical agreement and a deliberately detected data fault |
| `candidates`, `selected_learning_rate` | Selection by validation error before test reporting |
| `test`, `test_zero_baseline`, `test_training_mean_baseline` | Held-out action RMSE compared with two simple prediction rules |
| `rollout`, `rollout_error_bias_0_6_m`, `rollout_zero_baseline` | Final task outcomes under paired initial states, not just action agreement |

The constant training-mean baseline uses only training labels to choose its
output. The zero-action baseline is also useful for the motion experiment: zero
acceleration preserves an existing velocity, so it does not mean “stop the cart.”

The local reference run produced these **synthetic-fixture results**:

| Check | Local result | Interpretation |
|---|---|---|
| Split | 24 / 6 / 6 episodes | Training / validation / test remain separate |
| Deliberate leakage | Caught | A copied training episode was rejected |
| Gradient maximum absolute error | `3.8e-12` | Analytical derivatives match finite differences at the checked point |
| Learned test action RMSE | `0.02193 m/s²` | Close to the noisy synthetic teacher |
| Zero baseline action RMSE | `0.23942 m/s²` | Learning improves this prediction task |
| Biased-observation action RMSE | `0.47887 m/s²` | Wrong observation semantics overwhelm prediction quality |
| Clean / biased / zero-action rollouts | `12/12`, `0/12`, `0/12` | Different behaviors under the stated toy dynamics |

These are not robot benchmarks or universal pass thresholds. Different seeds can change numerical values. Keep exact raw outputs rather than copying this table as your own measurement.

## 3. Read the code in data-flow order

Open [policy_lab.py](labs/policy_lab.py). Start with these functions:

| Function | Explain before moving on |
|---|---|
| `make_dataset` | How neighboring observations share a demonstration |
| `split_episodes` / `validate_splits` | Why no episode identifier can appear in two partitions |
| `fit_scaler` / `features` | Which rows determine normalization and where the bias feature comes from |
| `loss_and_gradient` | How each sample contributes to average loss and derivatives |
| `gradient_error` | Why both a positive and negative nudge are used |
| `train_model` | Why gradients are subtracted and why the epoch limit is bounded |
| `evaluate` | How pooled frame RMSE differs from mean episode RMSE |
| `rollout` | How the policy's action changes its next input |
| `run` | How validation selects a candidate before test evaluation |

The three candidate learning rates are `0.01`, `0.05`, and `0.2`. Each starts from the same zero parameters. Candidate selection uses validation action RMSE. Only the selected candidate is reported on the test split.

The learned weights operate on normalized inputs. To compare them with the physical teacher coefficients, convert them:

```text
physical_weight[j] = normalized_weight[j] / training_std[j]
physical_bias = normalized_bias - sum(physical_weight[j] × training_mean[j])
```

Expect physical weights near `0.8` and `−0.6`, not necessarily normalized weights with those values. This is a common reason correct models appear to have “wrong” parameters.

### Use the saved model without training again

Training changes the weights. **Inference** applies the frozen weights to a new
observation. After the reference command has created `/tmp/ai-policy-report.json`,
run this self-contained block from any folder:

```bash
python3 - <<'PY'
import json
from pathlib import Path

report = json.loads(Path('/tmp/ai-policy-report.json').read_text())
weights = report['weights_normalized_inputs']
means = report['scaler_training_mean']
scales = report['scaler_training_std']
goal_m, position_m, velocity_m_s = 0.0, 0.5, 0.2
error_m = goal_m - position_m
features = [(error_m - means[0]) / scales[0],
            (velocity_m_s - means[1]) / scales[1], 1.0]
raw_acceleration = sum(w * x for w, x in zip(weights, features))
limit = report['rollout_contract']['acceleration_limit_m_s2']
acceleration = max(-limit, min(limit, raw_acceleration))
print(f'error={error_m:.3f} m, action={acceleration:.6f} m/s^2')
PY
```

Expect an action close to the teacher's `−0.52 m/s²`, with a small difference
because the parameters were fitted from noisy labels. This command performs
one prediction; it does not train or move anything. The trailing feature `1.0`
supplies the learned bias term. Removing normalization or exchanging the two
features changes the model even when the weights are identical.

The next layer must apply that acceleration using the documented timestep and
state update. The [timed-loop lab](Lab-02-Timed-Robot-Loop.md) adds capture age,
queues and command expiry around the same learned prediction.

## 4. Make a bounded fault and catch it

Work in a copy of the script or use version control to review and restore your edit.

- **Gradient fault:** negate the derivative contribution in `loss_and_gradient`. The finite-difference check should fail before training. Explain why this is more informative than waiting for a bad final score.
- **Data fault:** the reference already tries to put one training sample into validation. Inspect the resulting overlap check. Removing an identifier from the report does not repair the actual data leak.
- **Sensor fault:** inspect `error_bias=0.6` in the rollout. The policy settles near the wrong position because its observation says that the wrong position is the goal. A correct loss implementation cannot repair a wrong sensor contract.

Choose one fault, save evidence of its detection, restore the reference, and rerun the relevant check.

## 5. Run a guided variation

First predict what 20 training epochs will change. Then run:

```bash
python3 "Phase 3 - Artificial Intelligence/0. AI and Robot Learning Foundations/labs/policy_lab.py" --epochs 20 --output /tmp/ai-policy-20-epochs.json
```

Compare validation scores, selected learning rate, test RMSE, and paired rollout outcomes. The command limits epochs to 1–2,000. Do not increase training merely to chase a lower test score. An additional seed is a reproducibility variation, not a substitute for a large independent evaluation dataset.

## 6. Record what the result means

Your report contains Python/platform versions, script and generated-data SHA-256 values, seeds, split episode IDs, normalization, candidate selection, and per-episode results. Save the command and repository revision alongside it. Wall time measures the whole CPU lab; it is not a policy inference latency benchmark.

Answer in your own words:

1. Why is the noisy teacher's achievable test RMSE not exactly zero?
2. Why does the observation bias harm a model that passed the gradient check?
3. What can this exercise establish about robot learning, and what still needs a richer simulator or physical robot?
4. Which values must travel with the model to reproduce its predictions on another machine?

<details>
<summary>Answer guidance</summary>

1. Unpredictable label noise remains even when the deterministic teacher rule is learned. A finite test sample also fluctuates.
2. The gradient check verifies differentiation at sampled values. It does not establish correct observations, task semantics, distribution coverage, or closed-loop success.
3. It establishes the implemented train/evaluate procedure and outcomes for this mathematical fixture. It cannot establish camera perception, contact behavior, real actuator response, hardware latency, or physical safety.
4. Weights, feature order, units, sign conventions, normalization means/scales, action clipping, script revision, runtime version, and evaluation configuration. Dataset and split identities are needed to reproduce the training/evaluation result.

</details>

## Ship it

Submit the reference JSON, one variation JSON, numerical prediction, caught-fault evidence, and a one-page interpretation. Include at least one failure even if the clean reference succeeds. Then complete one [independent transfer task](Guide.md#independent-transfer-and-completion) and continue to [Neural Networks](../1.%20Neural%20Networks/Guide.md).

**Next simulator step:** [run this same saved policy in MuJoCo](Lab-03-CPU-Physics.md). Compare the original equations with actual CPU-engine dynamics, then change mass, force limits and physics timestep. Its package environment is optional; the training exercise above remains dependency-free.
