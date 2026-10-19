# Train, evaluate, and export a small robot stop classifier

**Purpose:** complete a real PyTorch experiment on a CPU and learn which evidence survives a change in sensor input. **Prerequisites:** Python functions/lists, weighted sums, and the [AI foundations](../../../0.%20AI%20and%20Robot%20Learning%20Foundations/Guide.md). **Time:** setup 5–15 minutes, reading and guided work 60–90 minutes, independent transfer 45–60 minutes. The default training run takes seconds on the tested host; it is bounded to 80 epochs.

**Artifact:** trained weights, an exported graph, and a JSON report with splits, errors, hashes, and a shift experiment. **Hardware connection:** count parameters and operations, distinguish a framework graph from a target executable, and preserve feature/scaling contracts before optimizing inference. This lab trains a nonlinear network; the earlier standard-library policy lab explains the arithmetic beneath a simpler model.

## 1. Understand the task before training

Our original synthetic fixture describes a point robot moving toward an obstacle. It supplies distance `d` in meters, speed `v` in m/s, available constant braking deceleration `a` in m/s², and response latency `t` in seconds. Its teaching rule is:

```text
required distance = v*t + v²/(2*a) + 0.2 m margin
stop_required = 1 when observed distance <= required distance
```

For `v=2`, `a=2`, and `t=0.1`, the required distance is `0.2+1+0.2=1.4 m`. At a distance of 1.3 m the label is one; at 1.5 m it is zero. The quadratic term makes the decision nonlinear in the raw features.

**Why synthetic data here?** We know the generating rule and can create a controlled sensor defect without downloading private or licensed robot logs. The generator is original course code under the repository's [MIT license](../../../../LICENSE); no outside dataset is copied. These are static classification examples with simplified physics, not real robot measurements or a closed-loop safety controller. The exact analytic rule should outperform an approximation on clean data. Beating it is not the objective.

**Diagnostic:** why does twice the speed need more than twice the braking distance? Why can't we train on one frame and test on a neighboring frame as if they were independent? Why is 77% accuracy potentially useless here? **Answers:** braking includes `v²`; frames share episode conditions; predicting “continue” for every case can achieve high accuracy while missing every required stop.

## 2. Install the tested CPU environment

Run from this lab directory. The exact lock targets **Linux x86_64, CPython 3.12**, and contains hashes for PyTorch **2.14.0+cpu**, NumPy **2.3.5**, and all installed dependencies. The CPU wheel is about 196 MB; allow roughly 1 GB for the environment. These commands were tested on 2026-10-18, including a fresh forced install with hash verification.

```bash
python3.12 -m venv /tmp/course-pytorch-cpu-venv
/tmp/course-pytorch-cpu-venv/bin/python -m pip install --require-hashes -r labs/requirements-linux-cpu-py312.txt

# Full framework experiment: no API keys, network calls, dataset downloads, or GPU.
/tmp/course-pytorch-cpu-venv/bin/python labs/train_stop_model.py --output /tmp/robot-stop-run

# Portable fixture tests; PyTorch is not required for these.
python3 -m unittest discover -s labs -p 'test_*.py' -v

# Actual framework tests: run separately inside the environment above.
/tmp/course-pytorch-cpu-venv/bin/python -m unittest discover -s labs/optional -p 'test_*.py' -v
```

On another OS, architecture, or Python version, use the [official CPU installation selector](https://pytorch.org/get-started/locally/) in a new environment and record a separate dependency lock. The supplied binary hashes do not claim cross-platform compatibility. Optional framework tests are deliberately separate from the repository's standard-library test discovery.

## 3. Follow one complete experiment

Read [data.py](labs/data.py) and [train_stop_model.py](labs/train_stop_model.py) in this order:

1. **Generate and split.** There are 144 episodes, each containing 24 examples. Whole episodes go to training/validation/test as 96/24/24 episodes, or 2304/576/576 examples. Deceleration and latency are shared within an episode. A test rejects episode overlap. Dataset and split seeds are fixed separately from the model seed.
2. **Normalize using training only.** Compute four means and standard deviations from training examples. Register them as model buffers so saving/export includes the same transformation used during training. Recomputing normalization on test or production inputs would change the model.
3. **Fit a nonlinear model.** Four inputs → 32 hidden units → ReLU → 32 hidden units → ReLU → two logits. There are 1282 trainable parameters. `CrossEntropyLoss` consumes raw logits and integer class labels; softmax is used later to inspect probabilities. AdamW updates weights, and gradients are cleared before each update.
4. **Select on validation.** Keep the epoch with lowest validation cross-entropy, then select a threshold from 0.05 through 0.95 using validation cost `5*missed_stops + unnecessary_stops`. This cost is declared before evaluation and is only a pedagogical example. Test data selects neither the epoch nor the threshold.
5. **Evaluate once under the stated protocol.** Report confusion counts, accuracy, stop recall/precision, Brier score, cross-entropy, high-speed/lower-speed slices, and individual errors. Compare with the training-majority baseline and the analytic rule.
6. **Challenge the sensor contract.** Add +0.8 m to the *observed* distance while retaining the original physical labels and episode identities. Evaluate the fixed network and analytic rule again. Relabeling from corrupted inputs would hide the defect.
7. **Save, reload, export, verify.** Save the state dictionary, restore it into a new model, and compare logits. Export with a dynamic batch dimension, save/load the exported graph, compare batches of 1/7/32, and deliberately pass five features to check shape rejection. Swapping distance and speed preserves shape but changes meaning; a separate numerical negative test detects that failure.

## 4. Read the measured result

Default measured run: Linux x86_64, Python 3.12.3, PyTorch 2.14.0+cpu, one PyTorch thread, seed 31, 80 epochs. The selected epoch was 65 and threshold 0.30. These are **local synthetic-fixture results**, not claims about a real robot or guaranteed results on another platform.

| Evaluation | Accuracy | Required-stop recall | Missed stops / 131 |
|---|---:|---:|---:|
| Training-majority baseline | 77.26% | 0% | 131 |
| Exact analytic rule, clean observations | 100% | 100% | 0 |
| Learned network, clean observations | 99.65% | 99.24% | 1 |
| Learned network, +0.8 m distance bias | 94.10% | 74.05% | 34 |
| Analytic rule, same biased observation | 94.10% | 74.05% | 34 |

Clean test cross-entropy was 0.01566 nats/example; biased-input cross-entropy was 0.50631. The [retained measurement report](labs/measured-run-2026-10-18.json) includes the full validation history, errors, split identities, and source hashes. Reload and export logit differences were zero for the tested batches. This does not prove equality for every input. The entire experiment, including export checks, took about 1.8 seconds on this host; that is **not** per-inference latency or a robot deadline measurement.

The output directory contains `state_dict.pt`, `stop_model.pt2`, and `report.json`. The report records normalization, feature order, script/dataset hashes, split IDs, validation history, threshold candidates, and test errors. `torch.export` produces a constrained graph, not a compiled device deployment. Its saved format can change across PyTorch releases; retain the environment and rerun parity when changing it. See the official [serialization tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) and [export reference](https://docs.pytorch.org/docs/stable/export.html).

## 5. Guided variations and answers

- **Use `--distance-bias-m 0`.** Predict the result before running. The two evaluation summaries should agree because both now receive identical observations; training and selection remain unchanged.
- **Use `--epochs 5`.** Predict a larger approximation error. Check the actual result instead of assuming more epochs always improve held-out quality. Validation selection still applies within the five available epochs.
- **Inspect an error near the boundary.** Recompute its required distance by hand. Is it a tiny margin error or a large mistake? Average accuracy does not distinguish these.
- **Inspect the majority baseline.** Its precision is `null` because it predicts no positives; treating an undefined denominator as a perfect precision score would mislead the comparison.
- **Count storage and compute.** The parameters use `1282*4=5128` bytes in FP32; eight normalization buffer elements add 32 bytes. Serialized files also have metadata. The three linear layers perform `4*32+32*32+32*2=1216` multiply-accumulates per example before biases, normalization, and activations. These counts do not predict measured latency by themselves.

The guided runs were also executed in the pinned environment. Zero bias produced identical clean/shifted summaries. Five epochs selected epoch 5 and threshold 0.40, with 96.18% clean accuracy and seven missed stops. Seed 32 selected epoch 80 and threshold 0.35, with 99.13% accuracy and two missed stops. These comparisons illustrate variation; they are not permission to select a seed using the test score.

## 6. Independent transfer and completion

Change one supported aspect: use seed 32, use a −0.8 m distance bias, or implement a new evaluation split with low-deceleration episodes withheld. Write a prediction before running. Keep training, selection, and final evaluation separate; treat an already inspected test set as development evidence if you tune against it, then create a new untouched final split.

Submit the exact command, report, environment lock, one hand calculation, one error example, and a comparison with the analytic baseline. Explain why export parity cannot repair an incorrect input and why a learned approximation is unnecessary when the exact physical rule and trustworthy inputs are already available. Use the [shared completion rubric](../../../../Learning-Guide.md#completion-rubric): 8/10, with full correctness and reproduction marks.

Next, take this numerical contract into the [CPU compiler lab](../../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md) or [ML engineering](../../../Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Guide.md). Keep deployment timing, closed-loop behavior, and real sensor calibration as separately measured outcomes.
