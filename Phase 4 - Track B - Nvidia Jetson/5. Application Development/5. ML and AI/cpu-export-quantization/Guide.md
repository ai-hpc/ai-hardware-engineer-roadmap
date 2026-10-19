# CPU lab: export a trained policy and expose bad calibration

**Actual framework run: 18 October 2026.** This experiment trains a small PyTorch classifier, exports it through the current ONNX exporter, executes it with ONNX Runtime, inserts static INT8 quantization, and compares good and deliberately poor calibration. No GPU, model download, paid API or physical robot is needed.

**Prerequisite:** the [PyTorch stop-model lab](../../../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md). Its original synthetic data, group split and training function are reused directly. Allow 45–90 minutes to run and explain the experiment; the computation itself is much shorter on the tested host.

## 1. Understand the task before running it

A point robot with speed `v`, response delay `t`, and constant deceleration `a` requires stopping distance `v*t + v*v/(2*a) + 0.2` metres. The label says whether the observed obstacle distance is within that distance. A `4 → 32 → 32 → 2` network learns an approximation to this known rule.

There are 144 original synthetic episodes with 24 observations each. A fixed group split gives 96 training, 24 validation and 24 test episodes. Normalization uses training data, the checkpoint and threshold use validation data, and the test set is evaluated after those selections. The model has 1,282 parameters. The labels and fixture are teaching material under the repository license; no outside dataset was copied.

The additional experiment calibrates only on training inputs. It compares 256 distinct examples with one example repeated 256 times. Both choices are fixed before evaluation. This intentionally bad variant teaches how activation clipping can change decisions; it is not a search using test outcomes.

## 2. Create the tested CPU environment

From the repository root, on Linux x86_64 with Python 3.12:

```bash
python3.12 -m venv /tmp/course-export-cpu
/tmp/course-export-cpu/bin/python -m pip install --require-hashes -r \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/cpu-export-quantization/labs/requirements-linux-cpu-py312.txt'
```

The full lock includes PyTorch **2.14.0+cpu**, NumPy **2.3.5**, ONNX **1.23.2**, ONNX Runtime **1.31.0**, ONNX Script **0.7.2**, and all transitive dependencies with wheel hashes. It is a host exercise lock, not a Jetson wheel recommendation. If you already created the earlier course PyTorch environment, the additional ONNX packages total about 35 MB of downloads; a fresh installation also downloads the CPU PyTorch wheel.

Run:

```bash
/tmp/course-export-cpu/bin/python \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/cpu-export-quantization/labs/export_quantize.py' \
  --output /tmp/course-export-result
```

Outputs include the training report/checkpoint, FP32 exported/preprocessed ONNX, two quantized ONNX files, and `report.json`. The repository retains the [measured report](labs/measured-run-2026-10-18.json); trained weights are reproducibly generated rather than committed.

The [source-review record](labs/source-evidence-2026-10-18.json) lists reviewed upstream URLs, immutable model/source metadata, and the stages that were and were not executed across these Jetson ML lessons.

## 3. Follow the transformations

1. Train using the existing function and reload with `weights_only=True` from the locally generated checkpoint.
2. Export with `dynamo=True`, opset 21, named tensors and a dynamic batch dimension. ONNX validation checks graph structure.
3. Preprocess the graph for quantization, then compare runtime logits against PyTorch at batches 1, 7 and 32.
4. Insert signed INT8 `QuantizeLinear`/`DequantizeLinear` nodes around supported linear operations, with per-channel weight scales.
5. Run both calibration variants on the fixed test set and on a +0.8 m sensor-bias variant. Reload a fresh runtime session and verify output preservation.
6. Swap distance and speed as a deliberate preprocessing failure.

A quantized value is `clip(round(x/scale)+zero_point)`. Calibration determines ranges; values beyond those ranges saturate. Q/DQ nodes specify numerical behavior. Their presence alone does not establish that every operation became an integer kernel on every provider.

The public inference wrapper requires finite float32 input `[batch,4]` with batch 1–1024. ONNX's symbolic batch annotation does not by itself preserve all export guards, so application validation remains necessary. [PyTorch exporter](https://docs.pytorch.org/docs/stable/onnx.html), [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## 4. Interpret the actual result

| Variant | Clean recall | Clean false negatives | Recall with distance bias |
|---|---:|---:|---:|
| FP32 ONNX | 99.24% | 1 / 131 | 74.05% |
| Representative INT8 QDQ | 99.24% | 1 / 131 | 74.81% |
| One repeated calibration row | 64.89% | 46 / 131 | 54.20% |

FP32 export maximum logit error was `3.81e-6` across the three tested batch sizes, within the declared `1e-5` tolerance. Fresh quantized session reload error was zero. Swapping the first two columns produced maximum logit error `87.18`.

FP32 and representative quantized files were **14,326** and **15,230 bytes**. At this size, metadata costs outweigh weight storage savings. No timing comparison is included: a small CPU quantization exercise cannot establish a Jetson speedup. These results verify this particular model, provider, fixture and software lock.

## 5. Run the checks

Portable arithmetic checks need only Python:

```bash
python3 -m unittest discover -s \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/cpu-export-quantization/labs' -p 'test_*.py' -v
```

The optional suite runs the real frameworks and retrains once:

```bash
/tmp/course-export-cpu/bin/python -m unittest discover -s \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/5. ML and AI/cpu-export-quantization/labs/optional' -v
```

It checks export at multiple shapes, disjoint calibration/test episodes, actual Q/DQ nodes, reload parity, the bad-calibration failure, sensor shift, and input rejection. Framework tests are deliberately separate from standard-library course checks.

## 6. Exercises with diagnostic answers

1. **Predict what happens if calibration repeats one input 2,560 times instead of 256.** More repetition does not add range coverage. Change only the repetition count and compare activation scales and recall.
2. **Replace the representative selection with 32 distinct training examples, seed 57.** Save a separate output directory. Compare clipping effects and decision changes; do not assume a monotonic improvement from sample count alone. Keep the held-out data out of calibration and report this as a diagnostic experiment.
3. **Why can logits change by 1.35 while every clean decision matches?** Classification depends on relative scores and the threshold. Large changes on already confident examples may leave decisions unchanged; examine near-boundary examples too.
4. **Would using the biased test observations for calibration validate shift robustness?** It would adapt to that test condition. Use a separately collected training/calibration domain, then evaluate on a fresh held-out shift set.
5. **What must be checked before TensorRT deployment?** Target-supported opset/operators and quantization forms, shapes, input preprocessing, dtype, provider/runtime versions, and task outputs. CPU execution is the numerical reference, not a TensorRT compatibility certificate.

**Completion artifact:** keep your report, versions, source hashes and one changed-calibration result. Explain why its numerical difference did or did not affect decisions. Continue with [ML deployment on Jetson](../Guide.md).
