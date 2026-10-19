# CPU lab: calibration, packed bytes and behavior

Build a reference 4-bit representation, calibrate it on a real trained PyTorch
network, and evaluate the tradeoffs. You need Python 3.12, matrix multiplication
and basic PyTorch. No GPU, downloaded model, paid API or external dataset is needed.
Allow 20–40 minutes for a fresh environment and 30–60 minutes for analysis.

## 1. What this experiment proves

The model is the existing [CPU robot-stop classifier](../../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md):
a 4→32→32→2 ReLU network trained on original synthetic stopping-distance episodes.
The generator and network are reused directly, with source hashes recorded. Its
labels come from an idealized braking equation; this is numerical systems practice,
not a trained real-robot safety controller. Dataset provenance is in that lab.

The reference format stores two signed E2M1 codes per byte, low nibble first;
last-dimension blocks have 16 values with row padding; block scales use actual
`torch.float8_e4m3fn` casts and each matrix has a positive FP32 global scale.
It handles zero blocks and ties-to-even. This **teaching layout is not a vendor
checkpoint ABI**. It dequantizes weights to float32 for CPU inference; activations
and bias remain float32. It cannot demonstrate native FP4 kernel speed.

## 2. Run portable checks first

From the repository root:

```bash
python3 -m unittest discover -s "Phase 5 - Advanced Topics and Specialization/Track G - ML Systems Engineering/Hardware-Aware LLM Quantization/labs" -p 'test_*.py' -v
```

Eight tests cover row-padding/scale storage, the pinned hybrid configuration,
exact speculative correction, a biased correction shortcut, acceptance versus
quality, conditional prefix survival and a greedy-knapsack failure. The model
metadata inventory is bundled, so these tests work offline without PyTorch.

## 3. Use the pinned CPU environment

Reuse the earlier robot-stop lab environment if available. For a fresh Linux
x86-64 / CPython 3.12 environment, the full wheel hash lock is supplied:

```bash
python3 -m venv /tmp/quantization-cpu-venv
/tmp/quantization-cpu-venv/bin/python -m pip install --require-hashes -r "Phase 5 - Advanced Topics and Specialization/Track G - ML Systems Engineering/Hardware-Aware LLM Quantization/labs/requirements-linux-cpu-py312.txt"
```

It pins torch **2.14.0+cpu** and its dependencies; a fresh Torch install is a
substantial download. Other platforms need a separately validated compatible
lock. Do not install CUDA wheels for this CPU lab. In the commands below replace
`/tmp/quantization-cpu-venv` with your existing environment if reusing one.

```bash
/tmp/quantization-cpu-venv/bin/python -m unittest discover -s "Phase 5 - Advanced Topics and Specialization/Track G - ML Systems Engineering/Hardware-Aware LLM Quantization/labs/optional" -v
/tmp/quantization-cpu-venv/bin/python "Phase 5 - Advanced Topics and Specialization/Track G - ML Systems Engineering/Hardware-Aware LLM Quantization/labs/calibration_experiment.py" --output /tmp/quantization-run --epochs 40
```

The tested run reused `/tmp/course-pytorch-cpu-venv`, installed no new packages,
and passed **9 optional tests**. The full experiment trains at most 40 epochs
by default, bounded to 1–100, and uses one CPU thread.

## 4. Predict before reading the result

The 144 episodes split into 80 training, 16 calibration, 24 development and 24 test
episodes, each with 24 frames. Preprocessing fits training only; the development
set selects the best epoch. Calibration selects each matrix's clipping ratio from
`[0.5,0.65,0.8,0.9,1.0]`. The original reference supplies layer input activations.
This is a layerwise proxy; it does not optimize the complete quantized network's
joint error. All policies are fitted before test evaluation.

Compare three policies: absmax, weight-MSE selection, and calibration-output-MSE
selection. Every matrix uses real packing/unpacking; biases and normalization
remain wide. The decision threshold is fixed at 0.5. The +0.8m distance sensor
bias changes observations, while physical ground truth stays fixed.

Predictions to write down:

1. Will output-aware calibration necessarily improve end-to-end KL?
2. Will a 4-bit payload necessarily create a smaller `.pt` file than raw FP32 tensors?
3. Will a small clean-test accuracy change predict shifted stop recall?

The answer to all three is **no**. Run the experiment to see concrete cases.

## 5. Actual CPU results, 2026-10-18

The recorded run selected epoch 38. All policies reloaded with **zero** maximum
logit difference. [Complete measured report](labs/measured-run-2026-10-18.json).

| Policy | Tensor payload bytes | Clean accuracy | Clean missed stops | Shifted missed stops |
|---|---:|---:|---:|---:|
| FP32 reference | 5,160 | 99.479% | 2 | 38 |
| FP4 absmax | 1,208 | 98.785% | 5 | 40 |
| Weight-MSE selected | 1,208 | 98.785% | 5 | 40 |
| Output-MSE selected | 1,208 | 99.132% | 5 | 41 |

There are 576 test frames, including 131 true stop cases. Output-MSE chose ratio
0.9 for the middle layer and 1.0 for the others; weight MSE chose 1.0 everywhere.
Output-aware calibration improved the middle layer's calibration objective but
had **higher** end-to-end reference KL than absmax (0.00820 versus 0.00387 nats on
clean test). This is a useful limitation, not a result to hide.

The tensor payload is 1,208 bytes including padded rows, scales, biases and
normalization. Serialized `.pt` files were 5,733 or 6,133 bytes because the small
artifact's container overhead matters. Compare consistent storage definitions;
neither payload count nor this CPU dequantization establishes a GPU speedup.

## 6. Variations and answer guidance

- **Padding:** calculate the first 32×4 weight matrix. Padding creates 32 blocks:
  `32×(8+1)+4=292 bytes`, compared with 512 raw FP32 weight bytes. Applying
  `128×0.5625=72` would silently omit padded elements and the global scale.
- **Ties:** quantize 0.75 with effective scale 1. It lies between codes 0.5 and 1;
  ties-to-even chooses code 1.0. The test checks actual nibble codes, not only error.
- **Selection:** use 20 epochs as a separately declared experiment. Keep episode
  partitions fixed; explain whether changed weights change clipping decisions.
  Do not select the better epoch budget using the final test.
- **Joint error:** compare the middle-layer output loss to full-network KL.
  Explain why minimizing a local squared loss need not minimize a nonlinear
  end-to-end distribution loss. Retain the adverse shifted result.
- **Speculation:** for p=(0.8,0.2), q=(0.2,0.8), derive accepted mass (0.2,0.2)
  and corrected residual mass (0.6,0). Sampling from p after rejection would be
  biased; the portable negative test exposes it.

## Completion

Submit `report.json`, a payload calculation, the chosen clipping ratios and a
short explanation of one disagreement between metrics. Declare a missed-stop
budget before selecting a deployment policy. Retaining FP32 is a valid decision
when all quantized candidates exceed it. Explain why none of these synthetic
results establishes safe behavior on a physical robot or LLM task accuracy.
