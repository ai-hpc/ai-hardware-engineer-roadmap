# Executed tinygrad CPU projects

Use [the guide's pinned setup](../Guide.md#1-begin-with-a-function-you-can-calculate): Python 3.11+, tinygrad **0.14.0**, and Clang for the tested Linux CPU route. All commands below start at the repository root. The short numbered scripts call the readable shared [course implementation](course.py), so each lesson uses the same environment, dtype and numerical checks.

## 1. Run the finite progression

```bash
/tmp/course-tinygrad-venv/bin/python "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/3. tinygrad for Inference/projects/course.py" all --output /tmp/tinygrad-course.json
/tmp/course-tinygrad-venv/bin/python -m unittest discover -s "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/3. tinygrad for Inference/labs" -p "test_*.py" -v
```

The default makes no dataset/model download. It runs tensor operations, gradients, a tiny synthetic classifier, checkpoint reload, compiler/JIT checks, custom operations and a steering-output contract. The reviewed environment passed all **13** tests: seven mathematical references plus six actual framework tests. Ordinary Python without tinygrad still runs the seven references and explicitly skips the optional framework class.

| Script / lesson | Predict before running | Change after it passes |
|---|---|---|
| [00 intro](00_intro.py) | Affine/ReLU of `[-1,0,2]` is `[0,1,5]` | Include a negative value whose output clips |
| [01 tensor basics](01_tensor_basics.py) | Transpose `[2,3]` becomes `[3,2]` | Materialize and compare an odd-shaped transpose |
| [02 operation types](02_op_types.py) | Matmul output `[[2,11],[-3,6]]` | Change inner dimension to 5 and retain a scalar oracle |
| [03 autograd](03_autograd.py) | Gradient `[4,6]`; SGD `w_20=2(1-.8^20)` | Compare RMSNorm gradients at new smooth values |
| [04 classifier](04_mnist.py) | Initial two-class zero-logit loss is `ln(2)` | Add a hidden layer and separate validation data |
| [05 compiler](05_compiler_pipeline.py) | Realization may change materialization/kernel count | Change input values during JIT replay |
| [06 custom operations](06_custom_ops.py) | Causal first output equals first value vector | Change only a future token and test earlier outputs |
| `course.py model_contract` | `[1,.1] → -.12` road-wheel radians | Introduce and reject wrong declared units |

The numbered files support `--output`. For example, replace `course.py all` in the first command with `03_autograd.py` to run one lesson.

## 2. Run the explicit real-data extension

```bash
/tmp/course-tinygrad-venv/bin/python "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/3. tinygrad for Inference/projects/04_mnist.py" --mnist --steps 40 --output /tmp/tinygrad-mnist.json
```

This downloads/caches MNIST through the pinned loader. It uses 512 train and 256 test images, 32-example batches and a linear classifier; it is not the full dataset benchmark. The reviewed result was loss **2.302599 → 0.502533**, test accuracy **202/256**, and exact output parity after save/reload. See [the recorded run](../evidence/mnist-cpu-2026-10-18.json).

No hidden MNIST download occurs in the default synthetic exercise. `--steps` is bounded to 1–500. Small step counts may intentionally fail the training-progress expectation; such a failure needs interpretation, not a relabeled success.

## 3. Inspect compilation separately

```bash
COURSE_TINYGRAD_DEBUG=4 /tmp/course-tinygrad-venv/bin/python "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/3. tinygrad for Inference/projects/05_compiler_pipeline.py" > /tmp/tinygrad-compiler-trace.txt 2>&1
```

The runner fixes `DEV=CPU` and `BEAM=0`. Do not substitute old `CLANG=1`, `DEVICE=CLANG` or `.schedule()` examples from another release. Debug levels and tuning controls are version-specific; inspect the generated trace rather than assuming a level dumps every IR pass.

The [recorded CPU report](../evidence/tinygrad-cpu-2026-10-18.json) contains three versus one executed model kernels for this tiny expression and four changing-input JIT results. It makes no throughput or latency claim. `NOOPT` is not a universal switch that disables every mathematical rewrite.

## 4. Compile a standalone runtime with guarded buffers

```bash
python3 "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/3. tinygrad for Inference/projects/07_custom_backend.py" --output /tmp/standalone-runtime.json
```

[Project 7](07_custom_backend.py) needs Clang and the Python standard library on Linux. It creates a temporary C shared library and binds an explicit function signature. Output elements start as NaNs; guard values surround the result. This catches missing stores and simple tail overwrites while testing sizes 0,1,3,31,33,257 with two inputs each. [Recorded 12-case result](../evidence/c-runtime-2026-10-18.json).

Its filename preserves the original project link, but the implemented milestone is a standalone runtime. There is no fictitious tinygrad device registration and no accelerator performance result. **Transfer:** add a non-square row-major matmul entry point, explicit dimensions and guarded output; compare with scalar Python before adding timing. A full framework backend is the next separate integration.

## 5. Submit one evidence bundle

Include predictions, test output, package/compiler identity, the JSON report and one failing semantic variation. Explain whether a result was mathematical, actual CPU execution, compiled C execution or an unexecuted extension. A lower loss, fewer kernels, and a valid command are three different observations; connect them with their own checks.
