# ML and AI on Jetson: preserve behavior while changing execution

<div class="course-identity auto-course" style="--course-accent: #db2777; --course-accent-rgb: 219, 39, 119;" markdown="1">
<div class="course-identity__icon">MAA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for ML and AI.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Updated 18 October 2026.** This module takes a trained model through export, calibration, deployment, and application validation. Start on a CPU; move the same input/output fixtures onto a supported Jetson stack when hardware is available.

**Before starting:** understand a tensor shape, a train/validation/test split, and a confusion matrix. Complete the [small PyTorch robot learning experiment](../../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md) if any of these are unfamiliar. Allow two sessions for the CPU experiment and additional sessions for target deployment.

**You will produce:** reference predictions, an exported graph, a quantized graph, a report of behavior changes, and an environment record. A successful import or fast engine is only one part of the result.

## 1. Follow one small model through the whole process

| Step | What to learn | Observable result |
|---|---|---|
| Train and reload | Fix feature meaning and select using validation data | [PyTorch CPU lab](../../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md) |
| Export and calibrate | Separate conversion errors from quantization errors | [Actual ONNX and INT8 CPU lab](cpu-export-quantization/Guide.md) |
| Match the target | Resolve module, BSP, runtime and model compatibility | [Jetson compatibility record](../../Compatibility.md) |
| Integrate | Include acquisition, preprocessing, queues and output age | [Real-time inference guide](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md) |
| Specialize | Apply task-specific coordinates, metrics and runtime choices | Lessons below |

The CPU lab uses an original synthetic stop/continue task. It is small enough to inspect every decision, but it is not evidence that a learned braking controller is safe on a physical robot.

## 2. Define the input and output contract first

For the lab, one input row is `[distance_m, speed_m_s, deceleration_m_s2, latency_s]`. The model normalizes these values using training-set statistics, then outputs two logits. A logit is an unnormalized score; apply softmax before comparing the stop probability with the validation-selected threshold.

Swapping distance and speed preserves shape `[batch,4]` but changes meaning. The actual experiment detects a maximum logit difference above 87. Shape checks alone cannot find a unit, channel-order, timestamp, or coordinate-frame error.

For a camera model, record RGB/BGR, resize versus letterbox, interpolation, padding, value range, channel layout and normalization. Save at least one raw image, its exact preprocessed tensor, and decoded outputs. For a robot policy also record state units, frame conventions, joint order and action normalization.

## 3. Quantization, with a number you can check

Quantization represents a real value by an integer and a scale:

\[
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),\qquad
\hat x=s(q-z).
\]

Here `s` is a positive scale and `z` is the integer representing zero. With signed symmetric INT8, choose `z=0` and often `q ∈ [-127,127]`. If `s=0.01` and `x=0.341`, rounding gives `q=34` and reconstruction gives `0.34`. The error is `0.001`. If `x=2`, clipping to `127` reconstructs `1.27`: the error is now `0.73`.

Without clipping, rounding error is at most half a quantization step. Clipping has no such small bound. Calibration chooses usable ranges from representative inputs; it does not learn the task again. A dataset can contain 256 samples but only one distinct observation, which is poor evidence for an activation range.

**Per-tensor** quantization uses one scale for a tensor. **Per-channel** quantization uses separate scales for output channels, reducing interference when channel magnitudes differ. It adds metadata and requires compatible kernel support. It is not guaranteed to improve a particular metric by a fixed multiplier.

**Weight-only** means activations remain in a higher-precision format. **Static activation quantization** derives ranges before inference. **Dynamic quantization** computes certain ranges during inference. **Fake quantization** rounds and reconstructs values while retaining floating-point execution, useful for studying numerical effects or training. These are different execution paths.

## 4. Read the measured calibration failure

The [CPU lab report](cpu-export-quantization/labs/measured-run-2026-10-18.json) records 576 held-out observations:

| Execution | Missed stops / 131 required stops | Recall | ONNX file bytes |
|---|---:|---:|---:|
| FP32 ONNX Runtime | 1 | 99.24% | 14,326 |
| INT8 QDQ, 256 distinct training inputs | 1 | 99.24% | 15,230 |
| INT8 QDQ, one training input repeated 256 times | 46 | 64.89% | 15,257 |

These are actual CPU results for this fixture, not Jetson benchmark predictions. The tiny quantized graph is larger because graph and scale metadata exceed the weight savings. Equivalent classification decisions also do not imply equal logits: the representative calibration's largest test-logit change is about 1.35.

When the distance sensor overestimates by 0.8 m, the FP32 model misses 34 stops and the representative INT8 model misses 33. Quantization parity did not solve sensor shift. Test conversion correctness, task behavior, and deployment conditions separately.

## 5. Export graphs and execute them deliberately

The lab uses PyTorch's `torch.onnx.export(..., dynamo=True)` path, explicit input/output names, ONNX opset 21, and a dynamic batch dimension. It then runs ONNX validation and executes with `CPUExecutionProvider`. It tests batches 1, 7 and 32, saves/reloads the graphs, and rejects invalid application inputs. See [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx.html) and [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

An exported graph is not a target-optimized executable. An ONNX symbolic dimension also does not automatically enforce every bound captured by `torch.export`; the application wrapper enforces the allowed batch range. A Q/DQ graph expresses quantization semantics, but provider support and graph partitioning determine which operations execute as integer kernels.

An unsupported operator may cause export failure, provider fallback, or deployment failure. Record which happened. For tinygrad's own compiler and current graph model, use the [actual tinygrad CPU compiler lab](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md); running ONNX in tinygrad is not the same operation as exporting an arbitrary tinygrad program to ONNX.

## 6. Cross the TensorRT version boundary correctly

Use the [current Jetson matrix](../../Compatibility.md) rather than installing a generic latest package. The checked JetPack 7.2.1 manifest includes TensorRT 10.16.2. NVIDIA's TensorRT 11.4 documentation says JetPack is not supported; its strongly typed defaults and removed APIs cannot validate 10.x snippets. [Official migration status](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html).

For an explicit Q/DQ ONNX model, preserve graph quantization scales and dtype semantics. Do not add an unrelated calibration cache and assume that it improves the same graph. Legacy implicit INT8 calibrators are a different, version-dependent route. Strongly typed TensorRT 10 networks infer and enforce tensor types; enabling builder precision flags is not the mechanism for choosing their types. [TensorRT 10 quantized types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html), [strongly typed networks](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/advanced.html#strongly-typed-networks).

A target exercise should name the model inputs, min/opt/max shape profiles, workspace budget, engine/runtime versions, required plugins and output decoder. Build an engine for the selected target and portability mode; do not promise that an arbitrary desktop engine transfers to Jetson.

Use the [named-tensor execution lesson](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Real-Time-Inference/Guide.md) for the supported 10.x Python path. It covers tensor addresses and `execute_async_v3`; this module does not mix it with old binding-index inference code.

## 7. QAT, pruning and distillation solve different problems

**Quantization-aware training (QAT)** simulates quantization during training so weights can adapt. A straight-through estimator supplies an approximate gradient through rounding. It does not guarantee that scales are learned parameters; observer- and learned-scale methods differ. Match the exported quantizer and the runtime's rounding, range and supported axes.

**Unstructured pruning** sets individual weights to zero. Dense matrix multiplication still processes a dense shape unless a compatible sparse representation and kernel are used. **Structured pruning** removes channels or blocks and changes connected shapes. For two linear layers `[8,16]` followed by `[16,4]`, retaining 8 hidden units requires selecting the same hidden indices from the first layer's outputs and the second layer's inputs, including biases and normalization parameters. A smaller shape can still map poorly to a target kernel.

**Knowledge distillation** trains a student to match a teacher as well as labels. One common loss is

\[
(1-\alpha)\operatorname{CE}(y,z_s)+\alpha T^2
\operatorname{KL}(\operatorname{softmax}(z_t/T)\;\|\;\operatorname{softmax}(z_s/T)).
\]

Freeze the teacher, use the correct KL direction and class dimension, and declare whether reductions average over examples or all tensor elements. Compare the student with a student trained on labels alone. Distillation is not proof of improved deployment latency or out-of-distribution reliability.

## 8. Measure the complete application

Separate cold load, compilation, warm inference, preprocessing and postprocessing. For a stream, measure capture-to-output age and queue depth as well as per-frame processing time. A 15 ms engine can produce 300 ms-old detections if frames accumulate.

Use correct synchronization when timing GPU work. Report median and tail latency, sample count, concurrent load, power mode, cooling and sustained duration. `1 / mean latency` is not generally the throughput of a pipelined multi-stream system. An input-power reading includes components beyond the GPU; subtract an explicitly measured idle baseline only when reporting an incremental metric.

CUDA Graph replay requires compatible shapes, memory addresses and execution paths; capture is not a promise of zero overhead. Orin Nano has no DLA. On modules that do have DLA, layers still depend on features from earlier layers: moving a backbone and head to separate threads does not remove those dependencies.

## 9. Choose an application lesson

- [Small-object detection](small-object-detection-jetson/Guide.md): pixel budgets, ignored labels, crop coordinates, merging and evaluation.
- [TAO Toolkit](tao-toolkit/Guide.md): a current versioned fine-tuning/export workflow and class/preprocessing contracts.
- [LLM optimization](llm-optimization-jetson/Guide.md): weight residency, KV cache, exact runtime support and useful measurements.
- [Experimental native LLM runtime](jetson-llm-runtime/Guide.md): source auditing and reference parity before performance claims.
- [VLA deployment](vla-deploy-jetson/Guide.md): action age, chunk scheduling, normalization and closed-loop evaluation.
- [RGB/depth and thermal preparation](non-contact-monitoring-edge/Guide.md): calibration units, synchronization, synthetic signal tests and explicit evidence limits.

## 10. Completion exercise and answers

Run the CPU export experiment and explain three failures using its artifacts: repeated calibration inputs, swapped features, and shifted distance. Then choose one application lesson and carry its input/output contract into an environment record.

1. **Can representative calibration use test labels if weights stay frozen?** No. Calibration and tuning must avoid the held-out test partition; even unlabeled test inputs adapt the deployment to its test distribution.
2. **The quantized graph is larger. Did quantization fail?** Not necessarily. Inspect stored weights, scales and graph overhead; this tiny model demonstrates the distinction.
3. **Maximum logit error is 1.35, but classifications agree. Is the model equivalent?** Only for those decisions at that threshold on those inputs. Check probabilities, margin cases, slices and the shifted condition.
4. **A 10.x Jetson engine works. Is an 11.x API upgrade safe to infer?** No. Resolve the target support and API migration separately.

Pass when another learner can reproduce the report, explain the failure cases, and distinguish CPU evidence from hardware work still pending. Use the [course completion rubric](../../../Learning-Guide.md#completion-rubric).
