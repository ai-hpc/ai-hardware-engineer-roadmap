# Deploying a VLA: preserve action meaning and observation freshness

<div class="course-identity auto-course" style="--course-accent: #be123c; --course-accent-rgb: 190, 18, 60;" markdown="1">
<div class="course-identity__icon">VDOE</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for VLA Deployment on Edge GPUs — Stack Selection, Compression, and Profile-Driven Optimization.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Reviewed 18 October 2026.** A vision-language-action model consumes images, instructions and robot state, then predicts actions. Deployment must preserve the trained model's numerical behavior **and** the meaning and timing of those actions. Fast inference alone does not establish a usable robot policy.

Begin with the [VLA grounding/contract lab](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Lab-Guide.md), the [recorded-episode and dispatcher lab](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Dataset-and-Execution-Lab.md), and the [CPU ONNX export experiment](../cpu-export-quantization/Guide.md). These are runnable without model downloads or actuators. They do not claim pretrained VLA inference or a physical-robot trial.

## 1. Draw the actual data path

```text
capture timestamps + images + measured state + instruction
  → preprocessing / normalization / tokenization
  → visual-language backbone
  → action head / optional iterative sampler
  → denormalization / frame and cadence conversion
  → action validity and age checks
  → timestamped queue
  → robot controller
```

Record image order, color/resize/mask conventions, instruction template, state units and joint order. For outputs, record absolute versus delta versus velocity, coordinate frame, orientation representation, gripper semantics, training normalizer, control interval and valid action dimensions. Padded action dimensions are not extra joints.

**Worked example:** normalized displacement `0.5` with mean 0 and scale `0.02 m` means `0.01 m`. Treating it as metres commands 50 times the intended displacement. Interpreting the 0.01 m interval delta as m/s also changes motion. The [episode lab](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Dataset-and-Execution-Lab.md) executes these unit/frame/rate checks with explicit provenance.

## 2. Prediction horizon is not a latency allowance

Separate three quantities:

- `K`: actions predicted by one model invocation.
- `H`: actions the controller plans to execute before replanning.
- `f`: controller updates per second.

A full `K=50` prediction spans `50/30≈1.67 s` at 30 Hz. It does **not** follow that waiting 1.67 s for each result is acceptable. The first result has no existing action buffer; obstacles may move; and the application may execute only `H=10` actions before requesting a fresh plan.

For asynchronous scheduling, if `R` valid actions remain and a guard time `g` is reserved, a simple buffer deadline is

\[
T_{new} < R/f-g.
\]

With 10 remaining actions at 30 Hz and a 50 ms guard, the next result must be ready in about **283 ms**. This is a buffer bound, not a safety proof or a guarantee of observation freshness. The [portable timing test](../cpu-export-quantization/labs/test_contracts.py) checks this arithmetic.

Observation age is measured from capture, not inference completion. A chunk may have future execution slots whose source image is already too old. Discard expired or already passed slots using the controller contract; do not replay them immediately as a burst to catch up.

## 3. Execute the scheduling failure cases first

The [implemented dispatcher lab](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Dataset-and-Execution-Lab.md) uses synthetic episode records, common-clock integer timestamps, normalizer identity, expiry checks and reset epochs. Run it and inspect these cases:

1. Image captured before a decision but arriving after it is unavailable to that decision.
2. A 70 ms-old result may retain only future actions satisfying the age bound.
3. A newer valid result replaces overlapping unexecuted actions.
4. An old reply cannot overwrite the new request.
5. Reset invalidates queued actions and late replies from the old epoch.
6. A valid zero action and a fallback zero action retain different status metadata.

The lab's zero-delta fallback is a numerical demonstration. A real arm, mobile base or legged robot needs a controller-defined response to underrun, disconnect and invalid commands. Smoothing or temporal ensembling must not blend incompatible frames, modes or normalizers.

## 4. Read the model configuration rather than a model-zoo slogan

SmolVLA's checked LeRobot configuration defaults to `chunk_size=50`, `n_action_steps=50`, 10 sampler steps and maximum padded state/action dimension 32. These are configurable values, not immutable properties of every checkpoint. The inspected source also declares image resizing and normalization mappings. Pin both model weights and the processor/configuration. [Immutable configuration](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/policies/smolvla/configuration_smolvla.py).

The [official SmolVLA introduction](https://huggingface.co/blog/smolvla) explains the small model and asynchronous execution approach. The [openpi repository](https://github.com/Physical-Intelligence/openpi) distinguishes π0's flow head, π0-FAST's autoregressive action representation, and the π0.5 implementation it provides. Do not infer a model's sampler, memory, or support from a family name alone.

A parameter count gives a weight-memory lower bound. A 450M-parameter model at 16 bits is roughly 0.9 GB of raw weights, but image features, caches, action-head buffers, graph copies, framework overhead and the OS still matter. “Fits on 8 GB” requires a measured complete configuration, not that multiplication alone.

## 5. Choose a runtime using explicit support

| Route | Evidence to obtain |
|---|---|
| Pinned upstream PyTorch implementation | Reference preprocessing, output and task behavior |
| ONNX Runtime CUDA/TensorRT providers | Exported subgraphs, provider placement, fallback and numerical parity |
| TensorRT Edge-LLM supported VLA example | Exact model, platform, precision, builder and action-head support |
| Custom CUDA/runtime | Independent kernel/reference tests plus the complete model/action contract |

ONNX Runtime's provider priority can split a graph across TensorRT, CUDA and CPU. List the installed providers and inspect partitioning/profiling. A successful session does not prove every layer uses TensorRT. TensorRT EP has its own graph-capture setting; CUDA EP graph capture has distinct restrictions. Do not assume a single toggle captures a heterogeneous model. [TensorRT EP documentation](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html), [CUDA EP graph requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview).

Current [TensorRT Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/examples/vla/index.html) includes model-specific VLA examples, including [π0.5](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/examples/vla/pi05.html). Follow that example's exact supported stages and constraints. Its general platform matrix does not establish support for every VLA on every listed board.

The checked π0.5 example is explicitly experimental, runs four TensorRT components in FP16, and compares numerical action chunks against the **openpi** policy contract. Its LIBERO/DROID/ALOHA configurations have different state/action dimensions, camera requirements and horizons (10/15/50). It requires the experimental-model build option and an Edge-LLM attention plugin. The page explicitly does not claim task success; a Hugging Face mirror's LeRobot configuration is not automatically the same policy contract.

Datacenter TensorRT-LLM is a separate project and has [removed its old TensorRT engine backend](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html). Do not claim it is categorically unsuitable for all VLA components, or treat its old `trtllm-build` recipes as the current Edge-LLM interface.

## 6. Keep the hardware descriptions accurate

Orin Nano Super 8 GB has a stated maximum 102 GB/s memory bandwidth and 67 sparse INT8 TOPS under its applicable configuration, with shared CPU/GPU memory and no DLA. Older 40-TOPS/68-GB/s modes are different operating points. Neither TOPS nor bandwidth directly predicts VLA latency. [Official Orin specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/).

Orin uses SM87; current Thor toolchains use **SM110**. Thor is not interchangeable with a datacenter SM100 GPU or an RTX Blackwell GPU. Architecture, supported precision, memory and software differ. [NVIDIA compute-capability table](https://developer.nvidia.com/cuda/gpus).

Use the [Jetson software matrix](../../../Compatibility.md): current generic TensorRT 11.4 does not support JetPack. Edge-LLM's checked Orin support restricts model engines to FP16/INT8/INT4; a BF16 or FP8 model file does not override that runtime restriction.

## 7. Optimize a measured stage

For a flow/diffusion policy, separately time vision encoding, language/state processing, each action-head iteration, transfers and postprocessing. An unrolled ten-step sampler has different export/capture tradeoffs from a model whose iteration count changes per request.

Quantize one component at a time. Keep a higher-precision reference and evaluate near-contact or high-curvature trajectories, where small action changes may matter. BF16 has a wider exponent range than FP16 but fewer fraction bits; neither promises zero error. FP8 support is hardware/runtime-dependent, and Orin has no native FP8 Tensor Core path.

Distillation or a different action head changes the trained model. Compare a student against an appropriate training baseline and reevaluate closed-loop tasks; it is not a numerically equivalent compiler optimization. Lowering sampler steps also changes approximation error and should be treated as a model/algorithm choice.

The NXP/Hugging Face [embedded study](https://huggingface.co/blog/nxp/bringing-robotics-ai-to-embedded-platforms) reports a particular optimized SmolVLA inference around 6.15 s on its i.MX 95 setup. That is an upstream experiment with its own model, precision and scheduling, not a Jetson latency prediction or a universal acceptable control delay.

## 8. Use two levels of parity

**Numerical parity:** give both implementations the same preprocessed tensors, state, masks and explicit sampler-noise tensor. Seeding two frameworks alone does not guarantee identical random inputs. Compare each stage and the denormalized action with tolerances justified by its units and dtype. Cosine similarity alone can hide a scale error; maximum absolute error alone can overemphasize an unused padded dimension.

For example, a 0.001 normalized error with action scale 0.02 m is 0.00002 m, or 0.02 mm, before later transformations. The same normalized tolerance on a coordinate scaled by 1 rad has a different meaning. Define per-coordinate tolerances and mask padded entries.

**Task parity:** evaluate paired episodes using the same tasks, initial states and disturbances. Report per-task success, collisions/interventions, timeout definition, action-rate/age violations and uncertainty. A point estimate such as “at least 95% of baseline success” is not a statistical noninferiority result. Set a meaningful margin and use enough independent episodes to estimate it.

There is no universal acceptable 7% or 10% task-success drop. Choose criteria from the task and controller requirements before viewing the candidate's results.

## 9. Separate analytical estimates from experiments

[NVlabs/vla-perf](https://github.com/NVlabs/vla-perf) provides analytical performance modeling. It can help compare assumptions about model shapes, hardware and precision. Its output is an estimate, not a recorded Jetson kernel trace, robotics benchmark result or industry-wide standard.

Validate the dominant predicted stage with a real trace when hardware is available. Record both overestimation and underestimation: launch overhead, CPU preprocessing, provider fallback and queueing can dominate small-batch runs. A desktop GPU can still be memory-bound; hardware class alone does not establish the bottleneck.

## 10. Completion exercise and answers

Run the CPU episode/dispatcher lab, then write an adapter contract for one pinned VLA checkpoint. Include one normalization error, one stale observation and one reset/late-reply case. With target hardware, add reference tensors, provider placement, physical-unit action differences and sustained end-to-end timing.

1. **Does a 50-action chunk give 1.67 seconds to infer at 30 Hz?** Only its nominal prediction horizon has that duration; buffer state, executed horizon and age limits determine the deadline.
2. **Can a runtime with cosine similarity 1 still command the wrong motion?** Yes: scale, units or frame conversion may be wrong.
3. **Can a robot execute a late chunk faster to recover its timeline?** That changes command cadence/dynamics and needs an explicitly valid controller policy; the teaching dispatcher discards missed starts.
4. **Can action smoothing replace a trained reactive controller?** No. It changes command transitions but does not recover missing observations or prove collision avoidance.

The locally executed evidence for this lesson is the linked CPU learning, export and scheduling work. Pretrained VLA GPU inference, simulator rollouts and physical trials remain separate experiments.
