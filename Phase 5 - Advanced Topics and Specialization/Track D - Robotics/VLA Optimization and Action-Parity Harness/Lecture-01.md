# 1. Optimize the policy path without changing its meaning

**Purpose:** identify what can be optimized, what changes the policy, and what
must be measured. **Before:** [course entry](README.md). **After:**
[the parity harness](Lecture-02.md). **Practice:** [CPU bridge](CPU-Action-Parity-Lab.md).

## 1. A policy tick is not every controller tick

A local controller may update at 200 Hz while a visual policy replans at 10 Hz.
The controller can track a previous target between policy results. That does
not make a 100 ms inference a 5 ms inference or establish a hard real-time bound.
Document what is held, how long it is valid, and which component reacts when
new information does not arrive.

```text
capture -> preprocessing -> pending work -> model -> action decoding/normalization
                                                        |
                      current state -> command checks -> local controller -> plant
```

A policy's action might be a joint-position target, Cartesian displacement,
velocity, acceleration, or a chunk of one of these. Seven components do not
necessarily mean seven joints. A local trajectory controller cannot interpret
a Cartesian delta merely because both are arrays of numbers.

**Worked handoff:** a chunk has 50 actions spaced 20 ms apart. Its horizon is
1 second, but executing only the first 5 actions gives 100 ms before replanning.
If inference takes 200 ms, dividing it by five produces an amortized compute
cost of 40 ms/action. The first result is still 200 ms late. Pipelining, an
existing action buffer and an explicit capture-relative lifetime are required;
amortization alone does not solve that scheduling problem.

## 2. Inspect the actual architecture and API

The following are **source-checked contracts, not checkpoint runs**. Revisions
and hashes are retained in [sources.json](labs/sources.json).

| Policy family | Checked entry point and output | Optimization consequence |
|---|---|---|
| OpenVLA original implementation | `predict_action(..., unnorm_key=..., ...)`; action count comes from the selected statistics | Preserve the tokenizer, visual preprocessing, bins, dataset statistics and mask |
| openpi π₀ / π₀.₅ family | `Policy.infer(obs, noise=...)` wraps model sampling and input/output transforms | Preserve transforms, supplied noise or RNG state, state features and chunk layout |
| GR00T | Its versioned model/embodiment and deployment configuration | Use the matching action head and supported export path; a text-server backend alone is insufficient |

OpenVLA's implementation supports a fused vision backbone; the original model
combines DINOv2 and SigLIP features. It is not correctly budgeted as one generic
400M-parameter image tower. The action dimension is read from the selected
normalization record, rather than hard-coded to seven for every embodiment.
Action dimensions share a discretization vocabulary; do not invent 256 distinct
bins times seven independent vocabularies. [Pinned OpenVLA model source](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/extern/hf/modeling_prismatic.py),
[model instructions](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/README.md).

For normalized scalar `z`, OpenVLA's masked unnormalization is
`0.5*(z+1)*(q99-q01)+q01`. With `q01=-0.04`, `q99=0.06`, and `z=0.2`, the result
is `0.02` in that action dimension's physical unit. Sending `0.2` as metres would
change the command tenfold. The statistics key, mask, action meaning and units
are therefore part of the executable model contract.

A flow action head iteratively updates an entire action tensor. Its integration
steps are not autoregressive action tokens. openpi's checked `sample_actions`
uses a configurable sampling-step count and action horizon; neither is a
universal fixed value for every exported checkpoint. Its policy wrapper also
applies output transforms. Comparing the raw model tensor with an already
unnormalized reference compares different boundaries.
[Pinned sampler](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py),
[policy wrapper](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py),
[transforms](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py).

Read timing fields at the same level of care: the checked wrapper records
`infer_ms` around the sampling call before converting results to host arrays.
An asynchronous backend may complete work during that later conversion.
Treat this field as the wrapper's stated interval, not automatically completed
GPU execution or capture-to-action latency.

## 3. Measure a budget instead of borrowing one

Measure capture-to-command age and a synchronized model-execution interval
separately. A host timer around an asynchronous launch may end before the GPU
has completed. Concurrent components can overlap, so blindly adding profiler
kernel durations can also misstate the critical path.

Use one table with input shape, camera count, dtype, device/power mode, warmup,
repetitions, queue state and timing boundary. Record preprocessing, transfers,
vision encoding, multimodal backbone, action head, transforms, and command
application. Record peak resident memory including cameras, workspace and other
processes. “KV reset between requests” does not mean KV memory or traffic is zero.

**Analytical example, not a device benchmark:** suppose a serial path takes
5 ms preprocessing, 20 ms vision, 60 ms backbone, 10 ms action head and 5 ms
transfer/dispatch. Total is 100 ms. A threefold backbone improvement reduces the
total to `5+20+20+10+5=60 ms`, a `100/60≈1.67×` improvement. It still misses a
50 ms budget. Use measured stage shares to decide which optimization matters.

## 4. Preserve arithmetic before changing precision

The CPU bridge starts with the trained policy

```text
a = w0*(error-mean0)/scale0 + w1*(velocity-mean1)/scale1 + b
```

Fold fixed normalization into weights:

```text
u0 = w0/scale0
u1 = w1/scale1
c  = b - u0*mean0 - u1*mean1
a  = u0*error + u1*velocity + c
```

For `weights=[2,-3,1]`, `means=[10,20]`, and `scales=[2,5]`, the folded parameters
are `[1,-0.6,3]`. At input `[12,25]`, both forms return zero. The operation count
changes while the real-number function stays the same. Floating-point
reassociation can change final bits, so test an explicit error tolerance.
This is an actual candidate in the supplied trained-policy lab.

CUDA Graphs and preallocated buffers can reduce scheduling/allocation overhead.
They also introduce state, pointer-lifetime, shape and stream-ordering contracts.
Graph capture does not establish numerical equivalence by itself. Repeated
identical shapes must still consume newly captured data, reset valid lengths,
and write fresh outputs. See the [executed CUDA graph exercises](../../Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/01-CUDA-Graphs.md)
for the distinction between compilation, CPU checks and GPU measurements.

## 5. Quantization is a component-specific experiment

| Change | Question to establish first | Required evidence |
|---|---|---|
| BF16 ↔ FP16 | Does the target/runtime support the intended arithmetic, and does the smaller exponent range overflow? | Finite-value checks, action error, target execution |
| INT8/INT4 weights | Does a kernel support this exact subgraph and layout? | Task-shaped calibration, scale/layout identity, decoded action and rollout comparison |
| Lower-precision vision | Which operators fall back, and which scenes are sensitive? | Embedding diagnostics plus end-to-end action/task tests |
| Lower-precision KV | What cache is actually reused by this architecture? | Valid lengths, scales, masks, reset and prefix-change tests |
| Smaller image, fewer cameras, student model | Has the observation/model class changed? | Training or adaptation as needed, fresh held-out evaluation |

The CPU lab rounds three learned weights using a symmetric signed INT8 scale,
then dequantizes them into Python floats. This measures the **numerical effect**
of a particular rounding rule. It does not execute an INT8 accelerator kernel,
implement AWQ/GPTQ, compress a VLA, or measure a speedup.

Ampere as an architecture family supports BF16 Tensor Core arithmetic; “Ampere
lacks BF16” is not a valid general rule. However, the checked TensorRT 10.x
matrix lists **BF16 and FP8 as N/A for Jetson AGX Orin, SM 8.7**. Architecture
instructions and a released runtime's supported precision are different facts.
Orin does not acquire native FP8 Tensor Core arithmetic by installing a newer
JetPack. Framework, runtime, operator and GPU support must all match. Use the maintained
[Jetson compatibility guide](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/Compatibility.md)
and the matching runtime matrix. The current generic TensorRT release is not
a substitute for JetPack's supported release.
[NVIDIA Ampere arithmetic guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
[TensorRT 10.x precision matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html).

Do not assume a vision embedding cosine threshold proves action fidelity.
Near-identical embeddings can cross an action decision boundary. Conversely,
a harmless scale change can alter an intermediate metric. Measure at several
boundaries and keep the final physical action/task checks.

## 6. Caches need an identity and a lifetime

Distinguish three caches:

- **Within-request KV:** valid for the cached prefix under the exact positions,
  attention masks, weights and current observation context.
- **Cross-request feature/prefix reuse:** valid only where those inputs and
  dependencies really remain unchanged. New image/state inputs commonly require
  recomputation; a constant instruction does not make all its contextual
  representations reusable.
- **Action chunks:** previously predicted future commands. Track capture time,
  chunk start, action spacing, executed prefix, cancellation and expiry. A
  cached action need not equal a newly replanned action at the same wall time.

A camera at 20 Hz produces a frame every 50 ms. A serial worker taking 120 ms
cannot process every frame as it arrives. In the shared timing lab, newest-only
pending work limits backlog but discards observations. Deadline checks still
apply to each whole held command interval. Repeating an old packet cannot
renew its source observation's capture time.

GPU-resident buffers can reduce copies when every participating component
supports the same ownership, format and synchronization rules. Installing
`image_transport` alone does not create a CUDA zero-copy path. Inspect the
selected NITROS/type-adaptation and camera pipeline, then measure copies and
capture-to-use age on that exact setup.

## 7. Speculation preserves a stated decoding rule

Greedy speculative verification must reproduce the target greedy decisions,
including tie rules and stop conditions. Correct stochastic speculative sampling
can preserve the target **distribution**, which does not imply identical
samples from the same seed or bit-identical intermediate logits. Replacing a
stochastic reference with argmax changes the reference policy.
[Original speculative sampling algorithm](https://proceedings.mlr.press/v202/leviathan23a.html).

A flow head is not made into an exact autoregressive verifier by calling its
intermediate actions “draft tokens.” Reducing its sampling steps or reusing a
previous chunk changes the procedure; grade that change empirically unless a
specific equivalence result applies. Measure verifier overhead and rejected
work. No decode speedup is guaranteed by this course.

## 8. Optional pretrained-policy extension

Begin with a checkpoint's own supported task evaluation and dependency versions.
For OpenVLA, the inspected source explicitly warns about mismatched Transformers
and tokenizer versions; installing the newest general stack is not a tested
migration. For GR00T, consult its [pinned deployment instructions](https://github.com/NVIDIA/Isaac-GR00T/blob/dac54b313160c201c450f97df0fd74ebd379c892/scripts/deployment/README.md)
and match the selected embodiment/runtime. Do not assume generic vLLM,
TensorRT-LLM or llama.cpp supports the entire policy merely because it supports
its language backbone.

Freeze the reference, optimize **one** component, and record actual target
measurements plus the [lecture 2](Lecture-02.md) comparisons. If the candidate
cannot meet the contract, report that result and identify the next hypothesis.
Changing the requirement after seeing a failure is a new experiment.

## Check your reasoning

1. An inference takes 200 ms and predicts five 50 ms actions. Has latency become
   40 ms? **No:** compute per action is amortized; the first observation-to-result
   delay is still 200 ms.
2. Both normalization forms give the same real-number equation. Must every
   floating-point bit match? **No:** ordering and precision can affect rounding.
3. A cache returns bit-identical old actions. Does same-observation parity pass
   automatically? **No:** the reference may replan on newer observations.
4. A quantized backbone passes action MSE. What remains? Timing, reset/state
   semantics, trajectory constraints, paired task outcomes and claim strength.

**Handoff:** save the source/target manifest, unchanged action contract, one
optimization hypothesis, and the measurements that could disprove it. Build the
[parity gate](Lecture-02.md) before expanding the candidate set.
