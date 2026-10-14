# Lecture 05 — Precision and Valid Reuse

**Question:** when does moving fewer bits or recomputing fewer tensors preserve a useful robot policy?

**Prerequisites:** [Lecture 04](Lecture-04.md), quantization scales, attention dependencies, and the difference between recorded-input evaluation and closed-loop task evaluation.

**By the end:** specify an executable precision contract, separate weight compression from low-bit arithmetic, classify reuse by dependency validity, and design tests that expose a fast but behaviorally incorrect optimization.

**Deliverable:** a per-stage precision manifest and cache-validity table, with one accepted optimization and one rejected candidate supported by evidence.

## 1. A precision name is not an execution contract

Write down five different properties: storage representation, arithmetic operands, accumulator, scaling behavior, and output representation. A checkpoint filename cannot substitute for them.

| Candidate | Stored weights | Arithmetic example | Information still required |
|---|---|---|---|
| BF16 reference | BF16 | BF16 operands with a selected accumulation path | Actual accumulator, reductions, and numerical settings |
| INT8 path | Scaled integers | Integer dot products and rescaling | Activation scheme, zero points, accumulator range |
| W4A16 | Four-bit weights plus metadata | Unpack/dequantize weights for 16-bit operands | Integer or floating payload, scale layout, conversion cost |
| Integer W4A4 | Four-bit integers plus metadata | Integer products, usually a wider accumulator | Signedness, zero points, accumulator width, requantization |
| NVFP4 path | E2M1 values with specified scales | A compatible scaled floating-point operation | Block layout, activation format, hardware/kernel support |

These are candidate designs, not promises that every backend implements every row. NVIDIA's PTX ISA contains integer `.s4`/`.u4` matrix operations; therefore “four-bit computation always means dequantizing to FP16” is also incorrect. Inspect the selected instruction and target requirements. [PTX matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)

A W4A16 kernel can reduce weight traffic without receiving the peak arithmetic rate advertised for a native four-bit operation. Conversely, a native low-bit kernel can lose its advantage through activation conversion, padding, scale handling, or unsupported shapes.

## 2. Worked example: distinguish storage from multiplication

Take a toy symmetric integer weight vector:

\[
q_w=[-7,0,3,7],\qquad s_w=0.25,
\qquad \hat w=s_wq_w=[-1.75,0,0.75,1.75].
\]

Let the activation vector be \(x=[1,2,-1,0.5]\). A weight-only path dequantizes the weights and evaluates:

\[
\hat w\cdot x=-1.75-0.75+0.875=-1.625.
\]

For this carefully chosen example, activations can also be represented exactly with \(s_x=0.5\) and \(q_x=[2,4,-2,1]\):

\[
q_w\cdot q_x=-14-6+7=-13,
\qquad s_ws_x(-13)=-1.625.
\]

The second computation uses integer products. Equal answers here do not establish equivalence on general inputs: rounding and clipping occur when values are not exactly representable.

For a length-\(K\) integer dot product with magnitude bounds \(Q_w,Q_x\), a conservative magnitude bound is \(KQ_wQ_x\), before bias and zero-point corrections. Check this against the accumulator range. Output saturation does not repair an overflow that already occurred during accumulation.

The compiler must carry scales and zero points through transformations rather than treating packed integers as the original real values. MLIR's quantization representation distinguishes stored values from expressed values; the final lowering must also define rounding and clipping behavior. [MLIR quant dialect](https://mlir.llvm.org/docs/Dialects/QuantDialect/)

## 3. NVFP4 needs a complete layout definition

For the 1D representation studied in Lecture 04, reconstruct a value using its E2M1 payload, E4M3 block scale, and FP32 global scale. The payload is floating point; reinterpreting its bit pattern as a signed INT4 integer changes the value. The scale format and block grouping also differ from other FP4 schemes. [NVIDIA NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)

Require the manifest to identify:

```text
tensor name and logical shape
payload encoding and packing order
scale dtype, group axes, group size, and physical scale layout
global scale convention and tensor partitioning
padding, transpose, and alignment rules
rounding, clipping, accumulator, and output conversion
checkpoint revision, converter revision, and kernel revision
```

A hardware unit accepting E2M1 payloads does not automatically accept an arbitrary “FP4 checkpoint.” A loader must map that checkpoint's semantics and layout to the unit's supported contract.

Keep one higher-precision execution path as the numerical reference. It need not occupy every accelerator tile: a prototype may use a host reference for unsupported operations, while reporting its cost separately from an eventual hardware proposal.

## 4. Select precision by stage and by evidence

Use the traced stages from Lecture 02: image encoder, multimodal backbone, repeated action block, and action output adapter where present. Change one region first so that regressions remain attributable.

For each candidate, record:

1. Calibration data provenance and separation from evaluation episodes.
2. Error distributions, clipping counts, and problematic channels or steps.
3. Actual supported kernels and fallback operations.
4. Total bytes, conversion work, latency, and measured energy if available.
5. Closed-loop task success and recovery behavior under the same protocol.

Include different action-generation steps and difficult observations in calibration. A scale fitted only to ordinary early-step activations may not represent later steps or rare inputs. This is a hypothesis to test for the selected policy, not a universal claim that all action heads require step-specific scales.

Choose the smallest useful format set that meets the quality and system budgets. Additional formats increase datapath, compiler, verification, and maintenance work. An advertised peak rate is insufficient evidence to include a format.

### October 2026 research: turn each result into a design question

The [research ledger](Sources-Models.md#accelerator-and-quantization-research) preserves the primary sources, versions, access checks, and reproduction limits for these methods:

| Method | Evidence relevant to precision | Question for the accelerator/compiler |
|---|---|---|
| [QuantVLA](Sources-Models.md#quantvla-februaryapril-2026) | Selective post-training quantization with attention and action-output calibration; public GR00T W4A8 path | Where do rescaling and retained floating-point operations execute, and what transfers do they introduce? |
| [FoldQuantVLA](Sources-Models.md#foldquantvla-21-september-2026) | Folded transforms and native integer projections; selective INT8 retention and floating-point non-projection work | Can lowering preserve transformed numerics while avoiding conversion/fallback costs? Include these costs in the full-policy timing. |
| [CHASE-VLA](Sources-Models.md#chase-vla-2-october-2026) | Prior-chunk and denoising-step-conditioned action-expert scales, evaluated on π₀.₅ and GR00T N1.6 | What state and scale metadata must remain available, and what happens on initialization, reset, or a missing chunk? N1.7 support needs separate evidence. |

These papers motivate implementation experiments. They do not jointly establish one universally supported low-bit VLA format or the quality of your proposed chip. Select one executable method, reconstruct its supported layer set, and compare it with the same graph and runtime in higher precision before attributing a benefit to hardware.

## 5. Reuse has a validity rule and a scope

The following categories are deliberately not three mutually exclusive hardware cache types:

| Description | Meaning | Proof or evaluation required |
|---|---|---|
| Dependency-exact reuse | The retained tensor would be unchanged under the declared numerical execution semantics | All relevant inputs, parameters, and execution conditions remain valid |
| Within-inference reuse | Retention occurs across steps of one evaluation | Determine whether this particular retention is exact or approximate |
| Approximate temporal reuse | Earlier observations supply values for a newer evaluation | Explicit approximation policy and closed-loop quality evaluation |

“Within one inference” describes the scope. It does not prove exactness: a tensor conditioned on the denoising step can change even when the observation remains fixed.

Cache entries should be associated with the dependency versions they actually require: model, preprocessing, observation, instruction, robot state, mask/position convention, and sometimes action step or random state. A dependency graph determines the relevant subset. Requiring an unrelated version unnecessarily reduces hit rate; omitting a real dependency corrupts results.

Version counters or immutable request objects can track these relationships. Computing a large content hash on every access has its own cost and is not required when the runtime already controls identity and mutation.

## 6. Worked example: valid conditioning reuse

Suppose a cross-attention block at step \(j\) computes:

\[
Q_j=X_jW_Q,\quad K=CW_K,\quad V=CW_V,
\quad A_j=\operatorname{softmax}(Q_jK^\top/\sqrt d) V.
\]

During one evaluation, assume \(C,W_K,W_V\), their layouts, and their numerical execution semantics remain fixed. Then \(K,V\) can be computed once and retained across steps. \(Q_j\) and the attention output still change with \(X_j\).

Now replace \(K\) with \(K_j=\operatorname{Norm}(C;\gamma_j,\beta_j)W_K\). If the normalization parameters depend on step \(j\), the previous key tensor is no longer generally valid. Retaining the unnormalized \(C\) can still be exact; retaining \(K_j\) requires a new argument.

Across camera frames, a changed image can change \(C\). Even identical pixels do not prove the whole policy input is unchanged: robot state, instruction, position indices, or temporal context may differ.

An autoregressive causal prefix can admit exact prefix reuse under the same prefix, position, mask, and model semantics. A jointly attended or bidirectional visual representation does not automatically inherit that rule. Derive validity from the actual graph before designing a cache instruction.

## 7. Approximate reuse is an algorithm change

An approximation might reuse visual features until a change detector triggers refresh. Its apparent speedup must include detector cost, refresh cost, retained-state memory, and cases in which it misses a relevant change.

Test changes that are small in pixels but important to the task: an object slipping, a gripper nearing contact, an instruction changing, or an obstacle entering the action path. Evaluate in simulation before transferring an experimental policy to moving hardware.

For every approximate cache policy, publish a maximum retention rule, invalidation triggers, fallbacks, and the quality threshold. “The scene looks similar” is not an exact dependency proof.

Use [Lecture 08](Lecture-08.md) to compare an optimization against the unmodified policy at matched task conditions. Distinguish numerical parity, offline action agreement, and actual task success; they answer different questions.

## 8. Exercise: make a wrong cache fail visibly

Build a small repeated cross-attention fixture, or use a matching block from the pinned policy. Keep a reference with recomputation on every step.

1. Cache only dependency-invariant conditioning projections and compare outputs.
2. Add step-dependent normalization; demonstrate why the previous cache key fails.
3. Change the observation, robot state, and instruction separately; record which tensors invalidate.
4. Compare a declared W4A16 path against one other precision candidate.
5. Record conversion cost and full retained-state storage, including scale tensors.

**Negative case:** intentionally omit one real dependency from the cache key. The test must detect the stale result, not merely report a higher hit rate. For stochastic blocks, control random inputs when testing parity and evaluate outcome distributions separately.

**Acceptance:** deliver the precision manifest, dependency graph, cache table, tests, and a decision log. Keep approximate reuse behind a separate evaluation result; do not label it exact after passing only a few samples.

## 9. Answered self-checks

**1. Does a four-bit checkpoint prove native four-bit multiplication?** No. It may use weight-only compression and dequantize into wider operands. The kernel and instruction path establish the executed arithmetic.

**2. Is within-inference reuse always exact?** No. Inputs can remain fixed while step-dependent conditioning, random state, or intermediate action state changes the retained tensor.

**3. Is small average action error enough to approve temporal reuse?** No. Average error can hide rare task-critical failures, and errors alter later observations in closed-loop execution. Evaluate behavior under the declared task protocol.

## Sources and next step

- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/): distinguish supported integer and floating-point instructions and their target requirements.
- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html): inspect concrete runtime restrictions rather than extrapolating from a format name.
- [MLIR quantization types](https://mlir.llvm.org/docs/Dialects/QuantDialect/): retain numerical interpretation through the compiler.
- [NVFP4 recipes](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html): verify the exact encoding and scaling variation selected.

Continue to [Lecture 06 — Compiler, Runtime ABI, and SoC Integration](Lecture-06.md). Numerical and dependency contracts become executable checks at the compiler/runtime boundary.
