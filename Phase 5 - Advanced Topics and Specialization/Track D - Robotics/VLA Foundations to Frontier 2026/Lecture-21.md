# Lecture 21 — VLA Inference Optimization and Edge Deployment

[Previous](Lecture-20.md) · [Course overview](README.md) · [Next](Lecture-22.md)

**Prerequisites:** Lectures 07–12, 18–20; tensor shapes, memory bandwidth, and GPU profiling basics.

**Effort:** 90 min teaching, 1–2 h runtime setup, 3–5 h profiling; CPU budget analysis takes 1 h.

**Evidence cutoff:** October 13, 2026. Numerical examples below are illustrative unless explicitly identified as executed lab results.

## Optimize the observation-to-action path

A VLA combines visual processing, language/state conditioning, and action generation. The expensive component depends on architecture, image count, history, action head, batch size, and runtime. Start with a trace of the whole path, including sensor transfer, image preprocessing, device synchronization, action decoding, and command publication. A faster matrix multiply can leave the robot's response time unchanged if camera capture dominates.

**Outcome:** choose one optimization, explain which measured cost it reduces, and prove that its physical-action and closed-loop behavior remain within a declared tolerance.

Define latency from a meaningful origin: sensor exposure, frame arrival, inference start, and command acceptance are different timestamps. Report warm and cold behavior separately, along with P50/P95/P99, missed deadlines, memory peak, and observation age at execution. CUDA kernel launches are asynchronous; host timing around a launch is not completed GPU work. Use device events or explicit synchronization in a diagnostic benchmark, then verify that measurement does not distort the intended pipeline.

## A worked latency and memory budget

Consider an illustrative single-robot pipeline: camera and transport 12 ms, preprocessing 5 ms, visual encoder 18 ms, transformer conditioning 26 ms, action generation 24 ms, and publication 5 ms. Serial latency totals 90 ms. Doubling only action-generation speed reduces it to 78 ms, a 1.15× end-to-end speedup rather than 2×. Overlap can change the critical path, so report a timeline as well as component sums.

If actions are consumed every 20 ms, a model taking 90 ms cannot generate one fresh independent action for every control tick. A chunk may sustain command availability, but its later commands use older observations. Throughput, freshness, and control frequency are different quantities. Lecture 22 derives the scheduling consequences.

A \(P\)-parameter dense weight tensor at \(b\) bits has an ideal payload of \(Pb/8\) bytes. For an illustrative 3-billion-parameter model, FP16 payload is 6 GB and four-bit payload is 1.5 GB in decimal units. Actual resident memory also includes scales, unquantized modules, activations, caches, allocator reserves, workspace, camera buffers, and simulator/ROS processes. A checkpoint's disk size is not a peak-memory benchmark.

## Match optimization to architecture

For an autoregressive action-token policy, count serial token steps and whether action dimensions can be predicted in parallel. For a diffusion or flow head, count denoising/vector-field evaluations and the cost per evaluation. Reducing a solver from ten evaluations to four can reduce compute but changes the sampler; evaluate action distribution and task success. Distillation can trade training cost for inference cost, but a distilled student requires its own validation set.

Cache only values whose dependencies remain unchanged. Language conditioning may be reusable for a fixed instruction. Visual features are invalid after a new image or crop. A transformer prefix cache is reusable only if tokenization, positions, attention mask, model weights, and prefix embeddings match the execution contract. Treat caches as model-state with explicit invalidation, not an unconditional speed switch.

Batching several robot requests can improve GPU utilization while adding queue delay. For one robot, optimize the latency path first. For a fleet service, evaluate arrival distributions and admission control; batching to improve average tokens/s may worsen P99 action age. CPU preprocessing, pinned-memory transfers, compiler specialization, graph capture, and kernel fusion are hypotheses to test against traces, not guaranteed improvements.

## Quantization is an experiment with an action contract

Quantization error propagates through an action head into physical commands. Preserve gripper decisions, rotation representation, action masks, and normalization when comparing reference and candidate. A small average action error can hide a rare sign flip at the grasp boundary. Inspect per-axis tails and event-conditioned errors before relying on an aggregate MSE.

[NVIDIA's TensorRT quantization documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) distinguishes formats, scaling schemes, and supported operators. “Four bit” does not uniquely specify a format. Verify the exact runtime, GPU, model architecture, serialized checkpoint, calibration scheme, and kernel path. Loading a compressed checkpoint does not prove native low-precision execution.

The roadmap's [DeepSeek V4.1 Flash NVFP4 course](../../Track%20G%20-%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/Part%205%20-%20DeepSeek%20V4.1%20Flash%20NVFP4/README.md) teaches checkpoint provenance and inference systems. Those methods transfer to VLA engineering; a DeepSeek language-model checkpoint does not thereby become a VLA, and its quantized weights do not substitute for a robot action head. Use the selected VLA's own artifact and deployment support matrix.

For an edge device, inspect supported precision and runtime versions directly. Do not infer NVFP4 acceleration from a Jetson product name or from the GPU vendor alone. Thermal state, power limits, memory sharing, and concurrent perception workloads belong in the report. A desktop benchmark cannot establish a robot's sustained deadline margin.

## Guided profiling experiment

Use the [specialist optimization course](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) for a deeper runtime implementation. Here, choose one supported model and one candidate change. Freeze task seeds, image preprocessing, action statistics, runtime configuration, and evaluation horizon. Collect a reference trace before applying the change.

Stage one compares physical action tensors on recorded observations, including contact transitions and unusual camera views. Stage two runs paired closed-loop episodes. Stage three adds representative concurrent robot workloads and sustained execution. Change one factor at a time before testing combinations; two individually acceptable approximations can interact.

CPU extension: derive the 90 ms example's new critical path for three changes—halve the visual encoder cost, halve action-generation cost, or overlap preprocessing with the previous action head. State assumptions about dependencies and buffer availability. The deliverable is an annotated timeline and a justified experiment order, not a claimed hardware speedup.

## Failure clinic

A smaller checkpoint that runs slower may be repeatedly dequantizing, falling back to unsupported kernels, or operating at shapes too small to amortize overhead. Higher tokens/s with worse success may indicate stale observations or a changed action cadence. Similar normalized MSE with large physical error suggests wrong scale statistics. If only first-run latency is poor, isolate engine building, compilation, allocation, and warmup rather than discarding the first sample without explanation.

## Self-check and answer guidance

1. Why is 4× smaller weight payload not 4× faster inference? The critical path includes non-weight costs and may not be memory-bound.
2. Can a cached visual prefix survive a camera update? Only if the architecture and cache scheme explicitly incorporate that update; ordinary reuse would condition on stale vision.
3. Why compare actions after denormalization? Physical units reveal task-relevant errors and scaling mistakes.
4. What supports a deployment speed claim? A pinned artifact, target hardware/runtime, measured full path, tail behavior, sustained conditions, and a passing behavioral evaluation.

---

[Previous](Lecture-20.md) · [Lab guide](Lab-Guide.md) · [Next](Lecture-22.md)
