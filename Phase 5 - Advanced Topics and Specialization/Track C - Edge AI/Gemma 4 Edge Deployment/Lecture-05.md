# Lecture 05 — Multimodal observations to a robot-facing result

**Input:** a validated runtime with documented modality support. **Output:** a timestamped observation/result interface and replay evaluation. [Previous](Lecture-04.md) · [Next: official assistant audit](Lecture-06.md).

## 1. Trace the selected model, not a generic VLM diagram

Gemma 4 family members do not all share one fixed “SigLIP-400M plus decoder” path. The [model card](https://ai.google.dev/gemma/docs/core/model_card_4) describes dedicated encoders for some variants and an encoder-free 12B Unified variant. Inspect your model's processor and configuration; do not assume a 224×224 image or 256 visual tokens. Supported visual token budgets affect detail, prefill cost, and memory.

Record image crop/resize, aspect handling, normalization, image count, token budget, and timestamps alongside the checkpoint. A text-only load is not a multimodal validation.

Start from the [executed E2B metadata inspection](README.md#start-with-the-supplied-cpu-exercises): it identifies separate vision/audio modules and their parameter shapes. That is evidence about the selected architecture; it does not establish recognition quality, preprocessing correctness, or latency at either visual token budget.

## 2. Separate VLM understanding from an action policy

A scene description, pointing response, or structured tool call is still a model output that needs interpretation. A VLA additionally needs a trained or otherwise validated mapping to the robot's observation and action spaces. Text JSON alone does not establish kinematics, action units, collision checks, or control stability.

Use this course's output for a bounded perception or planning task first: identify an object in a recorded scene, propose a task label, or select among approved high-level commands. Keep a separate controller responsible for action limits, freshness, and execution.

## 3. Calculate observation age

Define `t_capture`, `t_preprocess_done`, `t_inference_done`, and `t_controller_accept` in one declared clock domain, or map them into one with a measured synchronization error. Observation age is `t_controller_accept - t_capture`; it includes queueing and communication. A sensor's hardware timestamp, a host monotonic timestamp and Unix wall time cannot be subtracted just because all are expressed in nanoseconds.

**Teaching example:** a frame captured six 30 Hz periods before inference starts is already 200 ms old. Count capture-to-now time directly: queue length alone is not an age measurement when capture or processing stalls. A 15 ms kernel improvement cannot fix that backlog. Prefer a bounded latest-observation queue when task semantics permit dropping old frames; retain sequence IDs so dropped work remains visible.

## 4. Replay before connecting outputs to motion

Create episodes with instruction, timestamped images, optional proprioception, and expected high-level outcomes. Partition by episode/environment, not adjacent frames from the same recording. Include occlusion, changed illumination, absent targets, and dropped input.

A result envelope can be designed independently of the inference library:

```json
{
  "schema": "perception_result/v1",
  "model_manifest_id": "gemma-e2b-replay/v1",
  "episode_epoch": 7,
  "clock_id": "robot_monotonic/boot-42",
  "observation_id": 42,
  "capture_time_ns": 123000000,
  "valid_until_ns": 223000000,
  "frame_id": "camera_optical",
  "task": "identify_target",
  "status": "valid",
  "result": {"target_label": "blue_block"}
}
```

These values illustrate a schema, not a universal timing limit. The example is valid only when `capture_time_ns <= now_ns < valid_until_ns`, the clock and epoch match, and the result passes its schema/task checks. At 180 ms it is 57 ms old; at 223 ms it is expired. Check again when consuming a queued result. After reset, a delayed reply with epoch 7 must fail even if observation ID 42 has been reused in epoch 8. The producer's `status: valid` cannot bypass the consumer's checks.

Bind each result to the active checkpoint/processor manifest and expected camera calibration. Define clock synchronization and expiry policy there; with bounded clock-conversion uncertainty, use the conservative upper bound on age. This envelope contains a high-level label, so it intentionally defines no joint command or actuator units.

The [existing CPU dataset/execution lab](../../Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Dataset-and-Execution-Lab.md) supplies a runnable causal-arrival, expiry, out-of-order and reset-epoch replay. Use it to learn those contracts before implementing this envelope in a service. Its synthetic fixture and dispatcher do not execute Gemma, process real images, or establish closed-loop robot success.

## Lab and acceptance

Run fixed replay episodes at two visual token budgets. Report task quality, preprocessing/prefill/decode times, peak memory, and observation age. Demonstrate rejection of one expired or malformed result and recovery after an input gap. Hand the processor, result schema, and error cases to the [VLA deployment course](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/5.%20ML%20and%20AI/vla-deploy-jetson/Guide.md); only that later integration defines the action policy contract.

The supplied CPU path finishes with metadata and the synthetic timing replay. The two-budget Gemma image experiment additionally requires a supported runtime, model/processor assets and a held-out image set; those inference results are **not run** in the supplied record. Submit the CPU findings and the remaining experiment specification separately if those assets or hardware are unavailable.
