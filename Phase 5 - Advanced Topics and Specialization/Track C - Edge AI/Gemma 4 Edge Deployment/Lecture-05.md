# Lecture 05 — Multimodal observations to a robot-facing result

**Input:** a validated runtime with documented modality support. **Output:** a timestamped observation/result interface and replay evaluation. [Previous](Lecture-04.md) · [Next: official assistant audit](Lecture-06.md).

## 1. Trace the selected model, not a generic VLM diagram

Gemma 4 family members do not all share one fixed “SigLIP-400M plus decoder” path. The [model card](https://ai.google.dev/gemma/docs/core/model_card_4) describes dedicated encoders for some variants and an encoder-free 12B Unified variant. Inspect your model's processor and configuration; do not assume a 224×224 image or 256 visual tokens. Supported visual token budgets affect detail, prefill cost, and memory.

Record image crop/resize, aspect handling, normalization, image count, token budget, and timestamps alongside the checkpoint. A text-only load is not a multimodal validation.

## 2. Separate VLM understanding from an action policy

A scene description, pointing response, or structured tool call is still a model output that needs interpretation. A VLA additionally needs a trained or otherwise validated mapping to the robot's observation and action spaces. Text JSON alone does not establish kinematics, action units, collision checks, or control stability.

Use this course's output for a bounded perception or planning task first: identify an object in a recorded scene, propose a task label, or select among approved high-level commands. Keep a separate controller responsible for action limits, freshness, and execution.

## 3. Calculate observation age

Define `t_capture`, `t_preprocess_done`, `t_inference_done`, and `t_controller_accept` in synchronized clock domains. Observation age is `t_controller_accept - t_capture`; it includes queueing and communication.

**Teaching example:** at 30 frames/s, a queue containing six old frames contributes roughly 200 ms of age before model execution. A 15 ms kernel improvement cannot fix that backlog. Prefer a bounded latest-observation queue when task semantics permit dropping old frames; retain sequence IDs so dropped work remains visible.

## 4. Replay before connecting outputs to motion

Create episodes with instruction, timestamped images, optional proprioception, and expected high-level outcomes. Partition by episode/environment, not adjacent frames from the same recording. Include occlusion, changed illumination, absent targets, and dropped input.

A result envelope can be designed independently of the inference library:

```json
{
  "observation_id": 42,
  "capture_time_ns": 123000000,
  "valid_until_ns": 223000000,
  "frame_id": "camera_optical",
  "task": "identify_target",
  "status": "valid",
  "result": {"target_label": "blue_block"}
}
```

These values illustrate a schema, not a universal timing limit. Define clock synchronization and expiry policy in the system manifest. Validate schema and task-specific bounds; abstain on invalid or expired evidence.

## Lab and acceptance

Run fixed replay episodes at two visual token budgets. Report task quality, preprocessing/prefill/decode times, peak memory, and observation age. Demonstrate rejection of one expired or malformed result and recovery after an input gap. Hand the processor, result schema, and error cases to the [VLA deployment course](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/5.%20ML%20and%20AI/vla-deploy-jetson/Guide.md); only that later integration defines the action policy contract.
