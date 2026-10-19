# Replay lab: who owns the timestamp and the command?

**Time:** 60–90 minutes. **Runtime:** Python standard library. **Executed:** the repository's small replay fixture and geometric model. **Not executed:** openpilot process replay, production models, panda/CAN or a vehicle.

## 1. Reuse a known fault instead of downloading a route first

From the repository root:

```bash
python3 "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/1. Fundamentals/labs/vehicle_lab.py" --output /tmp/vehicle-replay.json
```

Read `fault_replay` in the JSON. The fresh case has capture/ready/use times 0.95/0.98/1.00 s. It is 50 ms old, with 30 ms from capture to ready and 20 ms from ready to use. A second result published at 0.99 s but captured at 0.60 s is **400 ms old**. Publication freshness is 10 ms; it cannot substitute for acquisition age.

The exact rejection sequence is stale, future capture, wrong clock, duplicate frame, wrong frame/schema, nonfinite state and not-yet-ready result. A final fresh frame recovers. [Implementation and tests](../1.%20Fundamentals/Lab-Guide.md#3-separate-publication-freshness-from-observation-age).

**Deliberate fault:** in a temporary copy of the gate, replace `now-capture_s` with `now-ready_s`. The republished-old-observation case should now be incorrectly accepted, and the known-answer test must fail. Explain which timestamp would need to be preserved through inference to prevent this error.

## 2. Map the teaching fields to source evidence

Use the [pinned checkout](Guide.md#1-freeze-the-source-before-drawing-the-diagram). For each row, distinguish the acquisition metadata from the containing message's publication timestamp:

| Teaching field | Where to investigate in the pinned source |
|---|---|
| Frame identity | `FrameMetadata`, camera-state `frameId`, modeld frame metadata |
| Acquisition interval | Camera `timestamp_sof` and `timestamp_eof` |
| Publication/receive identity | Cereal event timing and messaging subscriber metadata |
| Model meaning | Loaded model metadata, preprocessing and `get_action_from_model` |
| Control interpretation | `modelV2.action`, controller state, `carControl.actuators` |
| Final encoding | Vehicle interface apply path and `sendcan` publisher in `card.py` |

The CPU gate's names are not a drop-in Cap'n Proto schema. Do not create a field equivalence merely because two names contain “time”. Determine units, monotonic epoch, capture definition and reset behavior from the actual producer/consumer.

For two cameras, retain both acquisition identities and the matching rule. A process-level timestamp cannot reconstruct a lost pairing. Inspect modeld's source logic for frame synchronization; the numbers in that path are implementation decisions, not a universal sensor tolerance.

## 3. Carry the model contract across compilation

The [tinygrad model-contract lesson](../3.%20tinygrad%20for%20Inference/projects/README.md) computes an illustrative linear steering prediction. It verifies input order and output shape/units against the teaching manifest. This is not an extracted openpilot model.

For a real model artifact, record image format/stride, color conversion, crop/warp, calibration, normalization, history state, recurrent inputs, tensor shapes/dtypes, output decoding and timestamp semantics. A checksum of weights alone does not identify those decisions.

**Worked mismatch:** `-0.12` road-wheel radians is about `-6.875°`. Relabeling that number as steering-wheel degrees changes both unit and mechanical quantity. An accuracy benchmark can remain numerically identical while the interface becomes wrong.

## 4. Distinguish available evidence

| Experiment | What the observations can support | What is still missing |
|---|---|---|
| This fixed fixture | Tested gate decisions and recovery bookkeeping | Actual messaging/scheduling behavior |
| Upstream process replay on a pinned route | Process outputs for fixed recorded inputs | Future scene response to changed commands |
| Closed-loop simulation | Behavior under named simulator models/scenarios | Real sensor/actuator and operational validation |
| Isolated target measurements | Target latency/resource evidence | A complete vehicle-level release argument |

If you extend to actual route replay, record route/segment identity, permitted data provenance, schema/model/source revisions, selected processes, ignored fields, comparison tolerances and observed outputs. Follow the upstream replay tooling at the pinned revision; no full-stack replay result is supplied here.

**Independent transfer:** add a restart epoch to the teaching gate. Allow frame IDs to restart only when the accepted producer epoch changes under a defined reset protocol. Show that a late message from the previous epoch is rejected while the first fresh message from the new epoch is accepted. A global `last_frame=-1` on every received message would defeat ordering entirely.

**Completion:** provide the two age calculations, the deliberate failing test, a source-supported field map and the restart transfer. State the validation level of every result. Continue with [tinygrad](../3.%20tinygrad%20for%20Inference/Guide.md).
