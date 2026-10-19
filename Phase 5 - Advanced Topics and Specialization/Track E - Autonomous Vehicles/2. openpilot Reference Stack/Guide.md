# Module 2 — Read openpilot as a system of contracts

**Before:** [vehicle fundamentals and CPU lab](../1.%20Fundamentals/Guide.md), Python and basic processes/IPC. Jetson hardware is not required. **First milestone:** 3–5 hours of source reading and executed replay checks. **Outcome:** explain who owns a prediction, a controller output and a vehicle-specific command.

openpilot provides an inspectable driver-assistance system. Use its actual source to learn architecture, while keeping the distinction between reading source, running a test fixture, executing the full stack and validating a vehicle. This module performs the first two.

## 1. Freeze the source before drawing the diagram

The review used commit **`b1e88c2357f07dfc4e0872370b64d97ad606391b`**, whose `openpilot/common/version.h` declares **0.11.3**, on 18 October 2026. This is a development snapshot, not a claim that it is the latest deployed release branch. Application code is nested under `openpilot/` in this snapshot.

A read-only source checkout can be prepared without starting any daemons:

```bash
git clone --filter=blob:none --no-checkout https://github.com/commaai/openpilot.git /tmp/course-openpilot-source
git -C /tmp/course-openpilot-source checkout --detach b1e88c2357f07dfc4e0872370b64d97ad606391b
git -C /tmp/course-openpilot-source rev-parse HEAD
git -C /tmp/course-openpilot-source ls-tree HEAD tinygrad_repo opendbc_repo msgq_repo panda
```

These are source-inspection commands. A checkout of the superproject is not a full build environment and does not initialize every submodule or download model assets. If you later choose a full upstream build/replay, follow the instructions at that same revision and preserve its dependency pins.

The reviewed tinygrad submodule is `02e698160eed1751738212144a7ea2cfb160b3f1`; opendbc is `b3bd79a659dae961af169a3baac723961f720133`. The standalone [tinygrad 0.14.0 course](../3.%20tinygrad%20for%20Inference/Guide.md) is a separate learning environment. Installing its wheel into openpilot is not a supported integration recipe.

The [source-review manifest](source-review.json) records the snapshot, submodule identities, reviewed-file hashes and the explicit full-stack execution limitation.

## 2. Follow the main boundaries

| Boundary | Pinned source | What to inspect |
|---|---|---|
| Image acquisition | `openpilot/system/camerad/cameras/camera_qcom2.cc` | Frame/request identity, start/end acquisition timestamps, exposure state |
| Image preparation and inference | `openpilot/selfdrive/modeld/modeld.py` | Camera alignment, warping, model execution, action interpretation |
| Longitudinal planning | `openpilot/selfdrive/controls/plannerd.py` | Polls model output, uses vehicle/radar/control state, publishes plan |
| Controller | `openpilot/selfdrive/controls/controlsd.py` | Vehicle-specific lateral controller choice, longitudinal controller, `carControl` |
| Vehicle interface | `openpilot/selfdrive/car/card.py` | Car state, interface apply, `sendcan` publication |
| Transport and independent limits | panda/opendbc submodules | Message transport and vehicle-specific safety checks |

The [annotated flow](flow-diagram.md) turns this table into a message trace. Cereal defines typed messages; the transport and VisionIPC frame buffers are separate layers. “Shared memory” does not mean preprocessing, GPU transfer and every consumer are free of copies or ownership constraints.

## 3. A model action is not an actuator command

In this snapshot, `modeld` publishes `modelV2`, `drivingModelData` and `cameraOdometry`. Follow `get_action_from_model`: supported model paths produce desired acceleration and desired curvature, with interpretation and smoothing depending on the loaded model path. One action-output path scales a lateral-acceleration-like value by `max(1,v_ego)²` to obtain curvature. Do not infer the same raw output schema for every model artifact.

`controlsd` uses vehicle parameters and selects among angle, curvature, PID and torque lateral control paths in this revision. Its output includes `carControl`; `card` invokes the vehicle interface and publishes serialized `sendcan`. Neither a predicted path nor `modelV2.action.desiredCurvature` is already a raw steering CAN command.

**Worked dimensional check:** curvature 0.1/m and wheelbase 2.8 m correspond to `atan(0.28)≈0.273 rad` in the ideal bicycle model. That is neither 0.1 rad nor a universal normalized steering value. Real interface conversion also depends on platform dynamics and limits.

Sources: [modeld](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/selfdrive/modeld/modeld.py), [controlsd](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/selfdrive/controls/controlsd.py), [card](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/selfdrive/car/card.py).

## 4. Rates describe schedules, not age guarantees

The pinned `DT_MDL` is 0.05 s; `DT_CTRL` is 0.01 s. The service table declares `modelV2` at 20 Hz and `sendcan` at 100 Hz, while controller/interface loops use a 100 Hz ratekeeper. These are nominal schedules. They do not prove that every observation is less than 50 ms old or every command is applied in 10 ms.

A 20 Hz model result can be consumed by several 100 Hz control updates. Follow capture timestamps through preprocessing and inference, and separately inspect message publication time and validity. Include frame drops, process overruns, actuator delay and restart epochs in the analysis. [Timing constants](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/common/realtime.py), [service definitions](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/cereal/services.py).

Run the [replay lesson](Replay-Lab.md) to demonstrate why a freshly published old observation must fail a capture-age limit. The fixture's 150 ms limit is deliberately chosen for teaching; it is not asserted to be openpilot's global safety threshold.

## 5. Choose the right next experiment

Fixed-log replay holds observations fixed while changing a process or analysis. It cannot establish closed-loop driving behavior after a changed action, because the recorded future does not react to that action. A closed-loop simulator feeds commands back into its world; its conclusions still depend on sensor, actuator and environment models.

The pinned upstream bridge uses **MetaDrive**, not CARLA. Read its [simulator README](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/tools/sim/README.md) before following historical tutorials. The separate [CARLA exercise](../1.%20Fundamentals/CARLA-Lab.md) teaches sensor matching independently. Neither full simulator bridge was executed during this review.

For deeper platform work, continue with [camera timing and layout](camerad/Guide.md) and [AGNOS resource/timing analysis](agnos/Guide.md). Do not assume hardware sensor, SoC or kernel identity from a different comma device generation.

## 6. Assessment

Produce a table for one observation: acquisition identity and clock → preprocessing/model artifact → predicted units → controller → vehicle encoding → independent checks. Cite paths from the pinned snapshot. Then reproduce the replay rejection and recovery results and explain what still needs full-stack validation.

**Self-check answers:** a 100 Hz publisher does not guarantee a fresh camera frame; a successful log replay cannot prove the changed controller avoids a new obstacle; a predicted curvature is not a raw actuator angle; passing application tests does not replace independent interface/safety tests.

The [official scope](https://docs.comma.ai/concepts/safety/) describes ACC/automated lane centering with an attentive driver. Course changes remain in source inspection, replay and simulation. The handoff to Module 3 is an explicit model contract, not a vehicle deployment procedure.
