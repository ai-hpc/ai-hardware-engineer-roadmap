# Phase 5 — Track E: Autonomous Vehicles

Learn how perception, inference, timing and control fit together through a runnable vehicle model, a pinned production reference, and actual CPU neural-network experiments. The first milestones require Python and a CPU; simulator and hardware milestones are separate.

**Before:** Python, basic AI, frames and control from [robotics foundations](../Track%20D%20-%20Robotics/Robotics%20Foundations/README.md). **Roles:** perception, planning, embedded inference and vehicle systems engineers. **Hardware link:** each module produces a requirement about sensor age, memory, numerical semantics or command timing that can inform [AI chip design](../Track%20F%20-%20AI%20Chip%20Design/Guide.md).

## Start with one coherent project

1. Follow a straight lane with known state and explicit steering units in [Fundamentals](1.%20Fundamentals/Guide.md), then run its [CPU lab](1.%20Fundamentals/Lab-Guide.md).
2. Reject stale, out-of-order and wrong-frame observations in the [openpilot source/replay lesson](2.%20openpilot%20Reference%20Stack/Replay-Lab.md). Trace corresponding real process boundaries in a fixed upstream snapshot.
3. Execute tensor operations, gradients, training, checkpoint reload and changing-input JIT inference with [tinygrad 0.14.0](3.%20tinygrad%20for%20Inference/Guide.md). Connect its small output to the same named steering contract.
4. Add simulation through [CARLA's frame synchronization exercise](1.%20Fundamentals/CARLA-Lab.md), then perception and scenario variation. CARLA execution remains optional and was not available during review.

The baseline is deliberately small enough to calculate by hand. More ambitious datasets, controllers and runtime integrations build on a known result instead of hiding errors inside a large stack.

## Module map

| Module | Concrete first artifact | Initial effort |
|---|---|---|
| [1. Fundamentals](1.%20Fundamentals/Guide.md) | Bicycle-model trace, steering-rate check, delayed-observation failure | 3–5 hours |
| [2. openpilot reference](2.%20openpilot%20Reference%20Stack/Guide.md) | Pinned process/message map and timestamp fault replay | 3–5 hours |
| [3. tinygrad inference](3.%20tinygrad%20for%20Inference/Guide.md) | Executed CPU operator/model checks, checkpoint and compiler evidence | 6–10 hours |
| [4. Perception and prediction](4.%20Advanced%20Perception%20and%20Prediction/Guide.md) | Calibrated sensor/trajectory evaluation with held-out scenarios | Multi-session project |
| [5. Safety and deployment](5.%20Safety%20Standards%20and%20Deployment/Guide.md) | Traceable failure requirement and test outcome | Multi-session project |
| [6. TRACE32 debugging](6.%20Lauterbach%20TRACE32%20Debug/Guide.md) | Target timing/trace investigation, when equipment exists | Optional professional depth |

These are estimates for the named milestones, not promises of job readiness. Advanced perception, platform bring-up and safety engineering require substantially more practice.

## Version and evidence boundaries

Reviewed on **18 October 2026**:

| Component | Teaching baseline | What was actually checked |
|---|---|---|
| Vehicle/replay model | Repository's standard-library implementation | 14 portable tests and baseline/delay traces |
| openpilot | Commit `b1e88c2357f07dfc4e0872370b64d97ad606391b`, source version 0.11.3 | Source ownership, rates and submodule identities; full stack not built/run |
| tinygrad | Released package 0.14.0, hash pinned | CPU inference, gradients, training, JIT and save/reload; bounded real MNIST run |
| Standalone C runtime | Clang 18.1.3, Linux x86-64 | 12 guarded zero/tail/changing-input cases |
| CARLA | Tagged release 0.9.16 | Official API review and optional script; no server run |

openpilot's pinned tinygrad submodule differs from the standalone PyPI teaching package. Use the upstream checkout's own dependency revisions for an openpilot integration. Its current built-in simulator bridge targets **MetaDrive**; the separate CARLA lab is not advertised as a compatible openpilot bridge. [Pinned bridge documentation](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/tools/sim/README.md).

openpilot is an ACC/automated-lane-centering driver-assistance reference with an attentive-driver requirement. The exercises here use source inspection, replay, CPU models and optional local simulation. Their results do not establish road autonomy or release readiness. [Official system scope](https://docs.comma.ai/concepts/safety/).

## What to carry between modules

Preserve frame convention, capture timestamp/clock epoch, feature order, tensor dtype/layout, normalization, model/checkpoint identity, output units and command limits. A faster model that changes one of those can invalidate the controller comparison. Retain raw samples and failures, not only an average accuracy or FPS.

The completed first project includes a baseline, a deliberately failed timing case, recovery, an executed model output check and one independent transfer. Use the [shared completion rubric](../../Learning-Guide.md#completion-rubric). Subsequent modules add evidence to this artifact rather than replacing it with an untested list of tools.
