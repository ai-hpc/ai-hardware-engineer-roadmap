# CPU lab: follow a lane, then lose timely observations

**Access:** Python 3.10+ standard library; no simulator, dataset, GPU or vehicle. **Time:** 60–90 minutes. **Before:** [vehicle geometry and control](Guide.md). **Artifact:** two JSON traces and an explanation of one failed task.

This lab executes a kinematic bicycle model and a timestamp gate. It does not execute CARLA or openpilot. The 14 portable tests check hand calculations, geometry, time ordering, malformed model outputs and the delayed scenario.

## 1. Predict before running

The simulated vehicle starts at rear-axle position `(0,1)` m, heading zero, speed 5 m/s. Its target is the straight line `y=0`; positive `y` is left. The controller picks a point 8 m ahead along that line, limits the road-wheel angle to ±0.35 rad, and limits its rate to 0.5 rad/s. The speed target is 6 m/s. The timestep is 0.05 s and the run lasts 8 s.

The first steering request should be negative: the lane is to the vehicle's right. The maximum steering change in one tick is `0.5 × 0.05 = 0.025 rad`. These are chosen classroom parameters, not a named vehicle's actuator limits.

From the repository root:

```bash
python3 "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/1. Fundamentals/labs/vehicle_lab.py" --output /tmp/vehicle-baseline.json
python3 -m unittest discover -s "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/1. Fundamentals/labs" -p "test_*.py" -v
```

The baseline accepted 160 observations, ended with lateral error about **0.001576 m**, and had RMS lateral error **0.373641 m** over the run. The RMS includes the initial 1 m recovery. These values describe this deterministic toy model. They do not measure tire dynamics, road performance, collision avoidance or inference speed.

Open [vehicle_lab.py](labs/vehicle_lab.py). Trace `pure_pursuit → steering rate limit → bicycle_step`. The step integrates the held steering's constant-curvature arc using the distance traveled under constant acceleration. If braking reaches zero speed inside the step, the model integrates only until that stop and then stays still.

## 2. Inject a known acquisition delay

```bash
python3 "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/1. Fundamentals/labs/vehicle_lab.py" --delay .2 --output /tmp/vehicle-delayed.json
```

The age limit is 0.15 s. Four initial ticks have no ready observation; the following **156** available observations are stale. The classroom fallback requests −3 m/s² and steers toward zero at the same rate bound. The vehicle stops at `x = 5²/(2×3) = 4.166667 m`, still at `y=1`. Its lane-following task has failed even though the fallback brought speed to zero.

This fallback is a teaching policy, not a validated vehicle maneuver. It has no obstacle model, road-edge model, tire friction estimate or jerk limit. A real fallback must be designed for the operating conditions and actuator constraints.

**Guided variation:** use `--delay .1`. Predict whether the gate accepts the observations and why the first two ticks still brake. Compare accepted count, final offset, RMS offset and lateral acceleration. Explain why keeping the gate satisfied does not prove tracking stability for every speed or route.

## 3. Separate publication freshness from observation age

Every run also executes the fixed fault replay:

| Input | Expected result | Why |
|---|---|---|
| Captured 0.95 s, ready 0.98 s, used 1.00 s | accepted | Age 0.05 s |
| Captured 0.60 s, republished 0.99 s, used 1.00 s | stale | Age 0.40 s despite recent publication |
| Capture in the future | invalid time order | Same-clock chronology fails |
| Wall-clock label mixed into the simulation epoch | wrong clock | Subtraction has no defined shared epoch |
| Repeated accepted frame ID | out of order | Not a new observation |
| Camera optical coordinates supplied as world coordinates | wrong schema or frame | Geometry has changed meaning |
| NaN state | nonfinite | Cannot produce a valid geometric command |
| Result not ready until 1.10 s, read at 1.00 s | not ready | Inference/transport has not completed |
| Fresh frame after rejected frames | accepted | Rejection must not poison recovery identity |

The gate keeps the last **accepted** frame/capture pair. A large invalid frame ID must not prevent a later valid frame from recovering. Run that test and explain the state update order.

## 4. Check model outputs before interpreting them

The named teaching manifest expects FP32 input `[1,2]`: lateral error in metres and heading error in radians. The output is `[1,1]`, a road-wheel angle in radians in a forward/left/up vehicle frame. The illustrative fixed weights `[-0.08,-0.4]` produce `-0.12 rad` for `[1,0.1]`.

[tinygrad's model-contract lesson](../3.%20tinygrad%20for%20Inference/projects/README.md) executes that matrix multiply and passes each prediction through `decode_model_output`. A schema declaring steering-wheel degrees is rejected even when the numeric value looks plausible. Shape, finite values, declared units, model identity and bounded domain are separate obligations. A Python list carries no FP32 storage metadata, so the upstream runtime must also establish its tensor dtype; the decoder verifies the manifest and numeric payload, not an underlying array dtype.

**Independent transfer:** add a model whose output is curvature in `1/m`. Give it a distinct manifest and convert to road-wheel angle with `atan(wheelbase × curvature)`. Test a zero curve, a signed curve, wrong units and an out-of-domain request. Do not relabel curvature as radians.

## Completion and answers

Submit predictions, both traces, the 14-test output, and one deliberate wrong-unit rejection. Explain why the delayed run's stop is an observed fallback outcome while its lane task remains incomplete. The minimum correct explanation distinguishes acquisition age from inference duration and road-wheel angle from normalized actuator commands.

For a timestamped simulator extension continue with [CARLA](CARLA-Lab.md). For production source ownership continue with [openpilot replay](../2.%20openpilot%20Reference%20Stack/Replay-Lab.md).
