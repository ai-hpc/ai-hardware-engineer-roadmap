# Robotics foundations lab guide

Build a small evidence bundle that explains a robot's geometry, timing, perception, and planning failures before training a policy. These numerical exercises use Python's standard library and finish in seconds on a CPU. Budget 10–20 minutes for setup and 45–90 minutes per lesson's variation and analysis.

## Run and save the baseline

From the repository root, change into this course directory once. Skip the
`cd` line if you already did it in the course README:

```bash
cd 'Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Robotics Foundations'
python3 --version
python3 labs/foundations.py all --output foundations-results.json
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

**Tested here:** Python 3.12.3 on Linux, 2026-10-18; all 23 tests passed. Python 3.10+ is required by the source syntax. The output file is created in your current directory; preserve it in your own experiment folder. The program explicitly labels results as **synthetic numerical fixtures**. It does not connect to a robot, install software, download models, or run ROS.

Read [the source](labs/foundations.py) alongside each explanation. The public functions are intentionally small enough to inspect. Create a separate variation script that imports them; preserve the baseline so you can compare it with your changes.

If ROS 2 Jazzy is already installed, the [separate ROS integration lab](ROS2-Lab.md) runs real messages and TF with a controlled stale-data test. It is not included in the dependency-free unit suite. Its checked-in evidence identifies the exact packages and local execution limits.

## Predict, run, then explain

| Experiment | Expected baseline result | Meaningful variation | Fault or misconception caught |
|---|---|---|---|
| `frames` | World point `(1,3.2)` m; inverse returns `(1,0)` m | Change both transform rotations | Reversed transform order, wrong inverse translation, stale/future time |
| `kinematics` | Two IK branches both reach `(1,1)` m | Unequal links; target inside inner radius | Unreachable target silently clipped; q2 mistaken for absolute angle |
| `control` | Instantaneous case converges; delayed case diverges | Hold physical delay fixed while changing timestep | Confusing throughput with fresh feedback; saturation mistaken for convergence |
| `perception` | Pixel `(370,265)`; posterior `(1.24 m, 0.008 m²)` | Scale image and intrinsics together | Wrong image calibration; invalid depth; variance confused with deviation |
| `planning` | Point path has four steps; larger footprint has no route | Widen door or add a detour | Endpoints or reference point checked without the footprint |

Run an individual experiment with, for example:

```bash
python3 labs/foundations.py control --output control-results.json
```

This chooses the reported section of the small fixed fixture; it is not a
configurable simulator. Import the functions in your own script for variations.
Save this example as **`labs/delay_variation.py`** inside this course directory:

```python
from foundations import control_rollout

for dt, delay_steps in [(0.1, 1), (0.05, 2), (0.025, 4)]:
    values = control_rollout(kp=4.0, dt=dt, steps=round(8.0 / dt), delay=delay_steps)
    print({"dt_s": dt, "delay_s": dt * delay_steps,
           "horizon_s": dt * (len(values) - 1),
           "final_error_m": 1.0 - values[-1],
           "peak_error_m": max(abs(1.0 - x) for x in values)})
```

Run the file from this course directory:

```bash
python3 labs/delay_variation.py
```

Python finds `foundations.py` beside your new script. A file saved elsewhere
does not automatically share that import path; keep the two files together for
this exercise. The `round(8.0 / dt)` expression chooses enough steps to cover
eight simulated seconds at each timestep.

The delay and horizon are fixed in seconds while the timestep changes. State that assumption in your report. Do not conclude that a timestep change improved robustness if it also silently reduced physical delay or shortened the evaluation horizon.

## Know what each test establishes

The suite uses hand-computed transforms and poses, round trips, a central-difference Jacobian, an analytical no-delay control solution, scalar uncertainty arithmetic, and breadth-first search as an independent shortest-path oracle. Failure cases include stale/future timestamps, invalid depth/variance, unreachable IK, blocked starts, no path, and footprint overlap with the map boundary.

Passing these tests establishes properties of the supplied mathematical models. It cannot establish real camera calibration, wheel-slip behavior, 3D collision avoidance, actuator stability, ROS discovery, delivery deadlines, or simulation fidelity. Preserve the tests when replacing a toy component, then add checks for the new component's actual failure modes.

## Final artifact

Create a directory containing:

1. `foundations-results.json` and your test output, with Python version and repository revision.
2. One small variation script per numerical lesson, saved inputs, expected outcome, actual result, and explanation of any disagreement.
3. A labelled frame diagram, action/controller interface table, and timing budget from [lesson 6](Lecture-06.md).
4. A fault matrix showing how your checks detect at least one frame error, one stale observation, one unreachable target, and one collision-related planning failure.
5. A short handoff stating which claims are analytical, synthetic, locally measured, or not run. If you completed the ROS extension, keep its software versions and traces separate from the CPU fixtures.

For an independent transfer assessment, change the arm lengths, camera image size, observation delay, and doorway width. Predict which outcomes change before rerunning. Pass when the new cases are correct, reproducible, and explained under the [shared rubric](../../../Learning-Guide.md#completion-rubric). A valid rejection is a successful engineering outcome when the requested motion violates the contract.

Continue with [Advanced ROS](../Advanced%20Robot%20Operating%20System/Lecture-01.md) to integrate the components, [Isaac engineering labs](../Isaac%20Sim%20and%20Isaac%20Lab/Lab-Guide.md) to validate a simulator task, or [VLA action semantics](../VLA%20Foundations%20to%20Frontier%202026/Lecture-06.md) to inspect a learned policy's interface.
