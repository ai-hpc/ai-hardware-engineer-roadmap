# Lab: identify the first failed assumption

Use this guide after the four [course lessons](README.md). Setup takes about five
minutes when Python is installed; the default fixture executes in seconds. Spend
45–90 minutes on a variation and its explanation. The core fixture uses only the
standard library, Python 3.10+; its reference commands were tested on Python 3.12.3, Linux,
2026-10-18.

## Save one reproducible reference

From the repository root:

```bash
COURSE='Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Robot Dynamics and Task Execution'
python3 "$COURSE/labs/control_lab.py" all --output /tmp/robot-task-execution.json
python3 -m unittest discover -s "$COURSE/labs" -p 'test_*.py' -v
```

Copy the report to your experiment directory for long-term retention. The program
records its implementation hashes and Python version. Save the exact command,
repository revision, and your changed source alongside each variation.

| Reference observation | Expected interpretation |
|---|---|
| Horizontal holding torque is 9.81 Nm | The 8 Nm actuator cannot hold the specified 2 kg point mass at 0.5 m |
| Plain PD final error is about 0.8287 rad | Static error supplies missing gravity compensation |
| Compensated 12 Nm case ends within 0.001 rad | The stated exact-model controller settles in this fixture |
| Compensated 8 Nm case is still moving after 6 s | A final sample is not necessarily an equilibrium |
| LQR cost is about 60.2254, clipped-LQR cost about 90.1938 | Clipping changed the same unconstrained control problem's solution |
| MPC brakes, but the default run has `mpc_settled: false` | Feasibility and task completion differ; this action grid cannot reach zero velocity |
| Straight-path clearance is −0.3 m | Free endpoints did not establish a free swept region |
| One-metre cubic duration is 3 s | The specified velocity bound dominates the acceleration bound |
| Paired lift outcomes are true and false | Identical hand endpoints can hide different object outcomes |

These are synthetic calculations, not measurements of a robot, contact simulator,
solver latency, power, or GPU performance. Tests use hand results and numerical
tolerances rather than demanding identical last decimal digits.

## Read only the part needed for your question

| Source | Start here |
|---|---|
| [dynamics.py](labs/dynamics.py) | `Pendulum.acceleration`, then command → clamp → integrate in `pd_rollout` |
| [optimal_control.py](labs/optimal_control.py) | `step`, backward Riccati recursion, `position_range`, and candidate rejection in `finite_action_mpc` |
| [task_execution.py](labs/task_execution.py) | Closest-point projection, cubic derivatives, frame composition, and object-level success |
| [control_lab.py](labs/control_lab.py) | Fixed reference settings, replan/apply loop, and report construction |
| [tests](labs/test_control_lab.py) | Hand calculations and faults that distinguish correct from plausible behavior |

The pendulum uses numerical integration; the double integrator uses an exact
constant-acceleration transition. Do not swap those equations while retaining
expected results. The LQR and MPC examples also have different goals and initial
conditions, so their costs cannot rank one controller against the other.

## Optional: crosscheck the pendulum in MuJoCo

This extension executes a native CPU physics engine, with no rendering or contact
surfaces. First install the shared hashed environment from the
[CPU physics lab](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-03-CPU-Physics.md#2-reuse-the-saved-policy-in-an-isolated-environment).
It pins MuJoCo 3.16.0 for CPython 3.12 on Linux x86-64. Then, from the repository root:

```bash
COURSE='Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Robot Dynamics and Task Execution'
/tmp/course-mujoco-venv/bin/python -m unittest discover \
  -s "$COURSE/labs/optional" -p 'test_mujoco_pendulum.py' -v
/tmp/course-mujoco-venv/bin/python "$COURSE/labs/mujoco_pendulum.py" \
  --output /tmp/mujoco-pendulum-results.json
```

The [checker](labs/mujoco_pendulum.py) uses one hinge and a massless link with a
solid spherical bob of radius `0.01 m`. Its center is `0.5 m` from the hinge.
The simulator's joint inertia is `m l² + (2/5) m r² = 0.50008 kg m²`, compared with
the lesson's point-mass `0.5 kg m²`. Both have the same gravity torque because
their mass and center of mass agree. The seven
[actual-engine tests](labs/optional/test_mujoco_pendulum.py) check this inertia,
gravity signs, acceleration, holding torque, actuator clipping, reset behavior
and the controller trajectories. An instability warning is deliberately triggered
in one regression; it passes only if the episode is rejected.

There is a second difference: `dynamics.py` evaluates damping at the old velocity,
whereas MuJoCo's default Euler path treats joint damping implicitly. For inertia
`I`, old angle `theta`, old speed `v`, applied torque `tau`, and step `h`:

```text
explicit damping: v_next = v + h × (tau - g(theta) - b × v) / I
implicit damping: v_next = (v + h × (tau - g(theta)) / I) / (1 + h × b / I)
both:             theta_next = theta + h × v_next
```

The [retained run](labs/mujoco-pendulum-results.json) records five independent
acceleration probes and six 1,200-step engine rollouts. Three use implicit damping;
three disable `eulerdamp` and compare against the scalar equations with the bob's
finite inertia. With those assumptions matched, maximum trajectory differences
were below `1.2e-15` in angle or angular velocity. That comparison checks the
declared one-joint dynamics; it is not evidence for a physical motor or contact model.

| Default engine case, after 6 seconds | Position error | Angular velocity |
|---|---:|---:|
| Plain PD, 12 Nm limit | 0.828722 rad | approximately zero |
| Gravity compensated, 12 Nm limit | less than 0.000001 rad | approximately zero |
| Gravity compensated, 8 Nm limit | 0.735153 rad | 0.281935 rad/s |

Maximum angle differences against the original point-mass fixture were about
`0.000326`, `0.000447` and `0.003300 rad`, respectively. Do not erase these
differences by silently changing inertia or damping integration. **Predict and
check:** doubling the bob radius multiplies its extra rotational inertia by four,
while leaving the holding torque unchanged. Reducing a numerical timestep cannot
remove a difference between the physical models.

## Make a fault detectable

Choose one, preserve the failing output, then restore the reference:

1. Negate gravity torque in the plant. The horizontal acceleration and potential
   derivative checks should disagree with the model's sign convention.
2. Remove the MPC interior turning-point check. The `.49 m, .2 m/s` case with
   `−1 m/s²` for `.4 s` should be incorrectly accepted despite crossing `.5 m`.
3. Replace segment distance with endpoint distance. The straight route through
   the obstacle should wrongly appear free.
4. Define pick success using only hand height. The object-stays-on-table case
   should wrongly pass.

Explain the violated assumption, not just the name of the failing test.

## Perform an independent transfer

Create a small script inside `labs/` so local imports work. Choose a variation
from the lesson: mismatched payload/compensator mass, finer MPC action set,
quintic time scaling, or post-release placement verification. State what stays
fixed and predict the effect before running.

For example, the finer-action MPC experiment can call:

```python
from optimal_control import finite_action_mpc, step

state = (0.35, 0.5)
for tick in range(10):
    plan = finite_action_mpc(state, actions=(-1.0, -0.5, 0.0, 0.5, 1.0))
    if plan is None:
        raise RuntimeError("no feasible plan: the exercise needs an explicit response")
    state = step(state, plan["actions"][0], 0.2)
    print(tick, state)
print("settled:", abs(state[0] - 0.45) < 0.02 and abs(state[1]) < 0.05)
```

The local reference for this variation ends near `0.46 m` with velocity close to
zero and prints `settled: True`. The nominal grid ends near `0.45 m` but still
moves at `−0.1 m/s`. Apply the same position-and-speed criterion to both.

The reference search is capped at 50,000 candidate sequences and horizon eight.
The finer action grid has 3,125 sequences per solve. If the unchanged tiny
experiment takes longer than a minute, inspect the environment rather than
increasing the search blindly. No measured deadline guarantee is claimed.

**Submit:** baseline JSON, changed code and JSON or raw trace, one manual
calculation, caught-fault evidence, and a paragraph explaining model limits.
Pass with the [shared rubric](../../../Learning-Guide.md#completion-rubric),
including full correctness and reproduction marks. Continue to a simulator only
when you can explain both the successful and failed reference outcomes.
