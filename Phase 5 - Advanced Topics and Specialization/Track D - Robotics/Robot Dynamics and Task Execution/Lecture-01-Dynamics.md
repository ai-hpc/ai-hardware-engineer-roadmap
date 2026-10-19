# 1. A reachable pose still needs enough torque

**Before:** [kinematics and feedback](../Robotics%20Foundations/README.md).
**After:** [LQR and MPC](Lecture-02-LQR-and-MPC.md).
**Build:** predict a joint's holding torque, then compare proportional-derivative
control with gravity compensation and a saturated actuator.

## Start with one joint and a sign convention

Our joint rotates a massless rigid rod of length `l` carrying a point mass `m`.
Angle `theta=0` points downward; positive angle rotates away from that direction.
The motor applies signed joint torque `tau`. Viscous damping is `b`.

```text
inertia I = m l²
gravity term g(theta) = m g l sin(theta)

I × angular_acceleration + b × angular_velocity + g(theta) = tau
```

The gravity term appears on the left, so positive `theta` near zero produces a
negative acceleration with no motor input. Check this sign before tuning gains.
See the primary [MIT pendulum notes](https://underactuated.mit.edu/pend.html) for
this equation and its energy interpretation.

For `m=2 kg`, `l=0.5 m`, and `g=9.81 m/s²`, inertia is `0.5 kg m²`. At a horizontal
pose, `theta=pi/2`, holding torque is `9.81 Nm`. Releasing it from rest with zero
motor torque gives initial acceleration `−9.81/0.5 = −19.62 rad/s²`.

A distributed rod has a different inertia and center of mass. For a uniform rod
pivoted at one end, use `I=m l²/3` and gravity lever arm `l/2`; do not silently
reuse the point-mass model's coefficients. Gear and motor inertia can add further
reflected inertia.

## Position, velocity, acceleration, and torque are different commands

Kinematics tells you which joint angles produce a pose. Dynamics tells you which
torques can produce a motion from the current state. For a multi-joint robot, the
usual rigid-body equation collects these effects as:

```text
M(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau
```

`M` couples joint accelerations; `C q_dot` collects velocity-dependent terms;
`g` is gravity. Contact forces add generalized forces, commonly expressed with a
Jacobian transpose under a declared wrench/frame convention. The scalar fixture
isolates one part of this larger model.

A position-controlled drive may run its own faster inner loop. Sending it a value
in radians is different from sending a torque in newton-metres. Record what the
drive accepts and where saturation is enforced. A policy trained to output
normalized actions needs that exact mapping before execution.

## Why proportional-derivative control can stop short

Use a constant target and the command:

```text
requested_tau = Kp × (target - theta) - Kd × theta_dot
```

At rest, damping contributes zero. The controller needs a nonzero error to
produce the gravity-balancing torque. For our horizontal target and `Kp=8`, the
static equation is:

```text
8 × (pi/2 - theta) = 9.81 × sin(theta)
```

The CPU fixture settles near `theta=0.7421 rad`, leaving about `0.8287 rad` of
error. This result is consistent with the model; it is not an integration crash.
Increasing `Kp` reduces this bias but changes transients, effort, and sensitivity
to delay. An integral term can address constant bias, but needs saturation-aware
design and does not create unavailable motor torque.

Add a gravity estimate instead:

```text
requested_tau = g_hat(theta) + Kp × (target - theta) - Kd × theta_dot
```

With exact gravity compensation and no saturation, the error dynamics have
inertia `I`, damping `b+Kd`, and stiffness `Kp`. For the fixture, `I=0.5`, `Kp=8`,
`b=0.2`, and `Kd=3.8`: `b+Kd=4=2 sqrt(I Kp)`, the critical-damping value of this
continuous linear error model. The simulation approximates it at `dt=0.005 s`.

**Predict:** if the mass estimate is wrong, does compensation still eliminate
steady error? No. The uncompensated gravity difference must be supplied by
feedback or another estimator. The reference uses the true mass, so it isolates
the compensation mechanism rather than demonstrating parameter robustness.

## An actuator limit changes what is possible

The implementation clamps the requested torque before updating the state.
At the horizontal target, an 8 Nm limit gives:

```text
angular_acceleration = (8 - 9.81) / 0.5 = -3.62 rad/s²
```

It cannot remain at that target from rest. More controller compute, more training,
or a larger proportional gain cannot change this static requirement. Change the
load, leverage, target, or available actuation. The saturated six-second run still
has about `0.29 rad/s` angular velocity; do not describe its final angle as a
settled equilibrium.

For a simple transmission estimate, let reduction `N=50`, efficiency `eta=0.8`,
and motor torque constant `Kt=0.1 Nm/A`:

```text
joint_tau ≈ eta × N × Kt × current
holding_current ≈ 9.81 / (0.8 × 50 × 0.1) = 2.4525 A
```

This illustrative relation omits gearbox friction and detailed drive behavior.
At a stationary joint, mechanical output power `tau × angular_velocity` is zero,
while motor current can still produce winding heat. A short peak-torque rating
therefore cannot establish continuous holding capability.

## Inspect the experiment

From this course directory:

```bash
python3 labs/control_lab.py dynamics
```

Read [dynamics.py](labs/dynamics.py): `Pendulum.acceleration` computes the physical
response, while `pd_rollout` computes the command, clamps it, and integrates the
response. Semi-implicit Euler updates velocity before angle. It is a numerical
approximation; a smaller timestep is a convergence check, not a measurement of a
faster motor or controller.

The optional [MuJoCo crosscheck](Lab-Guide.md#optional-crosscheck-the-pendulum-in-mujoco)
runs this controller against an actual native physics engine. It corroborates the
gravity sign, `9.81 Nm` horizontal holding requirement, PD bias and insufficient
actuator torque. Its small spherical bob adds `0.00008 kg m²` of rotational
inertia to the point-mass value, and its default Euler method integrates joint
damping implicitly. Those are stated model differences, so the trajectories
should be close rather than identical. A separate comparison matches both
assumptions and checks agreement to numerical roundoff. Read MuJoCo's primary
[geometry mass/inertia reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-mass)
and [Euler damping option](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-flag-eulerdamp)
before translating the equations into another engine.

**Guided variation:** halve `dt` and double `steps` in a separate driver. Preserve
the same six-second duration, target, and gains. Compare final angle, peak torque,
and the transient. Then change the actuator limit as a separate experiment.

**Independent transfer:** add a known constant payload to the physical model while
keeping the compensator's original mass. Predict the direction of error and save
both the true and assumed mass. Do not feed the new true mass into both models and
call the result robustness to model mismatch.

**Exit question:** the arm holds the pose for two seconds in this model. What is
still missing for a drive choice? **Answer:** continuous torque/current/thermal
limits, transmission losses and compliance, motor/drive dynamics, sensing and
timing, the real load distribution, and the task's full motion profile. The small
model supplies an auditable starting calculation.
