# 3. Feedback, dynamics, and delay

**Purpose:** explain why repeatedly correcting a motion can either converge or oscillate, and connect that behavior to compute latency. **Prerequisite:** [kinematics](Lecture-02.md). **Next:** [perception and estimation](Lecture-04.md).

## Separate a target from a controller

A target says where to go. A controller compares the target with an estimate of the current state and chooses an input. The **plant** is the physical mechanism responding to that input. The plant's next state changes the observation, closing the loop.

```text
target -> controller -> actuator/plant -> sensor/estimator
              ^                              |
              +---------- feedback ----------+
```

In the simplest position model, the input is velocity: `dx/dt = u`. A proportional controller uses `u = kp * (target-x)`. Here `kp` has units of inverse seconds. This is a useful model when an inner velocity controller already tracks velocity commands well; it is not a torque-driven arm model.

With a fixed timestep `dt`, holding the velocity command between updates gives:

\[
x_{k+1}=x_k+\Delta t\,k_p(r-x_k),\qquad
e_{k+1}=(1-k_p\Delta t)e_k.
\]

The error shrinks only when `abs(1-kp*dt) < 1`, or `0 < kp*dt < 2`. For a one-metre target from zero, `kp=2 /s`, and `dt=0.1 s`, the errors are `1, 0.8, 0.64, 0.512, ...` metres. At `kp*dt=1`, this ideal model reaches the target in one step. At `kp*dt=2`, the error flips sign without shrinking. Above 2, it grows.

That inequality is specific to this plant, update rule, and zero-delay assumption. It is not a universal gain-setting rule for robots.

## Make the sensor late

If the controller sees the previous sample, the update instead uses `e[k-1]`:

\[
e_{k+1}=e_k-k_p\Delta t\,e_{k-1}.
\]

For `kp=12 /s` and `dt=0.1 s`, the instantaneous model has multiplier `-0.2`, so it converges. The one-step-delayed model has characteristic equation `z^2-z+1.2=0`; its complex roots have magnitude `sqrt(1.2)>1`, so its oscillations grow. You can also see the growth directly in the rollout without solving the polynomial.

The supplied experiment runs both for 80 steps. The delayed model reaches an absolute error of about 1664.47 m. That absurd value is a **numerical demonstration of instability in an unbounded toy**, not a plausible prediction of a real robot's travel.

Clipping velocity to 1 m/s bounds the change per 0.1 s step to 0.1 m. The supplied saturated case still cycles around the target, approximately between 0.9 and 1.1 m. A bounded command does not prove good tracking or stability of the desired equilibrium.

## Add the physical ideas that the toy leaves out

An actuator often commands force, torque, or a drive setpoint instead of ideal velocity. A one-dimensional mechanical model could be:

\[
m\ddot x+b\dot x=u+f_{external}.
\]

Mass resists acceleration, damping resists velocity, and external forces include contact. For an arm, the corresponding terms depend on joint configuration; gravity may require effort even at rest. A heavy link and a light link should not be expected to track identically with the same gains.

| Controller term | What it uses | Useful effect | Common problem |
|---|---|---|---|
| Proportional | Present error | Drives toward the target | High gain can oscillate with delay/dynamics |
| Integral | Accumulated error over time | Removes some steady disturbances | Windup during actuator saturation |
| Derivative | Rate of error change | Adds damping in suitable models | Amplifies noisy measurements; setpoint jumps can cause a kick |

**Integral windup** means error keeps accumulating while the actuator cannot deliver the requested input. When the error reverses, the accumulated term still pushes the wrong way. Anti-windup changes the integrator when the output saturates. It is not accomplished merely by clipping the final command. Real controller design also considers sample jitter, filtering, actuator limits, and the system's frequency response.

## Translate this into a hardware budget

Suppose a policy updates at 20 Hz (50 ms), the local controller at 200 Hz (5 ms), and physics at 1000 Hz (1 ms). There are ten controller updates and fifty physics steps per policy period when clocks align. A new policy output need not arrive at every controller tick; the controller needs defined behavior between outputs.

Capture-to-action age includes exposure, transport, preprocessing, inference, queueing, and command application. Model execution time alone omits several of these. A high throughput batch can make individual observations older. Record the timing boundary, missed deadlines, and the distribution of age. The local controller needs a documented hold, stop, or fallback behavior when the policy misses its contract.

## Practice and answer guidance

Run `python3 labs/foundations.py control`. The experiment always stops after 80 steps; no long simulation or training is needed.

**Guided variation:** keep `kp=12` and reduce `dt` to 0.05 s. Predict whether the no-delay multiplier shrinks. When studying delay, distinguish keeping a fixed number of delayed samples from keeping a fixed physical delay: one sample is now 50 ms rather than 100 ms. To preserve 100 ms, use two samples.

**Independent task:** compare zero, 50 ms, and 100 ms observation delays at a fixed timestep and gain. Save position/error traces, maximum absolute error, final error, and the timing parameters. Include one case that fails your tracking tolerance. Define settling as remaining inside a chosen error band for all later recorded samples; report “not settled within the horizon” when appropriate.

**Self-check:** Why does increasing publisher rate fail to cure a slow inference? The underlying state can still be old, and queues can grow. Why does a fast controller not make a slow policy irrelevant? It can track the last reference but cannot invent a fresh task-level decision. Why is “the commands are limited” insufficient? Saturation may still leave a limit cycle, collision, or unacceptable tracking error.

**Handoff:** write separate policy and controller rates, allowed state age, command limits, and timeout behavior. Use them to decide whether an inference optimization preserves closed-loop behavior.
