# 2. Optimize a motion without forgetting its constraints

**Before:** [dynamics](Lecture-01-Dynamics.md). **After:**
[trajectory execution](Lecture-03-Trajectory-Execution.md).
**Build:** verify a small LQR calculation, then use predictive control to reject
an impossible motion. These are two distinct fixtures, not a controller benchmark.

## Begin with an exactly stated model

Let `p` be position relative to a goal, `v` velocity, and `u` acceleration held
constant for one interval `dt`. Here positive position error means
`current_position - goal_position`. This is the opposite sign to the
`goal - current` error in the introductory proportional controller.

```text
p_next = p + dt × v + 0.5 × dt² × u
v_next = v + dt × u

x = [p, v]^T
x_next = A x + B u
A = [[1, dt], [0, 1]]       B = [[0.5 dt²], [dt]]
```

These transitions are exact for constant acceleration and this double-integrator
model. They differ from updating position with the new velocity in a
semi-implicit Euler step. For `p=2`, `v=3`, `u=4`, and `dt=0.5`, the next state is
`p=4`, `v=5`; substituting the new velocity into `p += dt*v` would give `4.5`.

The model assumes the requested acceleration is achieved immediately. A real
actuator may need force, torque, and an inner loop to approximate that command.

## What the quadratic objective means

For a horizon of `H` steps, minimize:

```text
J = sum from k=0 to H-1 of (x[k]^T Q x[k] + R u[k]²)
    + x[H]^T P_terminal x[H]
```

Use `Q=diag(10,1)`, `R=0.1`, and `P_terminal=diag(10,1)` in the LQR fixture.
The terminal term charges for where the trajectory ends. Positive `R` discourages
unlimited acceleration, but does not impose a maximum acceleration. The weights
also reconcile the different units of position, velocity, and acceleration.
Changing measurement units without changing weights changes the problem.

**Linear quadratic regulation (LQR)** solves this unconstrained linear model with
quadratic costs. Its finite-horizon policy is `u[k] = -K[k] x[k]`; the gains can
depend on how much horizon remains. The [MIT LQR notes](https://underactuated.mit.edu/lqr.html)
develop the value-function approach and explain local linearization of nonlinear
systems. A local model does not establish global control of an arbitrary robot.

## One backward step you can calculate

Assume the remaining value is `x_next^T P x_next`. Substitute `A x + B u`, collect
terms involving `u`, and differentiate:

```text
0 = 2 (R + B^T P B) u + 2 B^T P A x
K = (R + B^T P B)^-1 B^T P A
P_previous = Q + A^T P A - A^T P B K
```

Because control is scalar, the inverse here is division by one positive number.
Start at `P_terminal` and repeat backward to obtain all gains. The supplied
`finite_lqr` implements these discrete equations with small Python lists.

With `dt=0.1` and one remaining step:

```text
B = [0.005, 0.1]^T
B^T P B = 10 × 0.005² + 0.1² = 0.01025
B^T P A = [0.05, 0.105]
K = [0.05, 0.105] / 0.11025 ≈ [0.453515, 0.952381]
```

For state `[1,0]`, this one-step controller requests about `−0.4535 m/s²`.
With 40 steps available, the first gain is approximately `[7.61296,4.58493]`:
the longer horizon changes the optimization, so it need not choose the same
action. The implementation tests the independently expanded two-step objective
and checks that executed cost equals `x_initial^T P_initial x_initial`.

## Clipping LQR changes the solution

From `[1,0]`, the 40-step controller initially requests about `−7.61 m/s²`.
Clipping every command to `[-1,1]` changes the state sequence and invalidates the
claim that this is the unconstrained optimum. In the fixture, unconstrained cost
is about `60.2254`; clipped-controller cost is about `90.1938`, under the same
initial state, horizon, and cost definition.

Neither cost is a physical energy measurement. Nor does clipping establish the
optimal constrained controller: the optimization did not anticipate that its
requested actions would be clipped. Clipping can be a necessary interface limit,
but the planner should also account for available actuation.

## Predict constraints, apply one action, then replan

Model predictive control (MPC) repeatedly plans a finite horizon from the current
state, applies the first action, obtains a new state, and plans again. Depending
on the model and constraints, that optimization can be a quadratic or nonlinear
program. The [MIT trajectory-optimization notes](https://underactuated.mit.edu/trajopt.html)
describe these formulations and their computational tradeoffs.

Our reduced solver enumerates only `{-1,0,+1} m/s²` for five steps at `dt=0.2 s`:
`3^5 = 243` candidate sequences. It chooses the cheapest sequence satisfying
`−1 ≤ position ≤ 0.5 m` and `|velocity| ≤ 1 m/s`. The target is **0.45 m**, and
the initial state is **0.35 m, 0.5 m/s**. Those differ from the LQR experiment;
do not compare their numerical costs as a performance ranking.

The first selected action is `−1`. After one interval:

```text
position = 0.35 + 0.5×0.2 - 0.5×0.2² = 0.43 m
velocity = 0.5 - 0.2 = 0.3 m/s
```

Starting instead at `0.45 m, 0.5 m/s`, even maximum braking ends the first interval
at `0.53 m`: there is no feasible candidate. The function returns `None`. Treat
that as a distinct outcome that a supervisor must handle. Returning zero
acceleration would continue motion; it is not a stopping strategy.

### Constraints apply between samples too

At `p=0.49`, `v=0.2`, `u=−1`, and `dt=0.4`, both endpoint positions are `0.49 m`.
Halfway through, velocity reaches zero and position reaches `0.51 m`, crossing
the boundary. The helper checks the interior extremum at `t=−v/u` whenever it
lies in the interval. Endpoint checks would accept this invalid motion.

### A feasible plan can still fail to settle

The reference replans for ten ticks. Its final position is near the goal, but its
speed is `0.1 m/s`, failing the declared settling condition `|v|<0.05 m/s`.
With initial velocity `0.5` and increments in multiples of `0.2`, the action grid
cannot produce zero velocity: `v=0.5+0.2n` for integer `n`. This is a useful
counterexample to “MPC succeeded because it returned a plan.”

**Guided variation:** add `−0.5` and `+0.5 m/s²` to the action set. The tree grows
to `5^5=3125` sequences. Reuse the same start, limits, target, and settling test.
Record changed outcomes and search size. A larger tree has a compute cost; this
Python enumeration is a teaching method, not a scalable real-time solver.

**Independent transfer:** introduce a measured-state bias or delayed observation.
Keep the true state for evaluating constraints and the sensed state for planning.
Explain why predicted feasibility can disagree with actual motion. This lab has
no terminal invariant set, robust disturbance model, or proof of recursive
feasibility; one feasible horizon does not establish indefinite safe operation.

Run `python3 labs/control_lab.py optimal_control` from this course directory and
inspect [optimal_control.py](labs/optimal_control.py). For completion, explain one
Riccati step, one rejected candidate, and the unsuccessful settling result.
