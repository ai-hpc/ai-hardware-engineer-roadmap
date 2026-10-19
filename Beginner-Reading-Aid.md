# Reading the first robot and AI lessons

Use this page beside [Start here](Start-Here.md). It connects the notation and
examples you meet in the first ten sessions. You can look up one section when
needed; memorizing the whole page is not a prerequisite.

## 1. Know where a command starts

A terminal has a **current working directory**: the folder relative paths start
from. `pwd` prints it; `cd` changes it. Reading a page in your browser does not
change your terminal's directory. Quotes keep a path containing spaces together.

```bash
pwd
python3 -c 'from pathlib import Path; print("Course root:", Path("mkdocs.yml").is_file())'
```

`Course root: True` means you are in the repository root. A command beginning
`python3 'Phase 3 - .../labs/policy_lab.py'` starts there. A command beginning
`python3 labs/foundations.py` starts inside the Robotics Foundations directory.
The lab's `cd` instruction moves you there and stays in effect for subsequent
commands in that terminal. Open a new terminal at the repository root when
starting a lab that asks for the root again.

In a Bash block, copy the commands without the surrounding Markdown backticks.
The first experiment's `python3 - <<'PY'` sends the lines through the closing
`PY` to Python; paste the whole block together. A separate block labelled
`python` contains Python source to save or run as directed, not Bash commands.
`text` blocks and equations illustrate calculations; they are not shell commands.

**Self-check:** your terminal is inside Robotics Foundations and Python says it
cannot open `Phase 3 - .../policy_lab.py`. Installing a package will not fix the
relative path. Return to the repository root, then rerun the command.

## 2. The same cart has two different action interfaces

**State** describes the modelled world now. **Observation** is the information
given to the decision maker; a real observation can be incomplete, noisy or old.
An **action** is the quantity requested from the environment. Its units determine
what the next state update must do.

| Lesson | Goal | Action | State update in one step |
|---|---|---|---|
| First cart / robotics control | Position `1 m` | Velocity `u`, in m/s | `position += u * dt` |
| First learned policy | Position `0 m` | Acceleration `a`, in m/s² | `velocity += a * dt`, then `position += velocity * dt` |

The first model assumes velocity changes immediately. The second retains
velocity between steps. Substituting acceleration into the first model's
velocity input is a unit error. The different goals are deliberate, so always
compute `error = goal - position` from the local task contract.

For these one-dimensional examples, choose right as positive. Positive velocity
means moving right; negative goal error means the goal is to the left. Goal
error is signed, so replacing it with an absolute distance changes the controller.

**Work one policy step:** position is `+0.5 m`, velocity `+0.2 m/s`, and goal is
zero. The observation is `[error, velocity] = [-0.5, 0.2]`. The teacher's rule
and the lab's update give:

```text
acceleration = 0.8*(-0.5) - 0.6*(0.2) = -0.52 m/s²
dt = 0.1 s
new velocity = 0.2 + (-0.52)*0.1 = 0.148 m/s
new position = 0.5 + 0.148*0.1 = 0.5148 m
```

The teacher's gains have units too: `0.8 /s²` multiplies error in metres, and
`0.6 /s` multiplies velocity in m/s. Both terms are accelerations in m/s².

Negative acceleration slows the positive motion. The cart still moves farther
from the goal during this step because its velocity is still positive. This
update uses the **new** velocity; using the old velocity would produce `0.52 m`
and a different numerical integration rule. The lesson defines which rule to use.

A **policy** maps observations to actions. A **controller** uses feedback to
choose commands; a learned policy can itself be a controller. A **planner**
chooses a route or a sequence of targets. These are roles, not necessarily three
separate programs. ROS also uses the word **action** for a long-running request
with feedback and cancellation; that messaging abstraction is distinct from one
policy output such as `-0.52 m/s²`.

## 3. Read the notation without changing its meaning

| Notation | Read it as | Small example |
|---|---|---|
| `x_k`, `x[k]` | Position at step number k | `x[0]` is the initial position |
| `Δt`, `dt` | Elapsed simulation time per update | `0.1 s` is 100 ms |
| `ẋ`, `dx/dt` | Rate of position change | Velocity in m/s |
| `x²`, `x**2` | x multiplied by itself | `(-0.5)**2 = 0.25` |
| `∣x∣`, `abs(x)` | Magnitude without the sign | `abs(-0.5) = 0.5` |
| `Σ`, `sum(...)` | Add the listed terms | `0.8*0.5 + (-0.6)*0.2 = 0.28` |
| `[N, D]` | Array shape: N rows, D columns | 8 observations with 2 features have shape `[8, 2]` |
| `@` | Matrix multiplication in supported array types | Each output is a row-column weighted sum |

Plain Python lists do not implement `@`; the standard-library labs spell out
the sums. In a model, a **feature** is one input quantity and a **parameter** is
a number learned during training. A batch holds multiple observations. An
**epoch** is one pass over the training examples; this tiny lab computes one
full-batch gradient update per epoch. Other training programs may update once
per smaller batch.

For geometry, `pi` is approximately 3.14159. A half-turn is `pi` radians and a
quarter-turn is `pi/2` radians. Convert degrees using `radians = degrees*pi/180`.
Python's `math.sin` and `math.cos` take radians.

A frame is an origin and a set of oriented axes. For a 90° counterclockwise
rotation, `(x, y)` becomes `(-y, x)`. Thus `(1.2, 0)` becomes `(0, 1.2)`; adding
the robot's world position `(1, 2)` gives `(1, 3.2)`. This is the hand check in
[frames and time](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Robotics%20Foundations/Lecture-01.md).
`Rᵀ` means transpose: swap the matrix's rows and columns. For a rotation matrix,
this is its inverse and undoes the rotation. A transpose is not the inverse of
an arbitrary matrix.

## 4. Separate learning evidence from motion evidence

A **demonstration** contains observations and example actions. An **episode**
is one run from an initial condition; neighboring samples from that run are
related. Training episodes fit parameters. Validation episodes choose among
candidates. Test episodes are held aside for the final report. A **rollout**
applies a policy's actions and records how the environment changes.

A **reset** starts a new episode with its specified initial position and velocity,
clears the previous episode's controls, and restarts its simulated clock. It does
not retrain the policy or reset the computer's clock. For a queued controller,
old observations and commands must also be discarded or rejected as belonging
to the previous episode; otherwise the next trial is not a clean restart.

Suppose action errors are `+0.1` and `-0.3 m/s²`. Their average signed error is
`-0.1 m/s²`, but opposite signs partially cancel. Mean squared error is
`(0.1²+0.3²)/2 = 0.05 (m/s²)²`; root mean squared error (RMSE) is
`sqrt(0.05) ≈ 0.224 m/s²`. RMSE measures action disagreement. Success asks whether
the cart reaches the specified position **and** velocity tolerance. Neither
quantity determines the other. A **baseline** is a simple comparison, such as
always requesting zero acceleration, evaluated on the same conditions.

The first policy dataset adds **Gaussian noise** to the teacher's action labels:
random positive and negative errors centered on zero. A standard deviation of
`0.02 m/s²` describes their spread; it is not a hard error limit. The teacher's
deterministic action and the noisy label are therefore different quantities.
A fixed **random seed** reproduces the pseudo-random sequence under the same
implementation. It makes experiments repeatable; it does not create independent
evidence by rerunning the same sequence many times.

**Training is actually executed** in the CPU policy lab, but its demonstrations
and environment are synthetic. Both statements can be true. Its scores
establish behavior of that fixture; they do not measure a camera, motor or GPU.
The [learning guide's experiment record](Learning-Guide.md#keep-an-experiment-record)
helps you preserve that distinction when moving to hardware.

## 5. Follow time from capture to use

**Rate** counts events per second. **Latency** measures elapsed time from a
stated start to a stated finish. **Observation age** measures elapsed time from
capture to the moment the observation is used. A timestamp needs a known clock.

A sensor captures at `0 ms`. Delivery takes `20 ms`, waiting takes `30 ms`, and
inference takes `120 ms`. The action becomes available at `170 ms`; the
observation age is `170 ms`, even though inference took only `120 ms`. At a
20 Hz controller, one tick is `1/20 s = 50 ms`. Three ticks have already passed.
A result valid for 250 ms after capture expires at `250 ms`, leaving 80 ms when
it arrives; receiving it does not restart its lifetime.

The [timed policy lab](Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md)
makes these durations explicit. Completing a message check, applying a command,
and achieving the task are three separate events to inspect.
