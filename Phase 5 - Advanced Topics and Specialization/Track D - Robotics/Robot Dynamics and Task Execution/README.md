# Robot dynamics and task execution

A reachable pose can still require too much torque. A collision-free path can
still be impossible to track. A gripper can reach its commanded pose while the
object stays on the table. Learn to expose these gaps with small calculations and
runnable models before adding a learned policy.

**Prerequisites:** [Robotics Foundations](../Robotics%20Foundations/README.md),
especially frames, kinematics, control, and planning; Python functions and lists.
The optimal-control lesson introduces its matrix notation. **Layers:** L1 workload
and task contract, with L3–L5 consequences for scheduling, control execution, and
compute budgets. **Roles:** robotics, robot learning, embedded control, and
physical-AI systems engineering.

**Access:** Python 3.10+ and a CPU; standard library only. No simulator, GPU,
external model, or robot is required. The calculations describe specified
mathematical fixtures. Hardware control and contact simulation are later stages.

## What to learn and build

| Lesson | New idea beyond the foundations | Artifact | Study time |
|---|---|---|---|
| [1. Torque, gravity, and actuator limits](Lecture-01-Dynamics.md) | Motion depends on inertia and available effort | Torque/current calculation and three controller traces | 60–90 min |
| [2. LQR and constrained predictive control](Lecture-02-LQR-and-MPC.md) | Optimize a model, then distinguish costs from hard constraints | Riccati check, feasible plan, infeasible case, and failed settling result | 90–120 min |
| [3. From a path to executable motion](Lecture-03-Trajectory-Execution.md) | Check swept geometry and time the motion | Collision counterexample, duration calculation, tracking-error test | 60–90 min |
| [4. From a grasp pose to object success](Lecture-04-Grasp-and-Task-Success.md) | Evaluate contacts and object motion, not just the hand | Transform calculation, force assumptions, and paired task outcomes | 60–90 min |
| [Lab and independent transfer](Lab-Guide.md) | Change one model assumption and explain the result | Reproducible JSON, tests, prediction, and one failed case | 1–2 h |

## Entry diagnostic

Attempt these before reading. Each identifies a particular gap rather than a
reason to restart the roadmap.

1. A 2 kg point mass is 0.5 m from a horizontal joint. With `g=9.81 m/s²`, what
   holding torque is needed? Can an 8 Nm actuator supply it?
2. A controller minimizes position error and effort. Does that objective prevent
   crossing a wall or exceeding a motor limit?
3. Two trajectory endpoints are free of obstacles. Is the connecting motion free?
4. The hand rises 20 cm. What must you observe to establish that a pick succeeded?

<details markdown="1"><summary>Answers</summary>

1. `2 × 9.81 × 0.5 = 9.81 Nm`; 8 Nm is insufficient under this model.
2. No. A penalty and a hard constraint are different. Inspect the actual
   optimization problem and applied command.
3. No. The robot sweeps a region between endpoints. It can intersect an obstacle
   even when both endpoints are clear.
4. The object's lift and its relation to the hand, plus the task's retention and
   placement conditions. A hand trajectory alone is insufficient.

</details>

## Run the reference

From the repository root; commands were **tested here** with Python 3.12.3 on
Linux on 2026-10-18:

```bash
COURSE='Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Robot Dynamics and Task Execution'
python3 "$COURSE/labs/control_lab.py" all --output /tmp/robot-task-execution.json
python3 -m unittest discover -s "$COURSE/labs" -p 'test_*.py' -v
```

The supplied 24 tests check hand calculations, a finite-difference energy
derivative, an independent two-step control objective, the Riccati value identity,
interior collision extrema, timing peaks, transforms, and object-level failure.
They finish in seconds. The JSON includes implementation hashes and Python
version; retain it with your command and repository revision.

Some reference outcomes deliberately fail: an undersized actuator cannot hold the
target; a coarse action grid keeps moving near the goal; a hand can rise without
the object. Passing the exercise means predicting and detecting these outcomes,
not changing the success definition until every row is green.

## Continue into a complete system

Use [ROS 2 interfaces](../Advanced%20Robot%20Operating%20System/Lecture-01.md)
to carry the command, feedback, and timing contracts. Use the
[Isaac engineering sequence](../Isaac%20Sim%20and%20Isaac%20Lab/Lab-Guide.md)
to test an actual scene, moving robot geometry, and contact dynamics. Add
[robot learning](../Deep%20RL%20for%20Robot%20Learning/README.md) when a learned
component addresses a demonstrated limitation.

**Complete this course when:** another learner can reproduce your baseline and
variation, verify the worked numbers, identify the first failed assumption, and
explain what requires a simulator or physical measurement. Apply the
[shared rubric](../../../Learning-Guide.md#completion-rubric) to your evidence.
