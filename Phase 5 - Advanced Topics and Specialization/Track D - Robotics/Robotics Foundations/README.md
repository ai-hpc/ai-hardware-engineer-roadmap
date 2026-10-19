# Robotics foundations: from a point on a screen to a controlled motion

Build the geometry, estimation, planning, and feedback skills needed to understand what a robot actually does with an AI model's output. Start here before Advanced ROS, Isaac, or VLA deployment if those ideas are unfamiliar.

**Layer mapping:** application/workload requirements → algorithms and runtime → embedded sensing and actuation. **Roles:** robotics software, robot learning, perception, controls, and edge-AI engineer. **Prerequisites:** basic Python functions, vectors as lists of numbers, sine/cosine, and algebra. **Next:** [Advanced ROS 2](../Advanced%20Robot%20Operating%20System/Lecture-01.md), then [Isaac](../Isaac%20Sim%20and%20Isaac%20Lab/README.md) or [VLA foundations](../VLA%20Foundations%20to%20Frontier%202026/README.md).

The running example is a small mobile robot with a camera and a two-joint arm. The camera sees an object; the robot estimates where it is, chooses a reachable collision-free motion, and corrects the motion using feedback. Each lesson explains one part of that chain. A CPU laptop and Python 3.10+ are enough for every supplied lab. There is no simulator, GPU, ROS installation, or physical robot requirement until the optional ROS integration exercise.

New to the notation? The [beginner reading aid](../../../Beginner-Reading-Aid.md)
works through radians, a 90° rotation, a cart update, and observation age. You
can begin with those arithmetic examples and return to the general equations
afterward. “Layer mapping” above places the course in the larger hardware
roadmap; it is not extra prerequisite material.

## Try the entry diagnostic

Spend ten minutes without looking up the answers:

1. A robot is at `(1, 2)` metres and faces 90° counterclockwise from world x. Where is a point one metre directly ahead of it in world coordinates?
2. Two one-metre arm links are straight along x. Can an arbitrarily small joint motion move the tool directly farther along x?
3. A camera captures at 30 Hz, but an inference takes 100 ms. Does a 30 Hz publisher make the observation used by the controller fresh?
4. A grid path fits through a one-cell gap. What additional fact do you need before sending a real robot through it?

Answer guidance: **(1)** `(1, 3)`; rotate before translating. **(2)** No: at this singular configuration the instantaneous tool velocities lie along y. **(3)** No: measure capture-to-use age, including queues and inference. **(4)** The robot footprint and uncertainty margin, plus whether its motion constraints permit the path. If all four are easy, run the labs and attempt the independent tasks before reading the worked examples.

## Follow the chain

| Lesson | Question it answers | Inspectable artifact |
|---|---|---|
| [1. Frames, units, and time](Lecture-01.md) | What does this number refer to, and when was it true? | Transform round trip and rejected stale sample |
| [2. Kinematics and reachable motion](Lecture-02.md) | Which joint or wheel motion reaches the target? | Two IK solutions, numerical Jacobian check, wheel-motion calculation |
| [3. Feedback, dynamics, and delay](Lecture-03.md) | Why can a correct target still produce oscillation? | Stable and unstable traces with an explained timing boundary |
| [4. Camera geometry and uncertainty](Lecture-04.md) | How do pixels become a noisy spatial estimate? | Projection/deprojection check and an uncertainty update |
| [5. Planning and collision margins](Lecture-05.md) | Is there a valid route, and can the robot follow it? | Shortest-path comparison and a correctly rejected narrow passage |
| [6. Put the pieces into ROS 2](Lecture-06.md) | Which nodes exchange these quantities, and how do failures stop motion? | Interface contract, fault matrix, and optional ROS trace |

Allow about **8–12 hours** for a first pass: 10–20 minutes to set up the CPU lab, then roughly 30 minutes of study, 30–60 minutes of practice, and 15–30 minutes of explanation per lesson. Split the lessons across days; recall the previous lesson's example before starting the next. ROS installation and simulator integration need separate sessions and are not included in that estimate.

## Run the first experiment

Start at the **repository root**, then change into this course directory once.
The `cd` stays in effect for the remaining commands in that terminal. If you are
already in Robotics Foundations, skip the `cd` line. **Tested here:** Python
3.12.3 on Linux, 2026-10-18. Only the Python standard library is used.

```bash
cd 'Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Robotics Foundations'
python3 labs/foundations.py frames
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

The transformed camera point should be approximately `[1.0, 3.2]` metres. The inverse should recover `[1.0, 0.0]` within floating-point rounding. The suite contains 23 tests, including independent finite-difference and breadth-first-search checks. Read [the lab guide](Lab-Guide.md) to save all results and build the final artifact.

Keep using this course directory for `python3 labs/foundations.py ...` in lessons
1–5. Return to the repository root when switching to the AI policy lab's longer
`Phase 3 - ...` paths. The [working-directory explanation](../../../Beginner-Reading-Aid.md#1-know-where-a-command-starts)
shows how to check where you are.

These are **synthetic numerical fixtures**. They teach invariants and failure detection; they do not validate real robot dynamics, camera calibration, middleware timing, GPU throughput, collision geometry, or physical safety. Keep those claims separate when moving into a simulator or onto hardware.

**Have ROS 2 installed?** Continue with the [measured ROS 2 CPU lab](ROS2-Lab.md): real stamped-message transport and TF, plus controlled stale/future/frame rejections. Its saved run records exact package versions and separates local middleware evidence from the synthetic inputs.

## Why this matters for AI hardware

The geometry often needs little compute but very precise semantics. Camera ingestion, perception, and a VLA may consume most accelerator memory and time. The controller still needs recent state at its own rate. Your hardware budget must therefore include capture, transfer, inference, queueing, control, and actuator delay—not only a fast model forward pass. An optimized policy that changes action units or misses the required deadline has not preserved robot behavior.

## Finish with evidence

Produce the [lab-guide handoff](Lab-Guide.md#final-artifact): numerical results, one changed case per lesson, the intentional failures your checks detect, and the ROS interface contract. Use the [shared rubric](../../../Learning-Guide.md#completion-rubric): at least 8/10, with full correctness and reproduction scores. Continue to Isaac when you can explain why passing these CPU checks is necessary but insufficient for a valid simulator task. Continue to VLA deployment when you can name the policy's action frame, units, rate, limits, and consuming controller.
