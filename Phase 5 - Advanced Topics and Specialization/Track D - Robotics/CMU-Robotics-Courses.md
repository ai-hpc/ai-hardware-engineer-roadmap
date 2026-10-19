# Carnegie Mellon University: Robotics Courses Reference

**Reviewed against official sources: October 18, 2026.** This guide connects CMU robotics topics with exercises you can run in the roadmap. Course descriptions and public notes are useful references; university registration, equipment access and assessed credit are separate.

Begin with [Robotics Foundations](Robotics%20Foundations/README.md) if frames, feedback or planning are new. You can complete its CPU exercises without buying a robot or enrolling at CMU.

## 1. Bachelor of Science in Robotics (BSR)

CMU's School of Computer Science began the BSR in Fall 2023. The [program overview](https://www.ri.cmu.edu/education/academic-programs/bachelor-of-science-in-robotics/) emphasizes mathematics, sensing, mechanisms, control and integrated robot projects. The [2026–27 B.S. catalog](https://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/robotics/robotics-bs/) is the source for the 360-unit degree and its full requirements.

The robotics core includes:

| Course | Focus |
|---|---|
| 16-280 **or** 16-281 | Intelligent Robot Systems or General Robotics |
| 16-220 | Robot Building Practices |
| 16-299 | Introduction to Feedback Control Systems |
| 16-384 | Robot Kinematics and Dynamics |
| 16-385 | Computer Vision |
| 16-450 → 16-474 | Robotics Systems Engineering followed by Robotics Capstone |

**16-281 is the current number for the former 16-311**, effective Spring 2026; it is not a future replacement still to arrive. The catalog also includes programming/systems, calculus, linear algebra, probability, 16-211 robotics mathematics, ethics and electives. Its sample sequence assumes prior introductory programming and a semester of calculus, so it is not a starting plan for a complete beginner. Consult the catalog for the full schedule and approved probability options rather than treating a shortened course list as degree advice.

## 2. Robotics minor & additional major

The [current minor catalog](https://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/minors/robotics-minor/) lists 15-112 as the eligibility prerequisite and strongly recommends 15-122. It combines an overview course, controls, mechanisms/manipulation, building practice and two electives. Its 2.5 QPA and double-counting rules apply to enrolled CMU students, not to completing this roadmap.

For the [additional major](https://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/robotics/robotics-additional-major/), use that program's current curriculum and application links. Application windows and course substitutions can change; an undated January–February deadline copied from an older page is not reliable enrollment guidance.

## 3. Key robotics courses and accessible material

| Reference | Verified term or source | Useful scope and access boundary |
|---|---|---|
| [16-281 General Robotics](https://www.cs.cmu.edu/~16311/current/) | Spring 2026 page; legacy URL retains `16311` | Public schedule, homework/lab descriptions and prerequisite knowledge; team construction labs need physical components |
| [16-299 Feedback Control](https://www.cs.cmu.edu/~cga/controls-intro/) | Spring 2026 | Public notes/code and some recordings; includes balancing-robot work and simulation |
| [16-385 Computer Vision](https://16385.courses.cs.cmu.edu/spring2026/) | Spring 2026 | Public course information; signup uses a Canvas code, so not all resources are guaranteed public |
| [16-831 Introduction to Robot Learning](https://cmu16831.github.io/26fall/) | Fall 2026 | Imitation learning, RL, visual decision-making and sim-to-real; course logistics and access are separate from the public overview |

The 16-299 page describes a revision toward control, planning and learning while retaining the existing catalog title. Some recordings are explicitly missing or incomplete. Read its topic/date labels rather than assuming every linked note belongs to one fully recorded current course. Its hardware assignments are not required for the roadmap's smaller CPU control exercises. [Spring 2026 source](https://www.cs.cmu.edu/~cga/controls-intro/).

The 16-831 [Fall 2026 logistics](https://cmu16831.github.io/26fall/logistics/) identify Piazza, Gradescope and a homework repository. A public course homepage alone does not establish access to graded submissions, solutions or teaching support. The catalog's older `16-831.github.io` course URL returned 404 during this review; use the term-specific page above.

Other useful [catalog courses](https://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/courses/) are listed below. A catalog entry confirms a course description, not that a particular term has a complete public course package.

| Code | Course/topic |
|---|---|
| 16-180 | Concepts of Robotics |
| 16-211 | Foundational Mathematics of Robotics |
| 16-350 | Planning Techniques for Robotics |
| 16-362 | Mobile Robot Algorithms Laboratory; multirotor simulation |
| 16-663 | F1Tenth Autonomous Racing; vehicle hardware labs |
| 16-664 | Self-Driving Cars: Perception & Control |
| 16-735 | Ethics and Robotics |
| 16-884 | Deep Learning for Robotics; research discussion and project |

For 16-350, the catalog expects introductory robotics plus programming/data structures. For 16-362, it expects vectors, matrices, coordinate systems and proficiency in an object-oriented language. These are substantive foundations; “introduction” in a course title does not mean no prerequisites. The catalog describes ROS in 16-280 but does not specify a ROS 2 distribution, so use the roadmap's compatibility-tested ROS material for installation.

## 4. A connected self-study route

The order below is our learning recommendation, not a claim that these labs replace CMU courses or degree requirements. Mechanical construction can run alongside software study when equipment is available.

| Stage | CMU topic reference | Do this in the roadmap |
|---|---|---|
| Programming and mathematics | 15-122 preparation, 16-211 | [Foundations](Robotics%20Foundations/README.md): vectors, frames, joints and timing |
| Kinematics and feedback | 16-384, 16-299 | [CPU robotics lab](Robotics%20Foundations/Lab-Guide.md): predict an arm position and explain a delayed controller |
| Dynamics and constrained execution | 16-299, 16-384 | [Dynamics/task lab](Robot%20Dynamics%20and%20Task%20Execution/Lab-Guide.md): torque limits, LQR/MPC, collision and grasp-frame checks |
| Perception and estimation | 16-385, 16-299 | [AI/vision reference](../../Phase%203%20-%20Artificial%20Intelligence/CMU-AI-Courses.md) plus timestamp/frame-aware integration |
| Planning and software integration | 16-350, 16-280/16-281 | Explain the foundation lab's A* path, then run the [optional local ROS 2 lab](Robotics%20Foundations/ROS2-Lab.md) on its supported environment |
| Robot learning | 16-831 | [Train a small CPU policy](../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-01-CPU-Policy.md), then study [Deep RL](Deep%20RL%20for%20Robot%20Learning/README.md) |

A concrete readiness check: a camera's x-axis points along the robot base's y-axis. A camera-frame displacement of `(10, 0)` millimeters becomes base-frame `(0, 0.01)` meters. Translating the frame origin does not change a displacement, although it does change a point's coordinates. If that distinction is unclear, work through the frame lesson before applying a learned action tensor to a robot. If the math is clear but the command arrives late, the missing topic is timing and control, not a larger neural network.

After the CPU route, choose a physical integration only when you can explain the sensor frame, action units, observation age and fallback behavior. CAD practice, assembling a robot and operating a controller on real hardware remain separate skills from a successful simulation.
