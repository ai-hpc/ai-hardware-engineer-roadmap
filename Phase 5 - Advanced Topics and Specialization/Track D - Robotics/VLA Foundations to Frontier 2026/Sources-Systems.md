# Primary sources for evaluation, simulation, and deployment

[Course overview](README.md) · [Lecture 19](Lecture-19.md) · [Frontier register](Frontier-Register.md)

**Checked through October 13, 2026.** These are reading and provenance pointers for Lectures 19–24 and the capstones. The lecture derivations, synthetic fixtures, and worked numerical examples are course material; they are not benchmark claims attributed to these sources. Mutable documentation must be pinned to an exact version for execution.

| Source | Why to read it | Boundary of the evidence |
|---|---|---|
| [LIBERO project](https://libero-project.github.io/main.html), [datasets](https://libero-project.github.io/datasets), and [repository](https://github.com/Lifelong-Robot-Learning/LIBERO) | Understand language-conditioned lifelong learning suites and environment/data interfaces | Suite, split, initial states, checkpoint, and environment revision matter; historical install instructions may require an isolated legacy environment |
| [RoboCasa365 project](https://robocasa.ai/), [benchmark documentation](https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html) | Study task distribution and protocol design | A simulation benchmark is not a physical-robot guarantee; project release history records changes affecting comparison |
| [RoboCasa365 leaderboard](https://robocasa.ai/leaderboard.html), updated October 13, 2026 | Inspect protocol-specific model comparison and submission evidence | Entries cover a documented 50-task multitask setting; task budget and evaluation revisions must match |
| [Isaac Lab 3.0.0-EA installation](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html) | Resolve the teaching stack and backend-specific dependencies | Early-access APIs and backend support are version-specific; this course did not execute Isaac |
| [Isaac Lab task/project generator](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/developer-tools/template_generator.html) | Build a task under the selected version's supported structure | Generated task scaffolding still needs a policy adapter, observation/action contracts, and validation |
| [Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html) | Check supported hardware/driver requirements | Smaller scene size does not change the published support boundary; model plus renderer memory must be measured together |
| [ROS 2 clocks and time](https://design.ros2.org/articles/clock_and_time.html) | Distinguish clock domains and simulation-time behavior | Clock APIs do not by themselves provide distributed synchronization or real-time guarantees |
| [ROS REP-103](https://www.ros.org/reps/rep-0103.html), [official source](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst) | Establish units and coordinate conventions | Individual robot datasets can use different conventions and require conversion |
| [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339) (2025) | Read the research method for consistent asynchronous chunk execution | The supplied CPU scheduling helper only teaches expiry arithmetic; it is not an implementation of this paper |
| [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) | Distinguish quantized formats, scale granularity, and operator constraints | A moving runtime reference is not evidence that a specific VLA/device combination supports a native kernel |
| [LeRobot repository](https://github.com/huggingface/lerobot) | Inspect public robot-learning data, policy, and deployment implementations | Pin the repository and policy recipe; a common framework does not make all checkpoint contracts interchangeable |

## Reading sequence

Start with the LIBERO and RoboCasa protocol descriptions before reading model leaderboard numbers. Then trace the Isaac task boundary and ROS clock semantics into your own adapter. Read the asynchronous-flow paper after drawing the Lecture 22 timeline, so its treatment of committed action prefixes has a concrete systems meaning. Use quantization documentation only after identifying the architecture and target hardware you intend to measure.

For system identification, selective-execution analysis, and confidence intervals, this course teaches explicit simplified models and assumptions rather than claiming a universal robot deployment standard. Validate the statistical method and controller behavior needed by your project; a small pedagogical helper is not a complete production implementation.

## Claims requiring your own measurements

The course does not establish a standard VLA control frequency, a universal GPU memory requirement, a Jetson latency guarantee, a quantization speedup, or an Isaac task success rate. Those depend on the actual artifact, runtime, sensor pipeline, controller, and workload. The only locally executed model training in this course authoring is the tiny synthetic CPU exercise documented in [Lab-Guide.md](Lab-Guide.md).

When following an upstream command, mark it **upstream documented** until you run it in your pinned environment. Retain its output and failures. A paper's experiment, a model-card example, a course pseudocode sketch, and your own measured result are separate evidence classes.
