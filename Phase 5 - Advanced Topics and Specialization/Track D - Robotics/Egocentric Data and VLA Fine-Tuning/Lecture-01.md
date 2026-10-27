# Lecture 01 — Why Egocentric Human Data for Robot Learning

Decide what first-person human video can and cannot give a robot policy, and budget a dataset before collecting or downloading anything.

**Prerequisites:** [VLA Foundations Lecture 01](../VLA%20Foundations%20to%20Frontier%202026/Lecture-01.md) (what a VLA policy is). **Effort:** reading 1.5 hours, exercise 2 hours. **Hardware:** CPU track; the exercise is arithmetic and a written inventory. **Roles:** robot learning data engineer, post-training engineer.

**Misconception this lecture corrects:** "human video is robot data without the robot, so more of it always helps." Human video is missing the very thing a policy outputs (robot actions), and how much it helps depends on how the gap is bridged and on how broadly the base model was pretrained.

## 1. The data pyramid

Robot-learning teams describe their data as layers that trade scale against how directly the data matches the robot:

| Layer | Example | Scale | Has robot actions? | Matches the robot's body and cameras? |
|---|---|---|---|---|
| Web and generic video | Video datasets, internet clips | Largest | No | No |
| **Egocentric human video** | Head-mounted cameras recording hands doing tasks | Thousands to tens of thousands of hours | No; hand motion can be tracked | Partly: the viewpoint is close to a head camera, but the body is a human's |
| Simulation and generated data | Simulated demos, video models | Very large | Yes (simulated or inferred) | Approximately |
| Teleoperated robot demos | An operator drives the real robot | Hours to thousands of hours | Yes | Yes |
| On-robot practice | RL or autonomous collection | Small | Yes | Yes |

Published examples show the shape:

* **GR00T N1** (arXiv 2503.14734) trained on what NVIDIA calls a data pyramid: real-robot trajectories, human video, and synthetic data. Its paper reports turning 88 hours of in-house teleoperation into about 827 hours of generated video, and 780,000 simulated trajectories (about 6,500 hours) produced in 11 hours.
* **GR00T N1.7** (Isaac-GR00T README, April 2026) is "pretrained on 20K hours of EgoScale human video". **EgoScale** (arXiv 2602.16710) reports training on over 20,854 hours of action-labeled egocentric human video and a log-linear scaling law.
* **π0's base checkpoints** were "pre-trained on 10k+ hours of robot data" (openpi README). Physical Intelligence later reported that **human-to-robot transfer emerges** once a VLA is pretrained on enough scenes, tasks and embodiments (arXiv 2512.22414). Their blog reports about 2× on four generalization scenarios that only human data covered.
* **EgoMimic** (arXiv 2410.24221) co-trained one policy on Project Aria human video and robot data. Its abstract states that "1 hour of additional hand data is significantly more valuable than 1 hour of additional robot data" for its tasks. That is a result about their setup, not a general exchange rate.

## 2. What egocentric video gives, and what it lacks

**It gives:** many tasks, objects, homes and people; hands visible from roughly where a robot's head camera sits; natural task structure (reach, grasp, transport, place); and, from devices such as Project Aria glasses or Apple Vision Pro (EgoDex, arXiv 2505.11709), tracked 3D hands and head pose.

**It lacks:** robot actions; proprioception (joint angles, gripper state); contact forces; and a body that matches the robot's. A head camera also moves with the person, so the "camera frame" changes every frame, which a fixed robot camera does not.

## 3. The embodiment gap, broken into parts

"Embodiment gap" is too vague to engineer against. Split it:

| Gap | Human video | Typical robot | Bridged by (lecture) |
|---|---|---|---|
| **Visual** | A human arm and hand in the image | A robot arm and gripper | Masking or inpainting the arm and overlaying a rendered robot, as in Phantom (arXiv 2503.00779); robust features ([08](Lecture-08.md)) |
| **Kinematic** | Five fingers with far more degrees of freedom than a gripper, and a human reach envelope | A parallel-jaw gripper on a 6–7-DoF arm | Retargeting with feasibility checks ([07](Lecture-07.md)) |
| **Dynamic** | Human speed and force | Joint velocity, torque and payload limits | Time scaling and speed limits ([07](Lecture-07.md)) |
| **Sensing** | No proprioception or force | Joint encoders, sometimes force sensing | Latent actions and pseudo-labels; robot data supplies the rest ([08](Lecture-08.md)) |
| **Viewpoint** | A moving head camera | Fixed scene cameras and wrist cameras | SLAM to a stable world frame ([03](Lecture-03.md)); camera-aware training |
| **Action representation** | Hand poses over time | The robot's action space (joint or end-effector, absolute or relative) | A shared space, such as relative end-effector actions in GR00T N1.7 or UMI ([07](Lecture-07.md), [10](Lecture-10.md)) |

## 4. Three ways to use human data

| Use | What the model learns from human data | What you need from the data | Examples |
|---|---|---|---|
| **Representation pretraining** | Visual and language features of hands, objects and tasks | Video and text; no actions | PhysBrain (arXiv 2512.16793) |
| **Action pretraining** | To predict actions on human data before seeing robot data | Hand tracks turned into actions, latent actions, or pseudo-actions | LAPA (arXiv 2410.11758), GR00T N1, EgoScale, VITRA (arXiv 2510.21571) |
| **Co-training and fine-tuning** | A policy trained on human and robot data together | Human actions in a space shared with the robot, and a mixture plan | EgoMimic, Humanoid Policy ~ Human Policy (arXiv 2503.13441), π0.5 human-to-robot transfer |

This course builds toward the third row for your own task (Lectures 07, 10–13), and explains the second row so you can read and reuse pretrained models (Lecture 08).

## 5. The hardware view: video is the expensive part

Before downloading anything, convert hours into frames and bytes. The following is **worked arithmetic with stated assumptions**, not a measurement of any particular dataset:

$$
\text{frames} = \text{hours} \times 3600 \times \text{fps}, \qquad
\text{compressed bytes} \approx \text{hours} \times 3600 \times \frac{\text{bitrate}}{8}
$$

* One hour at 30 fps is 108,000 frames.
* At an assumed 5 Mbit/s compressed bitrate, one hour is about 2.25 GB on disk. 20,000 hours is then about 45 TB.
* Decoded to 224 × 224 × 3 bytes for a model input, one frame is 150,528 bytes, so one hour is about 16 GB of raw frames. That is why pipelines decode on the fly instead of storing decoded frames.
* A training run that consumes 1,000 frames per second needs a decoder that sustains 1,000 frames per second, on top of everything else the CPUs and GPU are doing. Lecture 04 budgets decode throughput before training starts.

The robot layer is tiny by comparison. LeRobot's SmolVLA guide recommends about 50 episodes for a new task. At roughly 30 s per episode, that is 25 minutes of robot data, and it is the layer that carries the robot's actual actions.

## 6. Worked example: budget one task

Task: "put the cup in the bowl" on a single 7-DoF arm with a parallel-jaw gripper, one wrist camera and one scene camera at 30 fps.

| Item | Amount | Arithmetic |
|---|---|---|
| Robot demos | 50 episodes × 30 s | 25 min, 45,000 frames per camera, 90,000 frames total |
| Human clips (same task family) | 10 hours from a licensed source | 1.08 M frames; about 22.5 GB at 5 Mbit/s |
| Retarget-ready subset | The clips with valid hand tracking and SLAM, estimated at 40% | About 4 hours; the real fraction is measured in Lectures 05 and 06 |
| Gaps to bridge | Visual (human arm), kinematic (5 fingers → gripper), viewpoint (head → wrist camera) | Section 3 |

Two decisions follow before any training. First, does the license of the human source allow your use (Lecture 02)? Second, is a parallel-jaw gripper action derivable from the hand tracks you have (Lecture 07)?

## 7. Exercise

**Exercise / pseudocode** — written, CPU only.

1. **Worked example → your task.** Pick a manipulation task and a target robot. Fill in the budget table from Section 6 with your own numbers, labeling every number as measured, documented or assumed.
2. **Guided variation.** Change the robot from a parallel-jaw gripper to a five-finger hand. Which rows of the embodiment-gap table get easier and which get harder?
3. **Independent transfer.** Pick one method from Section 4. Write two sentences on which gap in Section 3 it addresses and which it leaves to robot data.

**Deliverable:** `data_pyramid.md`, containing the budget table, the embodiment-gap inventory for your robot, and the method classification. **Acceptance:** every number has a label and every gap row names a bridging approach or states that robot data must cover it.

## Failure clinic and self-check

**Failure clinic:** a team downloads tens of terabytes before checking the license, and has to delete it (Lecture 02). A budget counts hours of video but not hours with usable hand tracking (Lecture 06). A plan assumes human data replaces robot data entirely for a dexterous task, with no evidence for that setup.

1. **Why can't you train a VLA's action output on raw human video alone?** There are no robot actions in it. You need hand tracks converted to a shared action space, latent or pseudo-actions, or co-training with robot data that supplies real actions.
2. **EgoMimic reports that an extra hour of human data beat an extra hour of robot data. Can you plan with that ratio?** No. It is a result for their tasks, hardware and pipeline. Treat it as a hypothesis and test it with a paired ablation (Lecture 12).
3. **Why do pipelines decode video on the fly?** Decoded frames are roughly an order of magnitude larger than compressed video (Section 5), so storing them costs far more disk and I/O than decoding them again.
4. **Which embodiment gap does SLAM help with?** The viewpoint gap. It expresses hand motion in a stable world frame instead of the moving head-camera frame, so the motion can be mapped to a robot base frame.

[Course index](README.md) · [Next: datasets and licenses](Lecture-02.md)
