# Lecture 13 — Robot Data Collection, Teleoperation, and Annotation

Build a collection protocol that produces causally aligned, diverse demonstrations another engineer can inspect and reuse.

**Prerequisites:** Lectures 04–05 and 09; coordinate frames, timestamps, and control semantics. **Effort:** setup 1 hour, collection/replay lab 4 hours, audit 2 hours. **Hardware:** CPU and synthetic or existing recordings for the required exercise; teleoperated robot and calibrated cameras for the optional extension. **Stack/roles:** sensors, control software, storage, and training data; robotics data engineer and integration engineer. **Artifact:** a collection specification, calibration record, and audited episode bundle.

## A demonstration is a timed measurement

A video and an action array are not enough. You need to know what the operator saw, what command the robot received, what the robot actually did, and when each event occurred. Commands and measured states differ under latency, joint limits, controller tracking error, and contact. Recording measured joint positions as though they were commanded positions can silently change the learning target.

For each stream retain the source timestamp, clock domain, sequence number, and reception timestamp where available. Calibrate clock offset and monitor drift. A camera frame's acquisition timestamp is preferable to a timestamp assigned after video decoding. Record the controller mode, command period, robot configuration, gripper semantics, camera intrinsics/extrinsics, and calibration revision with each session.

An illustrative alignment contract says: observation includes only measurements available by decision time \(t\); action is the command applied over \([t,t+\Delta t)\). If a dataset uses a different convention, document and convert it. Nearest-neighbor alignment can accidentally choose a future image. Causal interpolation uses past information; offline target construction may use future measurements only when those measurements are labels, not deployed inputs.

## Teleoperation changes the data distribution

A joint-mirroring leader/follower interface, Cartesian gamepad, VR controller, and kinesthetic teaching system have different delays, workspace constraints, and operator feedback. They also create different action smoothness and recovery strategies. The ALOHA work is a concrete example of an integrated teleoperation and imitation pipeline. Its hardware design should be studied with its controller assumptions, rather than treated as a universal collection interface. [ALOHA, RSS 2023](https://roboticsproceedings.org/rss19/p016.html).

The operator's strategy can become a hidden variable. One person approaches a handle from above, another from the side, while the policy receives identical observations. Either preserve enough state to disambiguate these choices or train a distribution model capable of representing them. Do not force contradictory demonstrations into a single deterministic target without examining the ambiguity.

Record autonomy and intervention intervals separately. An episode that ends successfully after a human rescue is not an autonomous success. Keep the original policy command, override command, override reason, and operator identity or anonymized code where permitted. Distinguish intervention data from evaluation data in aggregate reports.

## Coverage is more valuable than repeated convenience

Create a collection matrix across task, object instance, initial pose, background, lighting, camera pose, operator, robot, and failure/recovery condition. Prioritize missing combinations that matter to deployment. Repeating the easiest initial pose may reduce supervised loss while leaving the operational distribution untouched.

DROID is a useful primary example of distributed, diverse real-world collection. BridgeData V2 is another important reference for reusable manipulation data and variation across environments. These datasets motivate explicit coverage records, but neither establishes that any chosen downstream task is represented adequately. [DROID, 2024](https://arxiv.org/abs/2403.12945), [BridgeData V2, 2023](https://arxiv.org/abs/2308.12952).

Keep failures when their provenance and outcome labels are trustworthy. For basic BC, unsuccessful actions may be unsuitable supervision, while recovery segments can be valuable. For value learning, failures are essential negative evidence. Preserve raw recordings and implement selection in a versioned manifest rather than permanently deleting everything that failed.

## Annotation is a control interface

Use task-level language for goals and segment-level language for the current subtask. “Place the mug in the left bin” should have an unambiguous frame of reference. Annotate object identity and destination consistently. If the policy receives a subtask label at deployment, explain who generates that label and how transitions are detected.

Avoid language that reveals an outcome unavailable when the action is selected, such as a post hoc statement that a grasp will succeed. Outcome metadata can supervise a critic or a separate training condition, but its deployment use must be specified. Retain annotator disagreement and ambiguous intervals. A confident but incorrect label can teach the model to ignore language.

## Worked example: a small timing error becomes spatial error

At an illustrative end-effector speed of 0.30 m/s, an 80 ms image/action offset corresponds to 24 mm of motion. That may exceed the tolerance of a placement task even if every frame looks clear. A timestamp bug can therefore resemble weak visual grounding.

Storage estimates also guide collection. Three uncompressed 640×480 RGB cameras at 30 frames/s produce \(3\times640\times480\times3\times30\approx82.9\) MB/s in decimal units, roughly 299 GB/hour before metadata. Compression reduces storage but introduces decode cost and may obscure contact cues. Measure the actual codec, bitrate, and random-access latency instead of assuming video compression is free.

## Build the collection audit

**Exercise / pseudocode.** Use an existing episode bundle or synthesize three timestamped camera streams, robot states, and commands. The [shared lab guide](Lab-Guide.md) supplies CPU contract exercises; this collection audit is an additional implementation task.

1. Write an episode schema and controller contract. Include source IDs, calibration hashes, action units, control period, success definition, termination reason, and intervention spans.
2. Build causal alignment to a control timeline. Save alignment residuals and report maximum, median, and p99 observation age for each sensor.
3. Add dropouts, duplicated frame IDs, a clock offset, and one clock jump. Reject or mark invalid affected samples; do not silently stretch them into clean demonstrations.
4. Review ten aligned clips with overlaid command vectors and event markers. Select clips by a seeded rule, including failures and interventions.
5. Produce a coverage table and identify the next ten demonstrations that would add the most useful diversity. Explain the proposed value of each group.
6. For the optional robot extension, record a short bounded-motion session using the lab's established controller limits and stop procedure. Replay it offline before training.

Stop after 20 audited episodes or two hours of collection. **Acceptance criteria:** no future observations enter policy inputs; clock discontinuities are visible; success and intervention are distinct labels; command and measured state are separate fields; an independent reader can reconstruct action semantics and coverage gaps. The CPU exercise cannot validate camera calibration or physical controller behavior.

## Failure clinic and self-check

**Failure clinic:** an unexplained action lead suggests timestamp convention mismatch; successful videos with poor learned policies suggest narrow coverage or label ambiguity; frequent video decoding stalls suggest storage layout trouble; unexplained operator differences suggest interface or strategy variation.

1. **Why retain reception time if acquisition time exists?** Their difference helps reveal transport and processing latency; the two timestamps answer different questions.
2. **Is a successful rescued episode an autonomous success?** No; report the intervention and retain its useful correction data separately.
3. **Can failed data help BC?** Useful recovery segments can help, but blindly imitating failure actions can degrade behavior. Selection must match the objective.
4. **What makes this engineering work reproducible?** An explicit timing/action contract, immutable raw data, versioned selection, and audit artifacts.

[Previous: flow matching](Lecture-12.md) · [Course index](README.md) · [Next: dataset engineering](Lecture-14.md)
