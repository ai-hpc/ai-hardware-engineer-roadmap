# VLA Foundations to Frontier 2026 — Professional Course

<div class="course-identity robotics" markdown="1">
<div class="course-identity__icon">VLA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · Robotics · Professional Course</p>
<p class="course-identity__title">Understand, train, evaluate, and deploy vision-language-action policies—from their historical foundations to the October 2026 research frontier.</p>
<p class="course-identity__meta">32 detailed lectures · mathematical derivations · CPU labs · Isaac extensions · two capstones</p>
</div>
</div>

**Source cutoff: October 13, 2026.** This is a substantial engineering course, not a claim to cover every paper or developments later in October. Current-model coverage distinguishes announced results, papers, code, downloadable weights, API access, and partner access. Start with [Lecture 01](Lecture-01.md); use the [frontier register](Frontier-Register.md) when selecting an actual model artifact.

A VLA maps visual observations, language, and often robot state/history to actions. Professional work requires understanding the entire chain: data provenance, representation, learning objective, model architecture, action semantics, controller integration, inference timing, and closed-loop evidence. This course teaches that chain through worked examples, guided experiments, negative controls, and independent transfer tasks.

**Role targets:** Robot Learning Engineer · Robotics Foundation-Model Engineer · Embodied AI Research Engineer · VLA Training Engineer · Robotics Inference and Evaluation Engineer. **Roadmap layers:** L3–L8, from model/training choices through GPU execution, runtime integration, and robot behavior.

## The chip this course helps you design

This is the model and control foundation for the roadmap's larger goal: **a custom physical-AI inference SoC built for advanced VLA workloads, with a dedicated accelerator, an AI compiler, embedded Linux, and integrated wireless for robots, drones, and autonomous machines**. Learn how models work so their actual computation, state, numerical sensitivity, and timing requirements can guide the hardware.

Carry a design brief through the course: selected checkpoint and tasks; vision/language/action operator coverage; supported shapes and precision; memory and bandwidth; sensor-to-action deadlines; compiler/runtime interfaces; and closed-loop quality gates. Use [Capstone II's hardware co-design route](Lecture-32.md#hardware-co-design-route) to turn this into an experiment, then continue into the [physical-AI SoC capstone](../../Track%20F%20-%20AI%20Chip%20Design/Guide.md#physical-ai-soc-capstone). Completing the VLA course supplies workload evidence; the chip-design, compiler, Linux, and wireless tracks provide the other implementation milestones.

## Outcomes and prerequisites

By the end, you should be able to derive the central imitation/diffusion/flow objectives, read a VLA paper critically, build a causal robot dataset pipeline, adapt an accessible policy, define its physical action contract, evaluate it in closed loop, and defend one measured training or deployment result. The CPU route establishes foundations and tooling; the GPU/simulator route adds model adaptation and robot-learning integration. Physical deployment requires additional device-specific experience and supervised validation.

Required background is Python, linear algebra, probability, gradient descent, basic neural networks, and elementary feedback control. Familiarity with PyTorch helps with advanced exercises. You do not need to complete every robotics specialization first. For gaps, use [the robotics guide](../Guide.md), [Deep RL for Robot Learning](../Deep%20RL%20for%20Robot%20Learning/README.md), and the [Isaac course](../Isaac%20Sim%20and%20Isaac%20Lab/README.md) selectively.

**Entry diagnostic:** explain a train/test split, compute an affine normalization, differentiate a squared-error loss, distinguish pose from velocity, and draw a sensor-to-controller loop. If any answer is uncertain, work through Lectures 01, 06, 09, and 14 before attempting model adaptation. The [notation and glossary](Reference.md) is a companion, not a substitute for those lessons.

## Course map

Each lecture teaches a concrete mechanism, works through an example, specifies a lab or exercise, identifies failure modes, and provides self-check answer guidance. The historical sequence establishes why architectures changed; later modules teach how to reproduce and evaluate them.

<div class="lecture-map" markdown>

| # | Lecture | Inspectable learning artifact |
|---|---|---|
| 01 | [What a VLA policy is](Lecture-01.md) | Policy/control contract and POMDP diagram |
| 02 | [Historical foundations: control, imitation, and deep visuomotor learning](Lecture-02.md) | Historical mechanism timeline and distribution-shift experiment |
| 03 | [Language grounding: CLIP, CLIPort, and SayCan](Lecture-03.md) | Grounding and skill-affordance comparison |
| 04 | [RT-1, RT-2, and PaLM-E: three distinct architectural steps](Lecture-04.md) | RT-1/RT-2/PaLM-E architecture and evidence table |
| 05 | [Open X-Embodiment, RT-X, Octo, and OpenVLA](Lecture-05.md) | Cross-embodiment schema and baseline-selection record |
| 06 | [Geometry, robot state, and control contracts](Lecture-06.md) | Frame, unit, state, and action validation |
| 07 | [Visual and language encoders, fusion, and attention masks](Lecture-07.md) | Token shapes, attention masks, and compute budget |
| 08 | [Action representations, quantization, and FAST](Lecture-08.md) | Action/token codec and reconstruction analysis |
| 09 | [Behavioral Cloning, Sequence Losses, and Covariate Shift](Lecture-09.md) | Masked BC objective and covariate-shift experiment |
| 10 | [Action Chunking, ACT, and Temporal Ensembling](Lecture-10.md) | Chunk scheduling and temporal-ensemble comparison |
| 11 | [Diffusion Action Policies from Noise to Control](Lecture-11.md) | Diffusion objective, sampler, and action-mode analysis |
| 12 | [Flow Matching and the π₀ Architecture](Lecture-12.md) | Flow-matching derivation and π₀ component map |
| 13 | [Robot Data Collection, Teleoperation, and Annotation](Lecture-13.md) | Collection protocol and annotation audit |
| 14 | [Dataset Schemas, Transformations, Splits, and Mixtures](Lecture-14.md) | Causal dataset transform and leakage report |
| 15 | [Pretraining and Cross-Embodiment Generalization](Lecture-15.md) | Cross-embodiment objective and mixture experiment |
| 16 | [Fine-Tuning, Adapters, Normalization, and Calibration](Lecture-16.md) | Adaptation plan with normalization parity |
| 17 | [Offline and Online RL Post-Training for VLAs](Lecture-17.md) | RL objective audit and duration-aware credit assignment |
| 18 | [Training Infrastructure, Distributed Execution, and Reproducibility](Lecture-18.md) | Reproducible training/resume and resource record |
| 19 | [Closed-Loop Evaluation, Benchmarks, and Statistical Evidence](Lecture-19.md) | Paired evaluation with uncertainty and exclusions |
| 20 | [Integrating a VLA with Isaac Sim and Isaac Lab](Lecture-20.md) | Isaac task-to-policy adapter contract |
| 21 | [VLA Inference Optimization and Edge Deployment](Lecture-21.md) | Inference trace, candidate optimization, and parity plan |
| 22 | [Asynchronous Control, ROS 2, and Real-Time Action Chunking](Lecture-22.md) | Timestamped scheduler and fault fixtures |
| 23 | [Robustness, System Identification, and Sim-to-Real Transfer](Lecture-23.md) | Identification experiment and robustness matrix |
| 24 | [Deployment Boundaries, Recovery, and Human Intervention](Lecture-24.md) | Supervisor state machine and intervention analysis |
| 25 | [Physical Intelligence: from π0 to steerable generalists](Lecture-25.md) | PI lineage, conditioning, and artifact comparison |
| 26 | [NVIDIA GR00T and the Isaac engineering boundary](Lecture-26.md) | GR00T/Isaac compatibility and timing audit |
| 27 | [Gemini Robotics: reasoning, action, and access boundaries](Lecture-27.md) | Reasoning/action boundary and access-aware evaluation |
| 28 | [Open VLA systems: choose an artifact, not a headline](Lecture-28.md) | Executable open-model selection dossier |
| 29 | [Reasoning, memory, hierarchy, and world-action models](Lecture-29.md) | Memory/reasoning/world-model ablation design |
| 30 | [The October 2026 frontier: comparison and critical reading](Lecture-30.md) | Dated critical comparison of frontier evidence |
| 31 | [Capstone I: Train, Diagnose, and Integrate a Grounded Policy](Lecture-31.md) | Trained CPU grounding model and optional simulator adaptation |
| 32 | [Capstone II: A Professional VLA Research and Deployment Review](Lecture-32.md) | Reproducible professional project and oral defense |

</div>

## Study sequence and workload

Use a 16-week schedule with two lectures per week, followed by the associated lab. The full engineering route is approximately **180–240 hours**, depending on background and chosen capstones, excluding large-scale training and difficult hardware setup. Reading alone takes much less time but does not satisfy the practical assessment. Per-lecture estimates distinguish teaching, setup, and execution.

| Weeks | Module | Milestone |
|---|---|---|
| 1–2 | Lectures 01–04: foundations and historical turning points | Explain why language grounding, visuomotor control, and generalist pretraining are distinct problems |
| 3–4 | Lectures 05–08: datasets and representations | Validate geometry, schemas, visual tokens, and action representations |
| 5–6 | Lectures 09–12: imitation and generative action learning | Derive and diagnose BC, ACT, diffusion, and flow objectives |
| 7–9 | Lectures 13–18: data, adaptation, RL, and training systems | Produce a reproducible data/training design with held-out evaluation |
| 10–12 | Lectures 19–24: evaluation and deployment | Build contracts, timing fixtures, and an Isaac integration/robustness plan |
| 13–15 | Lectures 25–30: model families and current research | Audit a 2026 artifact and defend a protocol-specific comparison |
| 16 and project time | Lectures 31–32: capstones | Submit reproducible evidence and defend its scope |

For a faster role-specific route, retain the shared foundations and adjust the project emphasis:

- **Training/research:** 01–18, 19, selected 25–30, and both capstones; prioritize data, losses, adaptation, and ablations.
- **Inference/deployment:** 01, 04–08, 10–12, 14, 16, 19–24, 26, 28, 30–32; prioritize timing, memory, action parity, and controllers.
- **Robotics integration:** 01–06, 10, 13–14, 19–24, 26–27, 29, 31–32; pair with the Isaac engineering labs.

These routes reduce breadth; complete all 32 lectures for the full course. Do not skip a prerequisite solely because its topic is outside your intended job title.

## Learn through prediction, intervention, and explanation

Before each worked example, predict the result. Reproduce the reasoning or code, change one assumption, and retain a failure case. Answer the self-check without notes, then compare with the answer guidance. Revisit the key mechanism after two days and one week. Follow the [shared learning guide](../../../Learning-Guide.md) for experiment records and spaced retrieval.

The first CPU capstone trains a tiny nine-parameter grounding model with a fixed visual encoder. It makes gradients and language interventions inspectable; it is explicitly not a miniature claim of general robot intelligence. Later projects use a real released VLA only when its model, data, runtime, and controller contracts can be satisfied.

## Labs, hardware, and assessment

Start with the [laboratory guide](Lab-Guide.md). The supplied CPU exercises use the Python standard library and require no model downloads. They include a trainable synthetic policy, episode-safe chunks, action normalization, delayed-action eligibility, and paired task-level statistics. Tests run in curriculum CI.

GPU exercises depend on the selected model and environment. There is no single VRAM minimum for all VLAs. Measure simultaneous model, camera, simulator, workspace, and training-state memory, and use the selected upstream support matrix. The [Isaac lab guide](../Isaac%20Sim%20and%20Isaac%20Lab/Lab-Guide.md) separates supported simulation from reduced exercises. CPU success does not validate RTX simulation, checkpoint fine-tuning, or a physical controller.

Assessment combines eight module artifacts (40%), Capstone I (25%), Capstone II (25%), and an oral/code review (10%). Use the [professional project rubric](Lecture-32.md) for correctness, reasoning, measurement, reproduction, and transfer. Missing action-contract validation or unreported test leakage must be repaired regardless of aggregate score. This is a self-study professional curriculum, not an accredited credential.

### Eight assessed module artifacts

Each artifact contributes 5% of the course score. Use the same correctness/reasoning/measurement/reproduction/transfer rubric as the capstones, scaled to the assignment.

| Artifact | Lectures | Required evidence |
|---|---|---|
| 1. Historical and architectural explanation | 01–04 | Mechanism timeline, three architecture diagrams, one grounding counterexample |
| 2. Representation contract | 05–08 | Embodiment schema, frame/unit checks, action-codec reconstruction error |
| 3. Learning objectives | 09–12 | Derived losses, one checked gradient, multimodal-action failure analysis |
| 4. Data and adaptation plan | 13–16 | Collection protocol, causal transforms, split audit, normalizer/checkpoint manifest |
| 5. RL and training systems | 17–18 | Objective/likelihood audit, resume design, resource and duration-aware return calculation |
| 6. Evaluation and simulation boundary | 19–20 | Paired-statistics report and validated adapter specification; label unexecuted simulator work |
| 7. Deployment and robustness | 21–24 | Timing budget, fault fixtures, perturbation matrix, recovery state machine |
| 8. Frontier research review | 25–30 | Primary-source comparison with event dates, access, protocol, and reproduction limits |

## Research reading and current coverage

Read sources with a question: what is the problem, what changes in the mechanism, what is controlled, what evidence supports the claim, and what can actually be reproduced? The source dossiers provide reading order and limitations:

- [Foundations and history](Sources-Foundations.md)
- [Training, data, and RL](Sources-Training.md)
- [Evaluation, simulation, and deployment](Sources-Systems.md)
- [2026 model families and research](Sources-Frontier.md)
- [Dated frontier register](Frontier-Register.md)

The 2026 coverage includes verified work on π0.7, GR00T N1.7, Gemini Robotics 2, MolmoAct 2, LingBot-VLA 2.0, Helix 2.5, world-action models, and papers revised in October before the cutoff. An inclusion records evidence, not endorsement or a universal ranking. Access, model-card terms, and benchmark configurations can differ between versions.

## Connections to the roadmap

Continue with [VLA Optimization and Action-Parity Harness](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) for specialist inference work, [Deep RL](../Deep%20RL%20for%20Robot%20Learning/README.md) for deeper control learning, and [Isaac Sim and Isaac Lab](../Isaac%20Sim%20and%20Isaac%20Lab/README.md) for simulator engineering. The [DeepSeek NVFP4 course](../../Track%20G%20-%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/Part%205%20-%20DeepSeek%20V4.1%20Flash%20NVFP4/README.md) contributes checkpoint and runtime methods; language-model deployment and robot action deployment still require different output and evaluation contracts.


## Carry this course into the physical-AI system

**Start from:** The task/observation contract and a released, usable policy artifact.

**Build and challenge:** Follow history and architecture into action representation, data, training, evaluation, and deployment. Trace a complete action chunk through normalization, frame conversion, expiry, and controller consumption.

**Pass forward:** A workload dossier and closed-loop reference that constrain the future accelerator. Continue with [the connected course](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).
