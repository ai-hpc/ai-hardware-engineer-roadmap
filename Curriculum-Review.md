# Curriculum review — October 18–19, 2026

The course now has a guided beginner route, substantially revised lessons across all six phases, and runnable experiments connecting AI training to robot behavior and hardware execution. Start with [your first robot and AI experiments](Start-Here.md), use the [beginner reading aid](Beginner-Reading-Aid.md) for units and notation, then follow the [learning guide](Learning-Guide.md) and [Physical AI Design Studio](Physical-AI-Design-Studio.md).

This review prioritizes explanations that a learner can reproduce: a worked calculation, an executable baseline, a deliberate failure or variation, and a result with a clear interpretation. A smaller passing test is not presented as a measurement of a GPU, simulator or physical robot.

## Scope and evidence

The starting inventory contained **623 source Markdown pages**; the revised inventory contains **748**, including **194 course-entry pages**. The review covers the entire curriculum structure and learner entry points, with detailed rewrites and execution concentrated on robotics, AI, compilers, embedded systems and the most consequential technical defects. The [source inventory](Curriculum-Reviews/2026-10-19-inventory.json) records the resulting pages; course-entry counts include phase/track hubs and lab indexes, not only independent lecture series.

This is a substantial course improvement, not a certification that every historical statement, external repository or hardware command has been independently executed. Current claims are tied to dated primary sources and, where practical, immutable revisions or downloaded-file hashes. Existing October 13–14 reviews remain available as [historical expansion evidence](Curriculum-Review-2026-10-13-and-14.md) and [the earlier course-by-course ledger](Curriculum-Reviews/README.md).

## A connected robotics and AI route

| Stage | What the learner does | What the result establishes |
|---|---|---|
| [First experiments](Start-Here.md) | Change control gain and observation delay on a one-dimensional cart | Why timestep, units and information age matter |
| [Train a small policy](Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-01-CPU-Policy.md) | Train, save, reload and evaluate a CPU policy on held-out episodes | Actual parameter learning and prediction/control comparisons on the declared fixture |
| [Timed robot loop](Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-02-Timed-Robot-Loop.md) | Put that trained policy behind processing and transport queues | How FIFO backlog, stale commands, deadlines and fallback alter closed-loop behavior |
| [Actual CPU physics engine](Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-03-CPU-Physics.md) | Reuse the saved policy in MuJoCo across mass, force-limit, damping and timestep changes | Actual engine execution; matching-model parity and model-mismatch behavior on 72 episodes, with simulator auto-reset warnings rejected |
| [Robot dynamics and task execution](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Robot%20Dynamics%20and%20Task%20Execution/Lab-Guide.md) | Work through dynamics, LQR/MPC, trajectory/collision and grasp/task checks | Differences between numerical control exercises and task success; an actual MuJoCo pendulum crosscheck separates point-mass assumptions, finite-body inertia and integrator damping |
| [Learn diffusion and flow action heads](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/Action-Generation-Lab.md) | Train six small networks, compare three sampling budgets, and reload saved checkpoints | Actual learned sampling behavior on a known scalar distribution; no language, image or robot capability is implied |
| [Action-parity lab](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/CPU-Action-Parity-Lab.md) | Compare reference and changed policy paths using actual rollouts | Why arithmetic parity, timing, model use and task outcomes need separate checks |
| [Trace a trained model into a hardware budget](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Programmable%20VLA%20Inference%20Accelerator/Model-to-Hardware-Lab.md) | Export the trained flow network, check changed inputs after reload, and profile CPU calls | Measured operator shapes/calls plus separately derived operation counts, parameter bytes and hypothetical tile occupancy |
| [Isaac engineering labs](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Isaac%20Sim%20and%20Isaac%20Lab/Lab-Guide.md) | Inspect reset, frames, asset, action and episode semantics before the simulator milestone | A reproducible preparation route with explicit simulator/hardware requirements |

The timed-loop experiment records accepted commands **and commands actually applied**. A run can accept a command at the final deadline and never apply it during a control interval. The course now makes that distinction explicit. The action-parity exercise also rejects an apparent task success that never used the policy. Its small fixed episode suite does not establish population-level noninferiority or real-robot safety.

The action-generation experiment retains all three training seeds and both methods, including remaining distribution error. An oracle with known conditional expectations provides a separate reference. More integration steps change both the generated distribution and the work count; the accelerator lesson carries the trained checkpoint across that boundary instead of inventing a workload. Export and reload each pass 12 changed-input probes at batch sizes 1 and 7.

The [PyTorch stop-policy lab](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md), [camera/time-fusion lab](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Camera-Time-Fusion-Lab/Guide.md), VLA data exercises and ROS 2 integration provide additional executable steps. The beginner sequence states when its action changes from velocity to acceleration and works through the changed equations.

## Improvements across the curriculum

| Area | Substantive teaching and technical changes |
|---|---|
| Phase 1: foundations | Worked numbers/shapes/bytes, RTL simulations, native SIMD and RISC-V vector examples, real OpenMP/oneTBB/OpenCL work, corrected synchronization and memory ownership |
| Phase 2: embedded systems | Framed telemetry, observation freshness, a buildable Yocto application/layer, OpenThread simulation, circuit calculations and product-state/error behavior |
| Phase 3: AI | Actual training/evaluation/export, grouped data splits, finite-difference checks, perception/time fusion, and a local retrieval/agent workflow with real framework integration |
| Phase 4A: FPGA | Runnable RTL testbenches and failure cases; clearer boundaries between simulation, synthesis, timing closure and board evidence |
| Phase 4B: Jetson | Updated platform/software boundaries; corrected CUDA, camera, media, real-time, container, device-tree, CAN, update, factory and dashboard lessons |
| Phase 4C: inference/compiler | Executable CPU compiler and attention paths; corrected masks, gradients, memory accounting, quantization and backend compatibility |
| Phase 5A/B: GPU infrastructure and HPC | Corrected topology/collective accounting, actual CPU distributed jobs, long-context MoE training exercises, B200/checkpoint capacity planning and source-pinned deployment boundaries |
| Phase 5C/D/E: edge, robotics and autonomous systems | Connected learned policies, scheduling, ROS/industrial interfaces, planning/dynamics, simulator preparation and system-level task evidence |
| Phase 5F: chip design | Workload-derived budgets, executable vector examples, portable accelerator contracts and the distinction between analytical budgets and hardware results |
| Phase 5G: ML systems | Reworked model, compiler, attention, hybrid-state, quantization and speculative-decoding courses around actual source/configuration identities and reproducible CPU checks |
| Phase 6: interview preparation | Worked calculations, troubleshooting and system-design practice that use the learner's actual evidence |

The repository's [Jetson runtime project](Projects/jetson-llm-runtime/README.md) also received source-level repairs and documentation. Its [attention/layout lab](Projects/jetson-llm-runtime/labs/Attention-Layout-Lab.md) checks token-major KV indexing and decode scratch reuse. CUDA translation-unit compilation is recorded separately from execution on a GPU.

## Important AI and systems corrections

- [FlashAttention](Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/CPU-Attention-Lab.md): dense/tiled outputs and gradients, rectangular masking, grouped heads and dtype behavior are checked; I/O reduction is distinguished from quadratic attention arithmetic and device speed.
- [Hybrid recurrent/attention models](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/GLM-5.3-Flash%20Architecture%20Mastery/CPU-Hybrid-State-Lab.md): actual pinned upstream CPU functions are compared with references; recurrent state, token-indexed caches, pooling visibility and rollback are distinct. Fixed recurrent state does not make every hybrid model's total memory constant.
- [Quantization](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/Hardware-Aware%20LLM%20Quantization/CPU-Quantization-Lab.md): real packing, scales, calibration, saved artifacts and shifted inputs replace unsupported quality/speed claims. A better local calibration loss can worsen whole-model metrics. Quantizing a drafter is distinguished from quantizing its target.
- [Long-context MoE training](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Long-Context-MoE-Foundation-Training/CPU-Training-Lab.md): byte units, causal/full work, expert grids, drop denominators and real router gradients are explicit. Actual small-model training and two-process dispatch replace incomplete “run this” sketches and guaranteed benchmark outcomes.
- [TVM](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/TVM%20Deep%20Dives/labs/README.md): current pinned CPU compilation/execution and export are exercised. A failed tuning trial with no valid timing is retained as a failure, not described as an optimization result.
- [Inference engine source lab](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/Part%204%20-%20Optimizing%20a%20Real%20Engine/CPU-Source-Lab.md): upstream native/Python tests and approximation counterexamples are separated from full-model/GPU evidence.
- [Provider tool protocols](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Lectures/Provider-Contract-Lab.md): actual Anthropic/OpenAI SDKs serialize requests and parse local HTTP fixtures. Twenty SDK tests complement eight portable contract tests. The lessons distinguish continuation, incomplete output, refusal, tool-result correlation and bounded execution; no hosted model output is presented as measured.
- [Concurrency](Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/OpenMP%20and%20OneTBB/Guide.md): compiled counterexamples correct private-value lifetime, `lastprivate`, reduction/task semantics and work isolation. oneTBB isolation restricts task scheduling; it is not mutual exclusion. The example demonstrates two isolated regions overlapping without creating an intentional data race.
- [Attestation and channel binding](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Confidential-Computing-Attestation/CPU-Binding-Lab.md): actual CPU cryptography checks a bounded binding/replay exercise, including concurrent replay rejection. Hardware measurements, verifier policy, application identity and channel binding remain separate claims.
- [Operating-system tracing](Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/labs/BPF-Build-Lab.md): real BPF object, skeleton and userspace-reader compilation teaches the build boundary. No kernel attach or privileged execution is inferred from the successful compile.

## Reproduce the checks

From the repository root:

```bash
python3 scripts/audit_curriculum.py
python3 scripts/check_links.py
python3 scripts/run_portable_checks.py --list
python3 scripts/run_portable_checks.py --json /tmp/course-checks.json
python3 -m pip install -r requirements-docs.txt
STRICT=1 bash scripts/build_docs_local.sh
```

Use the lab-specific instructions for optional frameworks and native tools. The check runner isolates suites in separate processes, includes new source files, works in a source archive without Git, and reports skipped tests separately. A machine without a compiler, runtime or optional Python package may skip checks; that is not evidence those checks passed. CI installs the documented baseline native dependencies before running its portable suites and strict documentation build.

The [retained portable run](Curriculum-Reviews/2026-10-19-portable-checks.json) records **749 executed test methods with no failures or errors**, across 72 passing suites. It separately reports 30 skipped methods and one skipped setup group; one of the 73 suites requires its optional TVM environment and was wholly skipped in this baseline run. TVM was executed separately in its pinned environment. These counts describe this machine and run, not a promise that every optional dependency is installed by default. Additional lab manifests record executions in isolated PyTorch, TVM, tinygrad, ROS, compiler, CPU physics/scene tools and native-library environments. Review the relevant manifest before citing a result. CPU OpenCL, RTL simulation, cross-compilation and CUDA compilation each demonstrate different things.

The [validation summary](Curriculum-Reviews/2026-10-19-validation-summary.json) separates source checks, browser behavior and archive reproducibility. All **748 source pages** passed the 390-pixel phone review; **25 selected pages passed at 320, 390 and 1440 pixels**. All **2,021 formulas on 96 pages** rendered with external requests blocked, using the locally packaged MathJax renderer and fonts. Search-result navigation, keyboard search, the mobile menu, diagrams and downloadable helpers were checked separately.

The source-archive review uses directories containing spaces, no Git metadata and an unrelated working directory. It verifies the portable test runner, strict documentation build, and byte-identical publication of lesson assets, fonts and licenses. A separate syntax scan parsed **988 Python/shell examples**; parsing does not establish that every command was executed. These checks do not guarantee that external links remain available forever.

The career guide now emphasizes [role responsibilities and portfolio evidence](Roles%20and%20Market%20Analysis.md). Unsupported niche salary tables, remote-work percentages and hiring-count forecasts have been removed. Its dated occupational sources and live employer pages have explicitly different scopes.

## Remaining hardware milestones

The course provides instructions and completion criteria for GPU execution, full-model quality/performance, Isaac scenes and training, FPGA timing/board work, Jetson deployment, physical robots and custom silicon. Those milestones still require their stated equipment and environments. CPU fixtures, source inspection and successful compilation do not substitute for them.

Use the exercise's own evidence boundary when reporting completion: **calculated**, **executed on CPU**, **compiled for a target**, **simulated**, or **measured on the named hardware**. This makes the roadmap practical for learners with different resources while keeping the conclusions defensible.
