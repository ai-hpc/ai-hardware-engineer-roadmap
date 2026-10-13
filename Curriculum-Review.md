# Curriculum review — 2026-10-13

This update reviews repository structure across the roadmap and improves the learning and assessment layer at every existing phase/course entry page. The focused technical expansion covers DeepSeek V4.1 Flash NVFP4 inference and NVIDIA Isaac Sim / Isaac Lab.

## Scope and evidence

The initial inventory covered **534 tracked source Markdown pages**, including repository and project documentation. It screened structure, topic coverage, course entry points, and local links. Technical close reading concentrated on the shared authoring standard, phase/track guides, inference course, and Isaac course. This is not a claim that every command in all historical lectures was rerun or that every external resource was revalidated.

The existing uncommitted change in Isaac Lecture 01 was preserved. Generated `docs/` and `_site/` content is rebuilt from the source pages rather than edited directly.

## Findings and changes

| Finding | Learning consequence | Change |
|---|---|---|
| Many guides had long concept/resource lists without an explicit assessment | Reading could be mistaken for mastery | Added topic-specific entry diagnostics, baseline labs, variations, required evidence, and exit criteria to 126 course entry pages |
| Existing strong courses used different completion expectations | Learners could not compare progress across tracks | Linked the shared learning workflow from 29 additional course entry pages; 155 existing entries addressed in total |
| Hardware access requirements were mixed with course completion | Learners without a large node could stall or overclaim results | Defined analysis, fixture, simulation, and measured deployment milestones |
| Inference course lacked a V4.1 Flash checkpoint-specific path | Generic MoE/FP4 knowledge did not address actual serialization and compatibility | Added ten lectures and CPU tools for metadata, FP4 representation, and request metrics |
| NVFP4 naming could imply uniform format or automatic memory savings | Incorrect capacity plans or loader assumptions | Distinguished upstream FP4 experts, two community conversions, architecture support, and exact loader support |
| Isaac course needed a progressive validation sequence | Training and deployment failures could hide frame/reset/timing bugs | Added eight lectures, a lab sequence, and executable CPU contract checks |
| Eight-GiB simulation language could read as a supported baseline | Installation and sensor labs could fail before learning began | Clarified supported RTX requirements versus experimental reduced exercises |
| Link checker scanned code as if it were prose and skipped untracked files | False failures and missed new-page errors | Excluded fenced/inline code, added new-file and angle-link coverage, and regression tests |

## Coverage by curriculum area

| Area | Added or reinforced assessment focus |
|---|---|
| Phase 1 | RTL/testbench behavior, memory/cache reasoning, parallel correctness, synchronization and timing |
| Phase 2 | Board/interface contracts, firmware queues, connectivity recovery, reproducible Linux images, product setup/error behavior |
| Phase 3 | Numerical references, held-out evaluation, calibration and timestamps, model/application failure cases |
| Phase 4A | Constraints, AXI/DMA integrity, timing/CDC, HLS tradeoffs, driver ownership |
| Phase 4B | BSP/deployment identity, sensor/media timing, shared memory, thermal behavior, bounded recovery |
| Phase 4C | Rewrite legality, kernel reference checks, compiler traces, quantization quality, serving latency |
| Phase 5A/B | Topology, collectives, placement, scaling, I/O, library setup versus execution |
| Phase 5C | Edge quality/latency/memory tradeoffs and sustained operation |
| Phase 5D | Frames, assets, partial resets, reward/done semantics, evaluation uncertainty, performance, ROS timing, transfer |
| Phase 5E | Recorded/simulated scenarios, process boundaries, sensor timing, requirements-to-test evidence |
| Phase 5F | Workload-derived budgets, reference models, verification, constrained architecture decisions |
| Phase 5G | Checkpoint identity, numerical correctness, execution traces, per-rank capacity, quality-preserving useful throughput |
| Phase 6 | Evidence-backed explanations, calculations, design changes under a new constraint |

The practice additions use topic-specific assignments and preserve existing lectures. A common format makes expectations visible, but cannot replace a worked explanation of the underlying subject. New inference and Isaac lectures include worked reasoning, failure cases, and self-check guidance to make that teaching concrete.

## Technical source boundaries

- The [official DeepSeek V4.1 Flash model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), its [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json), and [technical report](https://arxiv.org/html/2609.19969v1) ground the architecture lessons.
- [s-zaizen's conversion](https://huggingface.co/s-zaizen/DeepSeek-V4.1-Flash-NVFP4) is the default community-checkpoint case. Its card does not claim validated serving commands or quality/throughput benchmarks. [LibertAI's conversion](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4) is a separate format/quality comparison.
- [vLLM's V4.1 module](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/) establishes architecture implementation evidence; each checkpoint/runtime/backend combination still needs verification.
- [Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html) and [Isaac Lab v3.0.0-EA documentation](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html) ground the simulator release family. Learners must save immutable commits and resolved configurations for their experiments.

## Reproduce the portable checks

From the repository root:

```bash
python3 scripts/audit_curriculum.py
python3 scripts/check_links.py
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 -m unittest discover -s 'Phase 5 - Advanced Topics and Specialization/Track G - ML Systems Engineering/AI Inference Engineer 2026/Part 5 - DeepSeek V4.1 Flash NVFP4/labs' -p 'test_*.py'
python3 -m unittest discover -s 'Phase 5 - Advanced Topics and Specialization/Track D - Robotics/Isaac Sim and Isaac Lab/labs' -p 'test_*.py'
STRICT=1 bash scripts/build_docs_local.sh
```

The audit script emits review signals and can export the complete inventory with `--json`. Keyword presence is not a quality grade. Link checks validate local file targets; MkDocs also reports anchor/navigation issues. Neither proves external URL availability.

**Local validation:** 23 portable tests passed, all local file targets resolved, all four CPU demos ran, and the strict MkDocs build passed. Hardware execution was not part of these checks.

The new CPU tests validate metric arithmetic, corrupt/invalid inputs, packing, reset isolation, coordinate checks, timing definitions, uncertainty, and command freshness. They do **not** establish full-model inference, NVFP4 kernel performance, simulator execution, trained-policy success, or real-robot transfer. Those require the hardware-specific experiments described in the courses.

[Learning guide](Learning-Guide.md) · [Authoring standard](Curriculum-Authoring-Guide.md) · [Roadmap](README.md)

## Professional VLA expansion — October 13, 2026

Added [VLA Foundations to Frontier 2026](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Foundations%20to%20Frontier%202026/README.md): 32 detailed lectures covering history, grounding, action representations, imitation/diffusion/flow learning, data engineering, adaptation, RL, training infrastructure, evaluation, Isaac integration, asynchronous control, robustness, and current research. Four source dossiers and a dated frontier register distinguish papers from available artifacts and restricted access.

The course adds a 16-week study path, notation/glossary reference, worked examples, negative controls, answered self-checks, and two capstones. Its CPU labs train an inspectable nine-parameter grounding model and check action normalization, episode masks, split groups, scheduling, and paired task statistics. These fixtures do not establish pretrained-model, simulator, or physical-robot performance. New VLA tests are included in curriculum CI; the 23-test result above records the earlier roadmap review, before this expansion.

**VLA expansion validation:** 35 portable tests pass across the link checker, inference, Isaac, and VLA suites. All four VLA lab-guide shell blocks execute successfully, local Markdown links resolve, and the strict documentation build renders all 32 lectures and their downloadable Python helpers. No GPU, simulator, API, or physical-robot execution was performed.
