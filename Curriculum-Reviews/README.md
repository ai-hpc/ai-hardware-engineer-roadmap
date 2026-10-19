# Course-by-course review — October 14, 2026

**Historical review.** See the [October 18–19 update](../Curriculum-Review.md) for the later teaching, technical and execution work.

This pass reviewed **all 158 course entry pages** in the starting inventory of **598 source Markdown pages**, then corrected selected lectures and examples in depth. Entry pages include phase/track hubs and lab indexes; 158 is not a count of independent lecture series. Every entry now has a concrete connection to the goal of building a VLA-focused physical-AI system and its compiler, accelerator, Linux, and wireless stack.

The [Physical AI Design Studio](../Physical-AI-Design-Studio.md) is the resulting learning path. Its nine gates connect courses through shared contracts, reference vectors, experiments, and system evidence.

## Coverage and review depth

| Area | Inventoried entries | Detailed ledger |
|---|---:|---|
| Foundations, embedded systems, interview preparation | 32 | [Course findings](foundations-embedded.md); also repairs the OS demo |
| AI models, robotics, ML systems | 39 | [Course findings](models-robotics-mlsys.md) |
| FPGA, Jetson, inference/compiler deployment, edge AI | 55 | [Course findings](deployment-edge.md) |
| GPU infrastructure, HPC, autonomous systems, chip design | 32 | [Course findings](systems-chip.md) |
| **Total** | **158** | Each ledger links individual entries and states the handoff or correction |

This is an entry-by-entry curriculum review with targeted technical corrections. It does not certify every statement or command in all 598 original pages. Large historical guides, external repositories, downloadable models, and device-specific labs still require revision pins and the execution checks described in their courses. The ledgers distinguish structural review, sampled technical review, rewritten material, and executed portable checks.

## Material improvements

| Issue | Correction and learning consequence |
|---|---|
| Courses ended as disconnected topic lists | Each entry now passes a specific artifact to the next course; the studio carries one task across the stack |
| Hardware features were attributed to the wrong Jetson SKU | Corrected Orin Nano DLA, encoder, memory, and I/O assumptions so labs target real capabilities |
| Numerical formats and hardware instructions were conflated | Corrected TF32 storage, integer INT4 arithmetic, NVFP4 metadata, and model/runtime compatibility boundaries |
| Performance claims lacked evidence or used wrong units | Removed unsupported benchmark promises; corrected roofline units, timing overlap, scaling efficiency, and speculative speedup inference |
| Framework/calibration examples taught misleading behavior | Corrected ReLU backward convention, gradient use, image bandwidth, Kalibr environment boundaries, and geometric assumptions |
| OS syllabus links described different topics from their target lectures | Rebuilt the sequence around the actual lessons and repaired a demo compilation failure |
| Model and hardware case studies mixed generations | Reworked Gemma, Qwen, and Blackwell material around named configurations and support checks |
| Robot inference was treated as a single rate | Separated observation age, policy updates, action chunks, controller ticks, and network supervision |
| Interview notes contained invented personal wins | Replaced them with explainable calculations and evidence-backed answer structures |

## Original teaching figures

Added **eight original diagrams**, each available as **SVG and PNG**, with explanatory captions or exercises in the courses. The assets are local, so rendering does not depend on an external image host. Provenance and generation/editing instructions accompany them.

- [Learning path, robot loop, critical-path timing, NVFP4 storage](../Assets/illustrations/learning-path/README.md)
- [Foundations and embedded figures](../Assets/illustrations/foundations-embedded/README.md)
- [Deployment memory ownership](../Assets/illustrations/deployment-edge/README.md)
- [Model-to-SoC design](../Assets/illustrations/systems-chip/README.md)

## Validation boundary

Portable validation covers Markdown targets, documentation build/rendering, existing CPU lab suites, selected corrected numerical examples, and the OS demo build/run. GPU throughput, full-model quality, Isaac execution, FPGA timing, wireless behavior, and physical-robot performance are **not measured by this review**. A fixture or analytical result is labeled accordingly in the teaching material.

**Results:** all 158 inventoried entries changed; local links resolve; the strict MkDocs build passes; all **35 existing portable tests** pass. Additional executed checks cover scalar autograd and finite differences, storage/timing arithmetic, Gemma metadata and NumPy exercises, online-attention numerics, and the repaired OS demo. Browser checks confirm the studio at desktop/tablet/mobile widths, all eight PNG downloads, loaded inline SVGs, and no failed local assets or JavaScript exceptions on the inspected pages. These checks validate the edited teaching surface, not the unexecuted hardware claims above.

See [the prior review](../Curriculum-Review-2026-10-13-and-14.md) for the October 13 expansion and its validation history. This page records the separate October 14 course-by-course pass.
