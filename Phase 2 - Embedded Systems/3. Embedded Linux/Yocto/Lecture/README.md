# Yocto — step-by-step lectures

Work through these in order. Each file is one session: read it, do the lab at the end, then move on.

**Hub:** [Yocto course guide](../Guide.md)

| Lecture | Topic |
|---------|--------|
| [Lecture-01](Lecture-01.md) | How to use this course |
| [Lecture-02](Lecture-02.md) | Module 0 — Prerequisites and host setup |
| [Lecture-03](Lecture-03.md) | Module 1 — What Yocto is (and is not) ([Lab 1 example](Lab-01-Worked-Example-Edge-AI-Camera.md)) |
| [Lecture-04](Lecture-04.md) | Module 2 — Architecture in one picture |
| [Lecture-05](Lecture-05.md) | Module 3 — First successful build |
| [Lecture-06](Lecture-06.md) | Module 4 — Recipes: the atoms of the system |
| [Lecture-07](Lecture-07.md) | Module 5 — Layers: how metadata stays maintainable |
| [Lecture-08](Lecture-08.md) | Module 6 — Images, packages, and features |
| [Lecture-09](Lecture-09.md) | Module 7 — Kernel, bootloader, device tree |
| [Lecture-10](Lecture-10.md) | Module 8 — SDK and application workflow |
| [Lecture-11](Lecture-11.md) | Module 9 — Debugging builds like an engineer |
| [Lecture-12](Lecture-12.md) | Module 10 — Licenses, compliance, supply-chain |
| [Lecture-13](Lecture-13.md) | Module 11 — Performance, caching, and CI |
| [Lecture-14](Lecture-14.md) | Capstone projects |
| [Lecture-15](Lecture-15.md) | Glossary and quick reference |
| [Lecture-16](Lecture-16.md) | Further reading |

## Practice and assessment

**Entry diagnostic (prerequisite check):** Explain recipe, layer, machine, distribution, image, and task signatures. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Add a small package to an image and rebuild from pinned layers and configuration.
2. **Guided variation:** Change one recipe input and identify which tasks rebuild; verify package contents on the target or QEMU.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Layer revisions, build/task logs, image manifest, size delta, and runtime smoke test. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
