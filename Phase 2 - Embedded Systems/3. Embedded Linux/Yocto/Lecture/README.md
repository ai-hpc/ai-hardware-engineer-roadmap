# Yocto — step-by-step lectures

Work through these in order. Each file teaches one engineering boundary and applies it to the same small observation-age checker. Reading and local exercises fit a study session; first image/SDK builds can take hours and should be scheduled separately.

Start with the [runnable native package and supplied layer](../Labs/README.md). Choose a native, metadata, or full-image route in Lecture 1, and keep the claims attached to the work you actually ran. The baseline is pinned Scarthgap 5.0.20; Lecture 2 compares it with the current Wrynose setup.

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

## Reuse the same package across all sixteen sessions

L1–5 establish the environment and a pinned baseline; L6–8 package the observation-age checker in a product layer and image; L9 ties boot artifacts to hardware; L10 uses the matching SDK; L11 records an intentional failure and repair. L12–13 add manifests and rebuild evidence. The capstone in L14 supplies exact target acceptance cases and optional service/hardware/release extensions; L15–16 support retrieval and maintenance.

For every session retain the diff, command, output artifact, and target observation. A list of concepts or a successful BitBake exit alone cannot prove that the right package reached the target.

**Carry forward:** A continuous product-layer history ending in an image plus matching SDK. Continue with [the next course](../../Jetson%20ESP-Hosted%20Host%20Code/Guide.md) and reuse that evidence.
