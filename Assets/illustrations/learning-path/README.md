# Learning-path illustrations

Original technical diagrams created for this curriculum on October 14, 2026. They are released under the repository's [MIT license](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/blob/main/LICENSE). No third-party photographs, search-result thumbnails, logos, or paper figures were copied into these assets.

Each figure has an editable SVG for the course website and a PNG for notes/slides. Labels and worked examples are part of the lesson rather than decorative artwork.

| Figure | Purpose | Files |
|---|---|---|
| Mission gates | Connect model, robot, compiler, accelerator, Linux, and wireless deliverables | [SVG](mission-gates.svg) · [PNG](mission-gates.png) |
| Robot loop | Distinguish policy, freshness checks, local control, and optional network planning | [SVG](robot-loop.svg) · [PNG](robot-loop.png) |
| Critical path | Explain overlap using a fully specified two-tile analytical example | [SVG](critical-path.svg) · [PNG](critical-path.png) |
| NVFP4 storage | Distinguish payload, block scales, tensor scale, and full allocation | [SVG](nvfp4-storage.svg) · [PNG](nvfp4-storage.png) |

The first three are original curriculum explanations. The fourth illustrates the common one-dimensional layout described by [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html); it is not copied from a vendor figure and does not describe every training recipe or checkpoint packing convention.

Regenerate from the repository root using Python and Matplotlib:

```bash
MPLCONFIGDIR=/tmp/roadmap-mpl python3 Assets/illustrations/learning-path/render.py
```

The source sets SVG text to remain text and PNG resolution to 180 dpi. A font or Matplotlib change may change layout; inspect all four outputs after regeneration. Inline Markdown supplies descriptive alt text and a lesson question. The diagrams use labels as well as color.
