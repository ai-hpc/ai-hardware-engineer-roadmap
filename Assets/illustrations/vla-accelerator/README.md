# Original VLA accelerator teaching figures

These figures are original analytical visualizations for the [programmable VLA accelerator course](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Programmable%20VLA%20Inference%20Accelerator/README.md). They contain no copied paper figures, product imagery, or model performance results. They follow this repository's license.

| Figure | Purpose | Downloads |
|---|---|---|
| Partial residency | Distinguish cold and warm weight traffic under a reserved SRAM budget | [SVG](residency.svg) · [PNG](residency.png) |
| Legal scheduling boundaries | Compare whole-job blocking against EDF at legal tile boundaries | [SVG](scheduling.svg) · [PNG](scheduling.png) |

Values come directly from the course's CPU lab functions. The residency figure assumes packed payloads, ideal retention, and 120 decimal GB/s sustained bandwidth; it omits activation traffic, quantization metadata, compute and power. The scheduling figure uses one abstract resource, independent jobs, and zero dispatch overhead. Neither predicts a real chip's performance.

To reproduce from the repository root with Matplotlib installed:

```bash
MPLCONFIGDIR=/tmp/vla-plots python3 Assets/illustrations/vla-accelerator/render.py
```

[Rendering source](render.py). Matplotlib is required only for figure generation, not for the CPU labs.
