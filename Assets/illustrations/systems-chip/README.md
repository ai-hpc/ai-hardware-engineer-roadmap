# Systems-to-chip teaching illustration

`model-to-soc.svg` is an original vector diagram created for this curriculum on 2026-10-14. `model-to-soc.png` is its 1800 × 1260 PNG export, rendered with CairoSVG. No third-party image was copied or traced. Both assets follow the repository license.

The diagram is conceptual, not a chip floorplan, wiring guide, verified implementation, or claim that a wireless RF block is integrated. It separates model/compiler contracts, the Linux/accelerator/memory path, and the controller boundary. Technical context and primary references appear in the AI-chip guide.

To regenerate the PNG with CairoSVG installed:

```python
import cairosvg
cairosvg.svg2png(url="Assets/illustrations/systems-chip/model-to-soc.svg",
                write_to="Assets/illustrations/systems-chip/model-to-soc.png",
                output_width=1800, output_height=1260)
```
