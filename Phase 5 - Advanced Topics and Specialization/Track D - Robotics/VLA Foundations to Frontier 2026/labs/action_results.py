"""Summarize saved action experiments; plotting optionally needs matplotlib."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    print("Either command: ranges across training seeds, not confidence intervals")
    for method in ("flow", "ddim"):
        for count in (8, 32, 64):
            rows = [r for r in result["metrics"] if r["method"] == method
                    and r["evaluations"] == count and r["label"] == "either"]
            cdf = [r["cdf_distance"] for r in rows]
            gap = [r["between_modes_fraction"] for r in rows]
            print(f"{method:4} {count:2} evaluations: CDF {min(cdf):.3f}–{max(cdf):.3f}; "
                  f"between modes {min(gap):.1%}–{max(gap):.1%}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        colors = {"flow": "#1468a8", "ddim": "#b65b0b"}
        for ax, key, title in zip(axes, ("cdf_distance", "between_modes_fraction"),
                                  ("Distance from target distribution", "Samples between the two modes")):
            for method in ("flow", "ddim"):
                seeds = sorted({r["training_seed"] for r in result["metrics"]})
                for index, seed in enumerate(seeds):
                    rows = [r for r in result["metrics"] if r["method"] == method
                            and r["label"] == "either" and r["training_seed"] == seed]
                    ax.plot([r["evaluations"] for r in rows], [r[key] for r in rows],
                            color=colors[method], marker="o", alpha=0.65,
                            label=f"learned {method}: {len(seeds)} seeds" if index == 0 else None)
                rows = [r for r in result["oracle_metrics"] if r["method"] == method and r["label"] == "either"]
                ax.plot([r["evaluations"] for r in rows], [r[key] for r in rows],
                        linestyle="--", color=colors[method], label=f"analytic {method}")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Network / oracle field evaluations")
            ax.set_xticks([8, 32, 64])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Empirical CDF distance (smaller is closer)")
        axes[1].set_ylabel("Fraction with |action| < 0.5")
        axes[0].legend(fontsize=7)
        fig.suptitle("Synthetic 'either' command · 2,048 samples per run · CPU", fontsize=12)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160, metadata={"Creator": "Matplotlib course action experiment"})
        print(f"Plot: {args.plot}; matplotlib {matplotlib.__version__}")


if __name__ == "__main__":
    main()
