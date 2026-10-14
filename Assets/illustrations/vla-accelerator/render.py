#!/usr/bin/env python3
"""Render original teaching figures directly from the CPU design-lab model.

Requires matplotlib; the course labs themselves use only the standard library.
Run from any directory: MPLCONFIGDIR=/tmp/vla-plots python3 render.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "Phase 5 - Advanced Topics and Specialization" /
                     "Track F - AI Chip Design" /
                     "Programmable VLA Inference Accelerator" / "labs"))
from design_lab import Job, memory_fixture, schedule

BLUE, TEAL, RED, INK = "#2563eb", "#0d9488", "#b4233c", "#172b4d"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "svg.fonttype": "none",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(HERE / f"{name}.svg", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def residency():
    sram = [16, 32, 64]
    cold = [memory_fixture(s, 8)["weight_bandwidth_floor_ms"] for s in sram]
    warm = [memory_fixture(s, 8, cold=False)["weight_bandwidth_floor_ms"] for s in sram]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(bottom=.23, top=.79, left=.11, right=.97)
    fig.suptitle("SRAM helps only the bytes that stay resident", x=.11, ha="left",
                 y=.97, fontsize=19, weight="bold")
    fig.text(.11, .88, "Analytical fixture · weight traffic only · sustained 120 GB/s", fontsize=11)
    ax.axhline(37.5, color="#64748b", linestyle=":", label="No retained weights: 37.5 ms")
    ax.plot(sram, cold, "o-", color=BLUE, linewidth=2.5, label="Cold: includes retained-subset fill")
    ax.plot(sram, warm, "s--", color=TEAL, linewidth=2, label="Warm: prior fill excluded")
    ax.annotate(f"{cold[-1]:.2f} ms", (64, cold[-1]), xytext=(-8, 10),
                textcoords="offset points", color=BLUE, ha="right", weight="bold")
    ax.annotate(f"{warm[-1]:.2f} ms", (64, warm[-1]), xytext=(-8, -20),
                textcoords="offset points", color=TEAL, ha="right", weight="bold")
    ax.set(xlabel="Total SRAM (MiB); 8 MiB reserved for other uses",
           ylabel="Weight-bandwidth floor (ms)", xticks=sram,
           xlim=(13, 67), ylim=(31.8, 38.4))
    ax.grid(axis="y", alpha=.16)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    fig.text(.11, .07, "3B 4-bit backbone read once + 300M 8-bit expert read ten times.\n"
             "Ideal partial residency; excludes scales, activations, compute, dispatch, and power.",
             fontsize=10, linespacing=1.6)
    save(fig, "residency")


def scheduling():
    jobs = [Job("vision", 0, 18, 40), Job("action", 2, 3, 8)]
    schedules = [schedule(jobs), schedule(jobs, 3)]
    fig, ax = plt.subplots(figsize=(10, 5.1))
    fig.subplots_adjust(bottom=.28, top=.72, left=.2, right=.97)
    fig.suptitle("A legal tile boundary can unblock a fresh action", x=.04,
                 ha="left", y=.97, fontsize=18, weight="bold")
    fig.text(.04, .88, "Abstract shared resource · independent jobs · zero dispatch overhead", fontsize=11)
    for row, result in zip((1, 0), schedules):
        for interval in result["timeline"]:
            start, end = interval["start_ms"], interval["end_ms"]
            action = interval["job"] == "action"
            ax.barh(row, end-start, left=start, height=.48,
                    color=TEAL if action else BLUE, edgecolor="white", linewidth=1.2)
            if action:
                ax.text((start+end)/2, row, "Action", ha="center", va="center",
                        color="white", fontsize=9, weight="bold")
        finish = result["jobs"]["action"]["finish_ms"]
        ax.text(21.4, row, f"{finish:g} ms\n" + ("miss" if result["jobs"]["action"]["missed"] else "meets"),
                va="center", color=RED if row else TEAL, fontsize=10, weight="bold")
    ax.axvline(2, color="#64748b", linewidth=1, linestyle=":")
    ax.axvline(8, color=RED, linewidth=1.5, linestyle="--")
    ax.text(2, 1.48, "Release: 2 ms", ha="center", fontsize=9)
    ax.text(8, 1.48, "Deadline: 8 ms", color=RED, ha="center", fontsize=9)
    ax.set(xlabel="Elapsed time (ms)", xlim=(0, 23.5), ylim=(-.6, 1.8),
           yticks=[1, 0], yticklabels=["Whole-job FIFO", "EDF at 3 ms\ntile boundaries"],
           xticks=[0, 2, 6, 8, 12, 18, 21])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    fig.legend(handles=[Patch(color=BLUE, label="Vision service"),
                        Patch(color=TEAL, label="Action service")],
               loc="lower left", bbox_to_anchor=(.19, .12), ncol=2, frameon=False)
    fig.text(.04, .055, "Both schedules finish all work at 21 ms. With 2 ms dispatch cost per segment,\n"
             "the tiled schedule finishes the action at 10 ms and misses its deadline.",
             fontsize=10, linespacing=1.6)
    save(fig, "scheduling")


if __name__ == "__main__":
    residency()
    scheduling()
