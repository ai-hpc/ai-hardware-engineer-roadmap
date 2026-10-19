#!/usr/bin/env python3
"""Optional Matplotlib figure from an actual timed_policy_lab JSON report."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("timed-policy.png"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads(args.report.read_text())
    rows = {row["timing"]["name"]: row for row in data["scenarios"]}
    labels = {"fast_latest": "10 ms, latest", "slow_fifo_without_age_check": "120 ms, FIFO, age unchecked",
              "slow_fifo_with_age_check": "120 ms, FIFO, age checked",
              "slow_latest_with_age_check": "120 ms, latest, age checked",
              "over_budget_latest": "350 ms, latest, age checked"}
    colors = {name: color for name, color in zip(rows, ("#2574a9", "#c0392b", "#8856a7", "#14845d", "#b87800"))}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for name, scenario in rows.items():
        trace = scenario["per_episode"][0].get("trace")
        if not trace:
            raise ValueError("Regenerate the report without --no-trace to plot it")
        time = [row["time_ms"] / 1000 for row in trace]
        target = axes[0, 1] if name == "slow_fifo_without_age_check" else axes[0, 0]
        target.plot(time, [row["position_m"] for row in trace], label=labels[name], color=colors[name], lw=1.7)
        if name in ("slow_fifo_without_age_check", "slow_latest_with_age_check"):
            axes[1, 0].plot(time, [row["observation_age_start_ms"] + scenario["timing"]["dt_ms"]
                                  if row["observation_age_start_ms"] is not None else float("nan")
                                  for row in trace], label=labels[name], color=colors[name], lw=1.5)
            axes[1, 1].plot(time, [row["pending_observations"] for row in trace],
                            label=labels[name], color=colors[name], lw=1.5)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel("Simulated time (s)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="best")
    for ax in axes[0]:
        ax.axhline(0, color="#555555", lw=.8, linestyle="--")
        ax.set_ylabel("Position relative to goal (m)")
    axes[0, 0].set_title("Same trained policy, different delivery rules")
    axes[0, 1].set_title("Unchecked FIFO drift (separate vertical scale)")
    axes[1, 0].axhline(250, color="#333333", linestyle="--", lw=1)
    axes[1, 0].text(.5, 275, "250 ms deadline", fontsize=9)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(100, 10000)
    axes[1, 0].set_ylabel("Age at end of applied interval (ms, log scale)")
    axes[1, 0].set_title("A bounded queue can still hold stale observations")
    axes[1, 1].set_ylabel("Pending observations")
    axes[1, 1].set_title("Latest-only discards work before it becomes backlog")
    fig.suptitle("A low prediction error does not guarantee a timely robot loop\n"
                 "Episode 0 • simulated timing and sensor outage • no physical robot", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
