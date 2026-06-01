"""
Plot episode-outcome breakdown vs. observation resolution for the
forward/centered sweep.

Expects runs under runs/PopulationWrapper-v0/ named:
  {forward|centered}_{resolution}_seed{NN}

where {resolution} is km/pixel (1, 2, 4, 8, 16, 32).

Each bar is stacked by termination reason (success / failed approach / max
steps / out of bounds), so the success segment height is the arrival rate and
the segments above it show *why* the remaining episodes failed. Computed from
trajectory CSVs (requires generate_trajectories.py to have been run first with
the termination_reason column present).
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.colors import (
    BASELINE_COLOR,
    FALLBACK_REASON_COLOR,
    MODE_COLORS,
    REASON_COLORS,
)
from scripts.common.sweep_plotting import (
    REASON_LABELS,
    SUCCESS_REASON,
    compute_baseline,
    compute_episode_length,
    compute_success_rate,
    compute_termination_breakdown,
    mean_breakdowns,
    seed_color_map,
    seed_legend,
)

DOT_ALPHA = 0.8
DOT_SIZE = 60
BAR_ALPHA = 0.6
BAR_WIDTH = 0.5

def collect_data(runs_root: Path, runway: str) -> pd.DataFrame:
    records = []
    for run_dir in sorted(runs_root.iterdir()):
        m = RUN_PATTERN.match(run_dir.name)
        if not m:
            continue
        mode, resolution, seed = m.group(1), int(m.group(2)), int(m.group(3))
        rate = compute_success_rate(run_dir, runway)
        if rate is None:
            print(f"  Skipping {run_dir.name}: no usable trajectory data")
            continue
        records.append({
            "mode": mode,
            "resolution": resolution,
            "seed": seed,
            "success_rate": rate,
            "breakdown": compute_termination_breakdown(run_dir, runway),
            "length": compute_episode_length(run_dir, runway),
        })
    return pd.DataFrame(records)


def _draw_baseline(ax, value: float | None, label: str) -> None:
    if value is not None:
        ax.axhline(value, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline ({label})", zorder=3)
        ax.legend(frameon=False)


def plot_episode_success(ax, df: pd.DataFrame, mode: str, color: str, baseline: float | None = None) -> None:
    resolutions = sorted(df["resolution"].unique())
    x = np.arange(len(resolutions))
    colors = seed_color_map(df)

    # Stacked outcome bars (success at the bottom, in the mode colour).
    ordered, means = mean_breakdowns(df, resolutions)
    bottom = np.zeros(len(resolutions))
    for reason in ordered:
        bar_color = color if reason == SUCCESS_REASON else REASON_COLORS.get(reason, FALLBACK_REASON_COLOR)
        ax.bar(x, means[reason], width=BAR_WIDTH, bottom=bottom,
               color=bar_color, alpha=BAR_ALPHA, label=REASON_LABELS.get(reason, reason))
        bottom += means[reason]

    # Per-seed success rate dots overlaid on the success segment.
    for i, res in enumerate(resolutions):
        seed_rates = {row["seed"]: row["success_rate"] for _, row in df[df["resolution"] == res].iterrows()}
        seeds = sorted(seed_rates)
        jitter = np.linspace(-0.08, 0.08, len(seeds))
        for xi, seed in zip(jitter, seeds):
            ax.scatter(x[i] + xi, seed_rates[seed],
                       color=colors[seed], s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.8)

    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline success ({baseline:.0%})", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} km/px" for r in resolutions])
    ax.set_xlabel("Observation resolution")
    ax.set_ylabel("Episode outcome fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{mode.capitalize()} observation window")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Two legends outside the axes: outcome categories + baseline, then seeds.
    outcome_handles, outcome_labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(outcome_handles, outcome_labels, frameon=False, fontsize=8,
                     title="Episode outcome", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    seed_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, label=f"Seed {s}")
        for s, c in colors.items()
    ]
    ax.legend(handles=seed_handles, frameon=False, fontsize=8,
              title="Seed (success rate)", loc="lower left", bbox_to_anchor=(1.01, 0.0))


def plot_mode_length(ax, df: pd.DataFrame, mode: str, color: str, baseline: float | None = None) -> None:
    resolutions = sorted(df["resolution"].unique())
    x = np.arange(len(resolutions))
    colors = seed_color_map(df)

    for i, res in enumerate(resolutions):
        res_df = df[df["resolution"] == res]

        all_lengths = []
        seeds = sorted(row["seed"] for _, row in res_df.iterrows() if row["length"] is not None)
        slot_width = BAR_WIDTH / max(len(seeds), 1)
        seed_centers = {seed: x[i] - BAR_WIDTH / 2 + (j + 0.5) * slot_width for j, seed in enumerate(seeds)}

        for _, row in res_df.iterrows():
            if row["length"] is None:
                continue
            lengths = row["length"].values
            all_lengths.extend(lengths)
            jitter = np.random.default_rng(row["seed"]).uniform(-slot_width * 0.35, slot_width * 0.35, len(lengths))
            ax.scatter(seed_centers[row["seed"]] + jitter, lengths,
                       color=colors[row["seed"]], s=DOT_SIZE * 0.5, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="none")

        if all_lengths:
            ax.bar(x[i], np.mean(all_lengths), width=BAR_WIDTH, color=color, alpha=BAR_ALPHA)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} km/px" for r in resolutions])
    ax.set_xlabel("Observation resolution")
    ax.set_ylabel("Mean episode length (s)")
    ax.set_title(f"{mode.capitalize()} observation window — episode length")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    seed_legend(ax, colors)
    _draw_baseline(ax, baseline, f"{baseline:.0f} s" if baseline is not None else "")


def main(runs_root: Path, output_dir: Path, baseline_dir: Path, runway: str) -> None:
    df = collect_data(runs_root, runway)
    if df.empty:
        print("No data found. Run generate_trajectories.py on the sweep runs first.")
        return

    baseline_rate, baseline_length = compute_baseline(baseline_dir, runway)
    if baseline_rate is None or baseline_length is None:
        if not baseline_dir.exists():
            print(f"Baseline — directory not found: {baseline_dir} (plotting without baseline)")
        else:
            print(f"Baseline — no usable '{runway}_map' trajectory data in {baseline_dir} "
                  "(plotting without baseline)")
    else:
        print(f"Baseline — success rate: {baseline_rate:.1%}, mean episode length: {baseline_length:.1f} s")

    for mode in ("forward", "centered"):
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            print(f"No data for mode '{mode}', skipping.")
            continue

        color = MODE_COLORS[mode]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_episode_success(ax, mode_df, mode, color, baseline=baseline_rate)
        fig.tight_layout()
        out_path = output_dir / f"episode_success_{runs_root.name}_{mode}_{runway}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_mode_length(ax, mode_df, mode, color, baseline=baseline_length)
        fig.tight_layout()
        out_path = output_dir / f"episode_length_{runs_root.name}_{mode}_{runway}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    RUN_PATTERN = re.compile(r"^(?:sweep_\d+_)?(forward|centered)_(\d+)_seed(\d+)$")

    parser = argparse.ArgumentParser(description="Plot resolution sweep results.")
    parser.add_argument("runs_root", type=str, help="path to runs root for comparison")
    parser.add_argument("--baseline", type=Path, default=None, help=f"Baseline run directory")
    parser.add_argument("--runway",help=f"Runway identifier used to select trajectory CSVs")
    parser.add_argument("--output_dir", type=Path, default=Path("plots/sweep_overview_plots"), help=f"Output directory for the plots")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = args.output_dir / runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = args.baseline
    runway = args.runway

    main(runs_root, output_dir, baseline_dir, runway)
