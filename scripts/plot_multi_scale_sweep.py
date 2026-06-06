"""
Plot episode-outcome breakdown for the multi-scale observation sweep.

Expects runs under runs/multi-scale-sweep/ named:
  multi_scale_{group}{variant}_seed{NN}

where group is 1-5 and variant is a or b.
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
    REASON_COLORS,
    qual,
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
)

DOT_ALPHA = 0.8
DOT_SIZE = 60
BAR_ALPHA = 0.6
BAR_WIDTH = 0.7

VARIANT_TO_OBSERVATION = {
    "1a": "C2 + C4",
    "1b": "F2 + C4",
    "2a": "C4 + C8",
    "2b": "C4 + F8",
    "3a": "C4 + C16",
    "3b": "C4 + F16",
    "4a": "C2 + C4 + C8",
    "4b": "C2 + C4 + C16",
    "5a": "C2 + C16",
    "5b": "C2 + C8",
}

# One color per group; variant b gets a lighter shade.
_GROUP_BASE_COLORS = {g: qual(i) for i, g in enumerate(range(1, 6))}


def _group_color(group_num: int, variant: str) -> tuple:
    base = _GROUP_BASE_COLORS[group_num]
    if variant == "b":
        return tuple(c + (1 - c) * 0.45 for c in base[:3]) + (base[3],)
    return base


def collect_data(runs_root: Path, runway: str) -> pd.DataFrame:
    records = []
    for run_dir in sorted(runs_root.iterdir()):
        match = RUN_PATTERN.match(run_dir.name)
        if not match:
            continue
        group_num, variant, seed = int(match.group(1)), match.group(2), int(match.group(3))
        rate = compute_success_rate(run_dir, runway)
        if rate is None:
            print(f"  Skipping {run_dir.name}: no usable trajectory data")
            continue
        records.append({
            "group_num": group_num,
            "variant": variant,
            "config_id": f"{group_num}{variant}",
            "seed": seed,
            "success_rate": rate,
            "breakdown": compute_termination_breakdown(run_dir, runway),
            "length": compute_episode_length(run_dir, runway),
        })
    return pd.DataFrame(records)


def plot_episode_success(ax, df: pd.DataFrame, baseline: float | None = None) -> None:
    config_ids = sorted(df["config_id"].unique(), key=lambda c: (int(c[:-1]), c[-1]))
    x = np.arange(len(config_ids))
    seed_colors = seed_color_map(df)

    ordered, means = mean_breakdowns(df, config_ids, pos_col="config_id")
    bottom = np.zeros(len(config_ids))
    for reason in ordered:
        if reason == SUCCESS_REASON:
            bar_colors = [_group_color(int(cid[:-1]), cid[-1]) for cid in config_ids]
        else:
            bar_colors = [REASON_COLORS.get(reason, FALLBACK_REASON_COLOR)] * len(config_ids)
        ax.bar(x, means[reason], width=BAR_WIDTH, bottom=bottom,
               color=bar_colors, alpha=BAR_ALPHA, label=REASON_LABELS.get(reason, reason))
        bottom += means[reason]

    for i, cid in enumerate(config_ids):
        seed_rates = {
            row["seed"]: row["success_rate"]
            for _, row in df[df["config_id"] == cid].iterrows()
        }
        seeds = sorted(seed_rates)
        jitter = np.linspace(-0.08, 0.08, len(seeds))
        for xi, seed in zip(jitter, seeds):
            ax.scatter(x[i] + xi, seed_rates[seed],
                       color=seed_colors[seed], s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.8)

    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline success ({baseline:.0%})", zorder=4)

    tick_labels = [f"{cid}\n{VARIANT_TO_OBSERVATION.get(cid, cid)}" for cid in config_ids]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Episode outcome fraction")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    outcome_handles, outcome_labels_list = ax.get_legend_handles_labels()
    leg1 = ax.legend(outcome_handles, outcome_labels_list, frameon=False, fontsize=8,
                     title="Episode outcome", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    seed_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, label=f"Seed {s}")
        for s, c in seed_colors.items()
    ]
    ax.legend(handles=seed_handles, frameon=False, fontsize=8,
              title="Seed", loc="lower left", bbox_to_anchor=(1.01, 0.0))


def plot_episode_length(ax, df: pd.DataFrame, baseline: float | None = None) -> None:
    config_ids = sorted(df["config_id"].unique(), key=lambda c: (int(c[:-1]), c[-1]))
    x = np.arange(len(config_ids))
    seed_colors = seed_color_map(df)

    for i, cid in enumerate(config_ids):
        cid_df = df[df["config_id"] == cid]
        all_lengths = []
        seeds = sorted(row["seed"] for _, row in cid_df.iterrows() if row["length"] is not None)
        slot_width = BAR_WIDTH / max(len(seeds), 1)
        seed_centers = {
            seed: x[i] - BAR_WIDTH / 2 + (j + 0.5) * slot_width
            for j, seed in enumerate(seeds)
        }
        for _, row in cid_df.iterrows():
            if row["length"] is None:
                continue
            lengths = row["length"].values
            all_lengths.extend(lengths)
            jitter = np.random.default_rng(row["seed"]).uniform(
                -slot_width * 0.35, slot_width * 0.35, len(lengths)
            )
            ax.scatter(seed_centers[row["seed"]] + jitter, lengths,
                       color=seed_colors[row["seed"]], s=DOT_SIZE * 0.5, zorder=5,
                       alpha=DOT_ALPHA, edgecolors="none")
        if all_lengths:
            color = _group_color(int(cid[:-1]), cid[-1])
            ax.bar(x[i], np.mean(all_lengths), width=BAR_WIDTH, color=color, alpha=BAR_ALPHA)

    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline ({baseline:.0f} s)", zorder=3)
        ax.legend(frameon=False)

    tick_labels = [f"{cid}\n{VARIANT_TO_OBSERVATION.get(cid, cid)}" for cid in config_ids]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Mean episode length (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    seed_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, label=f"Seed {s}")
        for s, c in seed_colors.items()
    ]
    ax.legend(handles=seed_handles, frameon=False, fontsize=8,
              title="Seed", loc="upper right")


def main(runs_root: Path, output_dir: Path, baseline_dir: Path | None, runway: str) -> None:
    df = collect_data(runs_root, runway)
    if df.empty:
        print("No data found. Run generate_trajectories.py on the sweep runs first.")
        return

    baseline_rate, baseline_length = None, None
    if baseline_dir is not None:
        baseline_rate, baseline_length = compute_baseline(baseline_dir, runway)
        if baseline_rate is None:
            print(f"Baseline — no usable trajectory data in {baseline_dir} (plotting without baseline)")
        else:
            print(f"Baseline — success rate: {baseline_rate:.1%}, mean length: {baseline_length:.1f} s")

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_episode_success(ax, df, baseline=baseline_rate)
    fig.tight_layout()
    out_path = output_dir / f"episode_success_{runs_root.name}_{runway}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_episode_length(ax, df, baseline=baseline_length)
    fig.tight_layout()
    out_path = output_dir / f"episode_length_{runs_root.name}_{runway}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    RUN_PATTERN = re.compile(r"^(?:multi_scale_)?(\d)([ab])_seed(\d+)$")

    parser = argparse.ArgumentParser(description="Plot multi-scale sweep results.")
    parser.add_argument("runs_root", type=str, help="path to runs root")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline run directory")
    parser.add_argument("--runway", required=True, help="Runway identifier used to select trajectory CSVs")
    parser.add_argument("--output_dir", type=Path, default=Path("plots/sweep_overview_plots"),
                        help="Output directory for the plots")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = args.output_dir / runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    main(runs_root, output_dir, args.baseline, args.runway)
