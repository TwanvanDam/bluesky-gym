"""
Compare episode outcomes and metrics between:
  - resolution_sweep_2 (original alignment, before meridian convergence fix)
  - convergence (fixed alignment)

Both sets use centered_4 and forward_4 configurations at 4 km/px resolution,
3 seeds each.

Produces per runway:
  1. Episode outcome breakdown (stacked bars + per-seed dots)
  2. Metrics boxplots (fuel, noise, normalized fuel, normalized noise, combined)
"""

import argparse
from pathlib import Path

import bluesky as bs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bluesky_gym.metrics.evaluation_metrics import build_metric_fn
from scripts.common.colors import (
    FALLBACK_REASON_COLOR,
    MODE_COLORS,
    REASON_COLORS,
)
from scripts.common.sweep_plotting import (
    REASON_LABELS,
    REASON_ORDER,
    SUCCESS_REASON,
    add_reward,
    compute_episode_metrics,
    compute_success_rate,
    compute_termination_breakdown,
    draw_boxplot,
    find_csvs,
    seed_color_map,
)

# ---------------------------------------------------------------------------
# Hard-coded run directories
# ---------------------------------------------------------------------------

ROOT = Path("runs")

GROUPS = {
    "sweep_2": {
        "centered": [
            ROOT / "resolution_sweep_2" / "sweep_2_centered_4_seed00",
            ROOT / "resolution_sweep_2" / "sweep_2_centered_4_seed01",
            ROOT / "resolution_sweep_2" / "sweep_2_centered_4_seed02",
        ],
        "forward": [
            ROOT / "resolution_sweep_2" / "sweep_2_forward_4_seed00",
            ROOT / "resolution_sweep_2" / "sweep_2_forward_4_seed01",
            ROOT / "resolution_sweep_2" / "sweep_2_forward_4_seed02",
        ],
    },
    "convergence": {
        "centered": [
            ROOT / "convergence" / "convergence_centered_4_seed00",
            ROOT / "convergence" / "convergence_centered_4_seed01",
            ROOT / "convergence" / "convergence_centered_4_seed02",
        ],
        "forward": [
            ROOT / "convergence" / "convergence_forward_4_seed00",
            ROOT / "convergence" / "convergence_forward_4_seed01",
            ROOT / "convergence" / "convergence_forward_4_seed02",
        ],
    },
}

GROUP_LABELS = {
    "sweep_2": "Before fix\n(sweep_2)",
    "convergence": "After fix\n(convergence)",
}

# Hatch pattern distinguishes sweep_2 from convergence within the same mode colour.
GROUP_HATCH = {"sweep_2": "//", "convergence": ""}

BAR_ALPHA = 0.6
BAR_WIDTH = 0.5
DOT_ALPHA = 0.85
DOT_SIZE = 55

# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------


def collect_outcome_data(runway: str) -> pd.DataFrame:
    records = []
    for group_name, modes in GROUPS.items():
        for mode, run_dirs in modes.items():
            for seed, run_dir in enumerate(run_dirs):
                rate = compute_success_rate(run_dir, runway)
                if rate is None:
                    print(f"  Skipping {run_dir.name}: no trajectory data for {runway}")
                    continue
                records.append({
                    "group": group_name,
                    "mode": mode,
                    "seed": seed,
                    "success_rate": rate,
                    "breakdown": compute_termination_breakdown(run_dir, runway),
                })
    return pd.DataFrame(records)


def collect_metric_data(runway: str, calculate_metrics, mean_episode_length: float) -> pd.DataFrame:
    frames = []
    for group_name, modes in GROUPS.items():
        for mode, run_dirs in modes.items():
            for seed, run_dir in enumerate(run_dirs):
                csvs = find_csvs(run_dir, runway)
                if not csvs:
                    print(f"  Skipping {run_dir.name}: no trajectory data for {runway}")
                    continue
                raw = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
                ep = compute_episode_metrics(calculate_metrics(raw), mean_episode_length)
                ep["group"] = group_name
                ep["mode"] = mode
                ep["seed"] = seed
                frames.append(ep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _mean_breakdowns(group_df: pd.DataFrame) -> tuple[list, np.ndarray]:
    """Mean outcome fractions for a single (mode, group) slice, averaged across seeds."""
    present = set()
    for bd in group_df["breakdown"]:
        if bd is not None:
            present.update(bd.index)
    present.discard("none")
    ordered = [r for r in REASON_ORDER if r in present] + sorted(present - set(REASON_ORDER))

    means = np.zeros(len(ordered))
    bds = [bd for bd in group_df["breakdown"] if bd is not None]
    if bds:
        mean_bd = pd.concat(bds, axis=1).fillna(0.0).mean(axis=1)
        for i, reason in enumerate(ordered):
            means[i] = mean_bd.get(reason, 0.0)
    return ordered, means


# ---------------------------------------------------------------------------
# Main plot functions
# ---------------------------------------------------------------------------


def plot_outcome_comparison(df: pd.DataFrame, runway: str, output_dir: Path) -> None:
    """Stacked bar + per-seed dots.

    Layout: four bars in a single panel grouped by mode —
      [centered_before, centered_after]  [forward_before, forward_after]
    with a gap between the two mode groups.
    """
    combos = [
        ("centered", "sweep_2",     0.0),
        ("centered", "convergence", 1.0),
        ("forward",  "sweep_2",     2.5),
        ("forward",  "convergence", 3.5),
    ]
    tick_positions = [c[2] for c in combos]
    tick_labels    = ["Before", "After", "Before", "After"]

    seed_colors = seed_color_map(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    legend_reasons_added = set()

    for mode, group, xi in combos:
        sub = df[(df["mode"] == mode) & (df["group"] == group)]
        if sub.empty:
            continue

        mode_color = MODE_COLORS[mode]
        hatch = GROUP_HATCH[group]
        ordered, means = _mean_breakdowns(sub)
        bottom = 0.0
        for reason, frac in zip(ordered, means):
            bar_color = mode_color if reason == SUCCESS_REASON else REASON_COLORS.get(reason, FALLBACK_REASON_COLOR)
            label = REASON_LABELS.get(reason, reason) if reason not in legend_reasons_added else "_nolegend_"
            ax.bar(xi, frac, width=BAR_WIDTH, bottom=bottom,
                   color=bar_color, alpha=BAR_ALPHA, hatch=hatch, label=label)
            legend_reasons_added.add(reason)
            bottom += frac

        seeds = sorted(sub["seed"].unique())
        jitter = np.linspace(-0.08, 0.08, len(seeds))
        for dx, seed in zip(jitter, seeds):
            row = sub[sub["seed"] == seed]
            if row.empty:
                continue
            ax.scatter(xi + dx, row["success_rate"].values[0],
                       color=seed_colors[seed], s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.8)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("")
    ax.annotate("Centered", xy=(0.5, -0.12), xycoords="axes fraction",
                ha="center", fontsize=10, fontweight="bold")
    ax.annotate("Forward", xy=(0.82, -0.12), xycoords="axes fraction",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Episode outcome fraction")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(1.75, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

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
              title="Seed (success rate)", loc="lower left", bbox_to_anchor=(1.01, 0.0))

    fig.text(0.5, 1.01,
             "Hatch (//) = before fix (sweep_2)  |  Solid = after fix (convergence)",
             ha="center", fontsize=8, style="italic", transform=fig.transFigure)

    runway_id = runway.replace("/", "_")
    fig.tight_layout()
    out_path = output_dir / f"outcome_comparison_{runway_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    runway: str,
    output_dir: Path,
    box_width: float = 0.35,
) -> None:
    """Boxplots per (group × mode) combination.

    Layout mirrors plot_outcome_comparison: four boxes grouped by mode —
      [centered_before, centered_after]  [forward_before, forward_after]
    with a gap between the two mode groups.
    """
    combos = [
        ("centered", "sweep_2",     0.0),
        ("centered", "convergence", 1.0),
        ("forward",  "sweep_2",     2.5),
        ("forward",  "convergence", 3.5),
    ]
    tick_positions = [c[2] for c in combos]
    tick_labels    = ["Before", "After", "Before", "After"]

    fig, ax = plt.subplots(figsize=(9, 5))
    legend_handles = []
    seen_modes = set()

    for mode, group, pos in combos:
        color = MODE_COLORS[mode]
        sub = df[(df["mode"] == mode) & (df["group"] == group)]
        data = sub[metric].values
        if len(data) == 0:
            continue
        draw_boxplot(ax, data, pos, color, box_width)
        if mode not in seen_modes:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6,
                                                label=mode.capitalize()))
            seen_modes.add(mode)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("")
    ax.annotate("Centered", xy=(0.5, -0.12), xycoords="axes fraction",
                ha="center", fontsize=10, fontweight="bold")
    ax.annotate("Forward", xy=(0.82, -0.12), xycoords="axes fraction",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.axvline(1.75, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.legend(handles=legend_handles, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(0.5, 1.01,
             "Before = sweep_2 (original alignment)  |  After = convergence (fixed alignment)",
             ha="center", fontsize=8, style="italic", transform=fig.transFigure)

    runway_id = runway.replace("/", "_")
    fig.tight_layout()
    out_path = output_dir / f"{metric}_comparison_{runway_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot alignment comparison (sweep_2 vs convergence).")
    parser.add_argument("--runway", default="EHAM_RW27",
                        help="Runway identifier for trajectory CSV selection (default: EHAM_RW27)")
    parser.add_argument("--map-path", default="./scripts/population_maps/europe_3035_1km.tif",
                        help="Path to population GeoTIFF for metric calculation")
    parser.add_argument("--noise-clip-percentile", type=float, default=99.9)
    parser.add_argument("--mean-episode-length", type=float, default=1400.0,
                        help="Reference episode length in seconds used for metric normalisation")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/sweep_overview_plots/alignment_comparison"),
                        help="Output directory for figures")
    parser.add_argument("--fuel-weight", type=float, default=0.5,
                        help="Fuel weight in reward = success_term - fw*norm_fuel - (1-fw)*norm_noise_clipped")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Skip metrics plots (quicker; does not require bluesky/map)")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runway = args.runway

    print(f"\nCollecting outcome data for runway {runway} ...")
    outcome_df = collect_outcome_data(runway)
    if outcome_df.empty:
        print("No outcome data found — check run directories and trajectory generation.")
    else:
        plot_outcome_comparison(outcome_df, runway, output_dir)

    if not args.no_metrics:
        print(f"\nCollecting metric data for runway {runway} ...")
        bs.init()
        calculate_metrics = build_metric_fn(
            Path(args.map_path), args.noise_clip_percentile
        )
        metric_df = collect_metric_data(runway, calculate_metrics, args.mean_episode_length)
        if metric_df.empty:
            print("No metric data found.")
        else:
            metric_df["combined"] = metric_df["normalized_fuel"] + metric_df["normalized_noise"]
            add_reward(metric_df, args.fuel_weight)
            for metric, ylabel in [
                ("fuel", "Fuel [kg]"),
                ("noise", "Noise [W·s]"),
                ("normalized_fuel", "Normalised fuel"),
                ("normalized_noise", "Normalised noise"),
                ("combined", "Normalised fuel + noise"),
                ("reward", "Reward"),
            ]:
                plot_metric_comparison(metric_df, metric, ylabel, runway, output_dir)
