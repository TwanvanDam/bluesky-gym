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

from bluesky_gym.maps.map_sources import TiffMapSourceConfig
from bluesky_gym.metrics.evaluation_metrics import build_metric_fn, make_pop_samplers
from scripts.common.colors import *
from scripts.common.figures import legend_right, outcome_ylim, paper_axes, save, PLOT_TYPE_TO_SIZE
from scripts.common.sweep_plotting import (
    REASON_LABELS,
    REASON_ORDER,
    add_reward,
    compute_episode_metrics,
    compute_success_rate,
    compute_termination_breakdown,
    draw_boxplot,
    find_csv,
    seed_color_map,
)

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

MODE_TO_COLOR = {"centered": CENTERED_COLOR, "forward": FORWARD_COLOR}

GROUP_TO_HATCH = {
    "sweep_2":     "",
    "convergence": "////",
}

BAR_ALPHA = BOXPLOT_ALPHA
BAR_WIDTH = 0.35
DOT_ALPHA = 0.85
DOT_SIZE = 55

# Load plot details
OUTCOME_WIDTH, OUTCOME_HEIGHT = PLOT_TYPE_TO_SIZE["sweep_breakdown"]
METRIC_WIDTH, METRIC_HEIGHT = PLOT_TYPE_TO_SIZE["alignment"]
LEGEND_STRIP_IN = 1.25

# One x layout for every figure in this script: [centered before, centered after]
# gap [forward before, forward after], so the outcome panel and the metric panels
# read the same way and their boxes/bars sit at the same places.
COMBOS = [
    ("centered", "sweep_2",     1 * BAR_WIDTH),
    ("centered", "convergence", 2 * BAR_WIDTH + 0.1),
    ("forward",  "sweep_2",     4 * BAR_WIDTH),
    ("forward",  "convergence", 5 * BAR_WIDTH + 0.1),
]
MODE_TICKS = [(3 * BAR_WIDTH + 0.1) / 2, (9 * BAR_WIDTH + 0.1) / 2]
MODE_TICK_LABELS = ["C4", "F4"]
DIVIDER_X = (6 * BAR_WIDTH + 0.1) / 2

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
                csv = find_csv(run_dir, runway)
                if csv is None:
                    print(f"  Skipping {run_dir.name}: no trajectory data for {runway}")
                    continue
                raw = pd.read_csv(csv)
                ep = compute_episode_metrics(calculate_metrics(raw), mean_episode_length)
                ep["group"] = group_name
                ep["mode"] = mode
                ep["seed"] = seed
                frames.append(ep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

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

def plot_outcome_comparison(df: pd.DataFrame, runway: str, output_dir: Path) -> None:
    """Stacked bar + per-seed dots.

    Same layout as :func:`plot_metric_comparison` — the shared :data:`COMBOS`
    x positions, one framed legend in the reserved right strip. Before/after is
    on the tick labels rather than in the hatch, since here the hatch already
    encodes the termination reason.
    """
    seed_colors = seed_color_map(df)

    # Extra bottom room for the "C4"/"F4" group labels under the tick labels;
    # they sit where the (unused) x axis label would go.
    fig, ax = paper_axes(OUTCOME_WIDTH, OUTCOME_HEIGHT,
                         right=LEGEND_STRIP_IN + 0.25, bottom=0.55)
    seen_reasons: set = set()
    min_seed_rates = 1.0

    for mode, group, xi in COMBOS:
        sub = df[(df["mode"] == mode) & (df["group"] == group)]
        if sub.empty:
            continue

        mode_color = MODE_TO_COLOR[mode]
        ordered, means = _mean_breakdowns(sub)
        bottom = 0.0
        for reason, frac in zip(ordered, means):
            hatch = REASON_HATCH.get(reason, "")
            if reason in FILLED_REASONS:
                ax.bar(xi, frac, width=BAR_WIDTH, bottom=bottom, color=mode_color,
                       alpha=BAR_ALPHA, hatch=hatch, edgecolor="black", linewidth=0.5)
            else:
                ax.bar(xi, frac, width=BAR_WIDTH, bottom=bottom, facecolor="none",
                       hatch=hatch, edgecolor="black", linewidth=0.5)
            seen_reasons.add(reason)
            bottom += frac

        seeds = sorted(sub["seed"].unique())
        jitter = np.linspace(-0.08, 0.08, len(seeds))
        for dx, seed in zip(jitter, seeds):
            row = sub[sub["seed"] == seed]
            if row.empty:
                continue
            min_seed_rates = min(min_seed_rates, row["success_rate"].values[0])
            ax.scatter(xi + dx, row["success_rate"].values[0],
                       color=seed_colors[seed], s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.8)

    ax.set_xticks([pos for _, _, pos in COMBOS])
    ax.set_xticklabels(["Before", "After", "Before", "After"])
    ax.set_xlabel("")
    # Mode labels on the line the x axis label would occupy, centred under each
    # before/after pair — same "C4"/"F4" naming as the metric panels.
    for pos, label in zip(MODE_TICKS, MODE_TICK_LABELS):
        ax.annotate(label, xy=(pos, -0.20), xycoords=("data", "axes fraction"),
                    ha="center", va="top")
    ax.set_ylabel("Episode outcome fraction")
    ax.grid(axis="y")
    outcome_ylim(ax, min_seed_rates)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axvline(DIVIDER_X, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

    mode_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=BAR_ALPHA, label=mode.capitalize())
        for mode, color in MODE_TO_COLOR.items()
        if not df[df["mode"] == mode].empty
    ]
    reason_handles = [
        plt.Rectangle((0, 0), 1, 1,
                      fc="lightgray" if reason in FILLED_REASONS else "none",
                      hatch=REASON_HATCH[reason], edgecolor="black",
                      label=REASON_LABELS.get(reason, reason))
        for reason in REASON_HATCH if reason in seen_reasons
    ]
    seed_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, label=f"Seed {s}")
        for s, c in seed_colors.items()
    ]
    # One framed legend in the reserved strip, like the metric panels: two
    # stacked legends collide in a figure this short.
    legend_right(ax, handles=mode_handles + reason_handles + seed_handles,
                 frameon=True, edgecolor="k")

    runway_id = runway.replace("/", "_")
    out_path = output_dir / f"outcome_comparison_{runway_id}.pdf"
    save(fig, out_path)
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
    fig, ax = paper_axes(METRIC_WIDTH, METRIC_HEIGHT, right=LEGEND_STRIP_IN)
    mode_handles = []
    seen_modes = set()

    for mode, group, pos in COMBOS:
        color = MODE_TO_COLOR[mode]
        sub = df[(df["mode"] == mode) & (df["group"] == group)]
        data = sub[metric].values
        if len(data) == 0:
            continue
        draw_boxplot(ax, data, pos, color, BAR_WIDTH, hatch=GROUP_TO_HATCH[group])
        if mode not in seen_modes:
            mode_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=BOXPLOT_ALPHA,
                                              label=mode.capitalize()))
            seen_modes.add(mode)

    # One tick per mode group, centred under the before/after pair. Before/after
    # is conveyed by the hatch (see legend), so no per-box "Before"/"After" labels.
    ax.set_xticks(MODE_TICKS)
    ax.set_xticklabels(MODE_TICK_LABELS)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.axvline(DIVIDER_X, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.grid(axis="y")

    fix_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="lightgray", edgecolor="black", label="Before fix"),
        plt.Rectangle((0, 0), 1, 1, fc="lightgray", edgecolor="black",
                      hatch=GROUP_TO_HATCH["convergence"], label="After fix"),
    ]
    legend_right(ax, handles=mode_handles + fix_handles, frameon=True, edgecolor="k")

    runway_id = runway.replace("/", "_")
    out_path = output_dir / f"{metric}_comparison_{runway_id}.pdf"
    save(fig, out_path)
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
    parser.add_argument("--cache", action="store_true", default=False,
                        help="Read/write cached per-runway metric CSV under runs/convergence; "
                             "a cache hit skips bluesky init and metric recomputation")
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
        runway_id = runway.replace("/", "_")
        cache_path = ROOT / "convergence" / f"cached_alignment_metrics_{runway_id}.csv"
        if args.cache and cache_path.exists():
            print(f"\nUsing cached metric data from {cache_path} ...")
            metric_df = pd.read_csv(cache_path)
        else:
            print(f"\nCollecting metric data for runway {runway} ...")
            bs.init()
            # Fixed-map overview: legacy TiffMapSource branch ignores bounds and is
            # shared across all runs (post-resample clip at the given percentile).
            samplers = make_pop_samplers(
                TiffMapSourceConfig(file_path=args.map_path), bounds=None,
                clip_percentile=args.noise_clip_percentile,
                train_resampling="cubic_spline", true_resampling="average")
            calculate_metrics = build_metric_fn(samplers)
            metric_df = collect_metric_data(runway, calculate_metrics, args.mean_episode_length)
            if args.cache and not metric_df.empty:
                print(f"Saving metric data to {cache_path} ...")
                metric_df.to_csv(cache_path, index=False)
        if metric_df.empty:
            print("No metric data found.")
        else:
            metric_df["combined"] = metric_df["normalized_fuel"] + metric_df["normalized_noise"]
            add_reward(metric_df, args.fuel_weight)
            for metric in ("fuel", "noise", "normalized_fuel", "normalized_noise",
                           "combined", "reward", "reward_unclipped"):
                plot_metric_comparison(metric_df, metric, METRICS[metric], runway,
                                       output_dir, box_width=BAR_WIDTH)
