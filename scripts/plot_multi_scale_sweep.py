"""
Plot the multi-scale observation sweep: noise/fuel metric boxplots and/or
episode-outcome breakdown for runs named multi_scale_{group}{variant}_seed{NN}
(group 1-5, variant a/b).

    python -m scripts.plot_multi_scale_sweep <runs_root> --baseline <C4_run>
    python -m scripts.plot_multi_scale_sweep <runs_root> --plots breakdown

--plots {both,metrics,breakdown} selects the views (default both). Only the
metrics view needs BlueSky + the noise/fuel metric fn.

The metrics view writes one PDF per metric *and* `metrics_grid_*.pdf`: the
reward/noise/fuel panels laid out as one matplotlib figure, the variant legend
in the cell the odd panel count leaves free, and the (a), (b), … captions above
the panels. That grid replaces the 2x2 block of `subfigure`s in the paper — the
panels are laid out in inches, so they are guaranteed the same axes size and
the same text size, and LaTeX never rescales them. The `figure` environment to
paste is printed at the end.
"""

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.colors import *
from scripts.common.figures import (
    PANEL_LETTERS, PLOT_TYPE_TO_SIZE, W_FULL,
    grid_caption, grid_latex_snippet, legend_in_cell, legend_right, metric_grid,
    outcome_ylim, paper_axes, save,
)
from scripts.common.sweep_plotting import (
    REASON_LABELS,
    SUCCESS_REASON,
    boxplot_stats,
    draw_boxplot,
    mean_breakdowns,
    collect_breakdown_data, add_reward, collect_baseline_metrics, collect_run_metrics,
    collect_baseline_breakdown, collect_baseline_seed_rates,
    run_sweep_args_parser,
)

# Figure geometry comes from common.figures: every panel is saved at exactly its
# LaTeX slot size, so nothing is rescaled on inclusion. The breakdown legend lives
# in a reserved right-hand strip; the variant legend for the metric panels is a
# separate PDF, one panel tall so the two line up side by side.
METRIC_WIDTH, METRIC_HEIGHT = PLOT_TYPE_TO_SIZE["sweep_metric"]
BREAKDOWN_WIDTH, BREAKDOWN_HEIGHT = PLOT_TYPE_TO_SIZE["sweep_breakdown"]
LEGEND_STRIP_IN = 1.7

# Panels of the combined figure (common.figures.metric_grid owns its geometry);
# the leftover cell holds the variant legend. Which metrics get a panel is
# common.colors.METRIC_TO_CAPTION, shared with every other sweep grid.
GRID_COLS = 2
GRID_WIDTH = W_FULL

BOX_WIDTH = 0.35
BOX_OFFSET = 0.2  # half-gap between the two variants in a group
BAR_WIDTH = 0.6
DOT_ALPHA = 0.8
DOT_SIZE = 60

# success/failed_approach are filled with the mode/baseline color (hatch drawn on
# top); the remaining failure modes are hatch-only so they don't compete visually
# with the arrival-rate segments.
FILLED_REASONS = {"success", "failed_approach"}

# {multi_scale_}{group}{variant}_seed{NN}, group 1-5, variant a/b
PATTERN = re.compile(r"^(?:multi_scale_)?(?P<group_num>\d)(?P<variant>[ab])_seed(?P<seed>\d+)$")

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

VARIANT_TO_ALPHA = {
    "a": BOXPLOT_ALPHA,
    "b": BOXPLOT_ALPHA_LIGHT,
}

def _config_ids(df: pd.DataFrame) -> list[str]:
    return sorted(df["config_id"].dropna().unique(), key=lambda c: (int(c[:-1]), c[-1]))


def _add_config_id(df: pd.DataFrame) -> pd.DataFrame:
    """config_id is the composite of the group_num / variant named groups."""
    df["config_id"] = df["group_num"].astype(str) + df["variant"]
    return df


def _group_tick_labels(groups: list[int]) -> list[str]:
    return [str(g) for g in groups]


# ---------------------------------------------------------------------------- metrics

def _config_x_positions(config_ids: list[str]) -> dict[str, float]:
    """Place variant 'a' at group_num - BOX_OFFSET, variant 'b' at group_num + BOX_OFFSET."""
    positions = {}
    for cid in config_ids:
        group = int(cid[:-1])
        offset = -BOX_OFFSET if cid[-1] == "a" else +BOX_OFFSET
        positions[cid] = group + offset
    return positions


def draw_metric_boxplot(
    ax,
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    metric: str,
    ylabel: str,
    report: bool = True,
) -> list[dict]:
    """Draw one metric's boxplot group on ``ax`` and return its box statistics.

    Split out of :func:`plot_metric_boxplot` so the standalone PDF and the
    combined grid draw the exact same panel; ``report`` is off for the grid so
    the stats are not printed a second time.
    """
    config_ids = _config_ids(df)
    x_pos = _config_x_positions(config_ids)
    rows: list[dict] = []

    # Baseline box + reference lines
    if baseline_df is not None and not baseline_df.empty:
        draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
        s = boxplot_stats(baseline_df[metric].values)
        if report:
            print(f"  {'baseline':>8}  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
        rows.append({"config_id": "baseline", "metric": metric, **s})
        # for val, ls in [(s["q50"], "--"), (s["q25"], ":"), (s["q75"], ":")]:
        #     ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=BOXPLOT_ALPHA)

    # Per-config boxes
    for cid in config_ids:
        xp = x_pos[cid]
        data = df[df["config_id"] == cid][metric].values
        if len(data) == 0:
            continue
        s = boxplot_stats(data)
        if report:
            print(f"  {cid:>8}  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
        rows.append({"config_id": cid, "metric": metric, **s})
        draw_boxplot(ax, data, position=xp, color=MULTI_SCALE_COLOR, box_width=BOX_WIDTH, alpha=VARIANT_TO_ALPHA[cid[-1]])

    groups = sorted({int(cid[:-1]) for cid in config_ids})
    ax.grid(axis="y")
    ax.yaxis.set_inverted(METRIC_TO_AXIS_REVERS[metric])
    ax.set_xticks([0] + groups)
    ax.set_xticklabels(["C4"] + _group_tick_labels(groups))
    ax.set_xlabel("Observation configuration group")
    ax.set_ylabel(ylabel)
    return rows


def variant_legend_handles() -> list:
    return [
        plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=BOXPLOT_ALPHA, label="Baseline (C4)"),
        plt.Rectangle((0, 0), 1, 1, fc=MULTI_SCALE_COLOR, alpha=BOXPLOT_ALPHA, label="Variant a"),
        plt.Rectangle((0, 0), 1, 1, fc=MULTI_SCALE_COLOR, alpha=BOXPLOT_ALPHA_LIGHT, label="Variant b"),
    ]


def plot_metric_boxplot(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    metric: str,
    ylabel: str,
    scenario: str,
    runs_name: str,
    output_dir: Path,
) -> list[dict]:
    fig, ax = paper_axes(METRIC_WIDTH, METRIC_HEIGHT)
    rows = draw_metric_boxplot(ax, df, baseline_df, metric, ylabel)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_{runs_name}_{scenario}.pdf"
    save(fig, out_path)
    plt.close(fig)
    return rows


def save_legend(output_dir: Path, runs_name: str, scenario: str) -> None:
    fig = plt.figure(figsize=(LEGEND_STRIP_IN, METRIC_HEIGHT * TEXTWIDTH_IN))
    legend = fig.legend(handles=variant_legend_handles(), loc="center left", ncol=1)
    legend.get_frame().set_edgecolor("k")
    out_path = output_dir / f"legend_modes_{runs_name}_{scenario}.pdf"
    save(fig, out_path)
    plt.close(fig)


def plot_metric_grid(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    runs_name: str,
    scenario: str,
    output_dir: Path,
    metrics: Mapping[str, str] = METRIC_TO_CAPTION,
    width: float = GRID_WIDTH,
    ncols: int = GRID_COLS,
) -> Path:
    """One figure holding every grid metric, with the variant legend in the spare cell.

    Replaces the 2x2 block of ``subfigure``s in the paper: laying the panels out
    here means they are guaranteed the same axes size and the same text size,
    and the legend costs nothing extra because it goes in the cell the odd
    number of metrics leaves empty.
    """
    fig, panel_axes, legend_ax = metric_grid(len(metrics), ncols=ncols, width=width)

    for ax, letter, (metric, caption) in zip(panel_axes, PANEL_LETTERS, metrics.items()):
        draw_metric_boxplot(ax, df, baseline_df, metric, METRICS[metric], report=False)
        grid_caption(ax, letter, caption)
    legend_in_cell(fig, legend_ax, variant_legend_handles())

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"metrics_grid_{runs_name}_{scenario}.pdf"
    save(fig, out_path)
    plt.close(fig)
    print("\n" + grid_latex_snippet(out_path, list(metrics.values()), width) + "\n")
    return out_path


def plot_metrics(run_metrics, baseline_metrics, runs_root, scenario, output_dir):
    _add_config_id(run_metrics)
    all_rows: list[dict] = []
    for metric, ylabel in METRICS.items():
        all_rows.extend(plot_metric_boxplot(
            run_metrics, baseline_metrics, metric, ylabel,
            scenario, runs_root.name, output_dir,
        ))
    csv_path = output_dir / f"boxplot_stats_{runs_root.name}_{scenario}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")
    save_legend(output_dir, runs_root.name, scenario)
    plot_metric_grid(run_metrics, baseline_metrics, runs_root.name, scenario, output_dir)


# --------------------------------------------------------------------------- breakdown

VARIANT_OFFSET = {"a": -BOX_OFFSET, "b": +BOX_OFFSET}


def print_success_rates(breakdown: pd.DataFrame, baseline_seed_rates=None) -> None:
    if baseline_seed_rates:
        mean_bl = sum(baseline_seed_rates.values()) / len(baseline_seed_rates)
        print(f"  {'baseline':>8}  success_rate={mean_bl:.1%}  (seeds: {', '.join(f'{v:.1%}' for v in baseline_seed_rates.values())})")
    df = breakdown.copy()
    _add_config_id(df)
    for cid in _config_ids(df):
        rates = df[df["config_id"] == cid]["success_rate"].values
        print(f"  {cid:>8}  success_rate={rates.mean():.1%}  (seeds: {', '.join(f'{r:.1%}' for r in rates)})")


def plot_breakdown(breakdown, baseline_breakdown, baseline_seed_rates, runs_root, scenario, output_dir):
    print_success_rates(breakdown, baseline_seed_rates)
    _add_config_id(breakdown)

    config_ids = _config_ids(breakdown)
    groups = sorted({int(cid[:-1]) for cid in config_ids})

    fig, ax = paper_axes(BREAKDOWN_WIDTH, BREAKDOWN_HEIGHT, right=LEGEND_STRIP_IN)
    seen_reasons: set = set()

    def _bar(x_, h, bottom_, color, reason, alpha):
        hatch = REASON_HATCH.get(reason, "")
        if reason in FILLED_REASONS:
            ax.bar(x_, h, width=BOX_WIDTH, bottom=bottom_, color=color,
                   alpha=alpha, hatch=hatch, edgecolor="black", linewidth=0.5)
        else:
            ax.bar(x_, h, width=BOX_WIDTH, bottom=bottom_, facecolor="none",
                   hatch=hatch, edgecolor="black", linewidth=0.5)

    # --- baseline bar at x=0 ---
    if baseline_breakdown is not None:
        bottom = 0.0
        for reason in [SUCCESS_REASON] + [r for r in baseline_breakdown.index if r != SUCCESS_REASON]:
            frac = baseline_breakdown.get(reason, 0.0)
            if frac <= 0:
                continue
            _bar(0, frac, bottom, BASELINE_COLOR, reason, BOXPLOT_ALPHA)
            bottom += frac
            seen_reasons.add(reason)

    if baseline_seed_rates:
        seeds = sorted(baseline_seed_rates)
        jitter = np.linspace(-0.06, 0.06, len(seeds))
        for jit, seed in zip(jitter, seeds):
            ax.scatter(jit, baseline_seed_rates[seed],
                       color="black", s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.5)

    # --- per-variant stacked bars ---
    min_seed_rates = 1.0
    for variant in ("a", "b"):
        variant_cids = [c for c in config_ids if c.endswith(variant)]
        if not variant_cids:
            continue
        variant_df = breakdown[breakdown["config_id"].isin(variant_cids)]
        ordered, means = mean_breakdowns(variant_df, variant_cids, pos_col="config_id")
        xi = np.array([int(cid[:-1]) for cid in variant_cids])
        bottom = np.zeros(len(variant_cids))
        alpha = VARIANT_TO_ALPHA[variant]
        for reason in ordered:
            _bar(xi + VARIANT_OFFSET[variant], means[reason], bottom, MULTI_SCALE_COLOR, reason, alpha)
            bottom += means[reason]
            seen_reasons.add(reason)

        for cid in variant_cids:
            xi_base = int(cid[:-1]) + VARIANT_OFFSET[variant]
            seed_rates = {row["seed"]: row["success_rate"]
                          for _, row in breakdown[breakdown["config_id"] == cid].iterrows()}
            seeds = sorted(seed_rates)
            jitter = np.linspace(-0.06, 0.06, len(seeds))
            min_seed_rates = min([min_seed_rates, *seed_rates.values()])
            for jit, seed in zip(jitter, seeds):
                ax.scatter(xi_base + jit, seed_rates[seed],
                           color="black", s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                           edgecolors="white", linewidths=0.8)

    ax.set_xticks([0] + groups)
    ax.set_xticklabels(["C4"] + _group_tick_labels(groups))
    ax.set_xlabel("Observation configuration group")
    ax.set_ylabel("Episode outcome fraction")
    ax.grid(axis="y")
    outcome_ylim(ax, min_seed_rates)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    legend_handles = []
    if baseline_breakdown is not None:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=BOXPLOT_ALPHA, label="Baseline (C4)"))
    for variant in ("a", "b"):
        if any(c.endswith(variant) for c in config_ids):
            legend_handles.append(plt.Rectangle(
                (0, 0), 1, 1, fc=MULTI_SCALE_COLOR, alpha=VARIANT_TO_ALPHA[variant], label=f"Variant {variant}"))
    for reason in [r for r in REASON_HATCH if r in seen_reasons]:
        fc = "lightgray" if reason in FILLED_REASONS else "none"
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, fc=fc, hatch=REASON_HATCH[reason], edgecolor="black",
            label=REASON_LABELS.get(reason, reason)))
    legend_handles.append(plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="black",
        markersize=8, label="Per-seed success rate"))
    legend_right(ax, handles=legend_handles, frameon=True, edgecolor="k")

    out_path = output_dir / f"episode_success_{runs_root.name}_{scenario}.pdf"
    save(fig, out_path)
    plt.close(fig)

if __name__ == "__main__":
    args, selected = run_sweep_args_parser()

    runs_root = Path(args.runs_root)
    output_dir = args.output_dir / runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if "metrics" in selected:
        import bluesky as bs
        from bluesky_gym.maps.map_sources import TiffMapSourceConfig
        from bluesky_gym.metrics.evaluation_metrics import build_metric_fn, make_pop_samplers

        bs.init()
        # Fixed-map overview: legacy TiffMapSource branch ignores bounds and is shared
        # across all sweep runs (post-resample clip at the given percentile).
        samplers = make_pop_samplers(
            TiffMapSourceConfig(file_path=args.map_path), bounds=None,
            clip_percentile=args.noise_clip_percentile, train_resampling="cubic_spline", true_resampling="average")
        calculate_metrics = build_metric_fn(samplers)

        cache_path = runs_root / f"cached_metrics_{args.scenario}.csv"
        if args.cache and cache_path.exists():
            print("Using cached metrics...")
            run_metrics = pd.read_csv(cache_path)
        else:
            run_metrics = collect_run_metrics(
                runs_root, PATTERN, args.scenario, calculate_metrics, args.mean_episode_length)
            if args.cache:
                print(f"Saving metrics to {cache_path} ...")
                run_metrics.to_csv(cache_path, index=False)

        baseline_metrics = None
        if args.baseline:
            baseline_cache_path = runs_root / f"cached_baseline_metrics_{args.scenario}.csv"
            if args.cache and baseline_cache_path.exists():
                print("Using cached baseline metrics...")
                baseline_metrics = pd.read_csv(baseline_cache_path)
            else:
                baseline_metrics = collect_baseline_metrics(
                    list(args.baseline), args.scenario, calculate_metrics, args.mean_episode_length)
                if args.cache:
                    print(f"Saving baseline metrics to {baseline_cache_path} ...")
                    baseline_metrics.to_csv(baseline_cache_path, index=False)

        for frame in (run_metrics, baseline_metrics):
            if frame is not None and not frame.empty:
                frame["combined"] = frame["normalized_fuel"] + frame["normalized_noise"]
                add_reward(frame)

        plot_metrics(run_metrics, baseline_metrics, runs_root, args.scenario, output_dir)

    if "breakdown" in selected:
        breakdown = collect_breakdown_data(runs_root, PATTERN, args.scenario)
        if not breakdown.empty:
            baseline_breakdown = None
            baseline_seed_rates = {}
            if args.baseline:
                baseline_breakdown = collect_baseline_breakdown(args.baseline, args.scenario)
                baseline_seed_rates = collect_baseline_seed_rates(args.baseline, args.scenario)
                if baseline_breakdown is None:
                    print(f"Baseline — no usable trajectory data in {args.baseline} (plotting without baseline)")
            plot_breakdown(breakdown, baseline_breakdown, baseline_seed_rates, runs_root, args.scenario, output_dir)
        else:
            print("No breakdown data found. Run generate_trajectories.py on the sweep runs first.")
