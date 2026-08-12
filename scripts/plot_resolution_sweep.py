"""
Plot the forward/centered resolution sweep: noise/fuel metric boxplots and/or
episode-outcome breakdown for runs named {forward|centered}_{resolution}_seed{NN}
(resolution in km/pixel: 1, 2, 4, 8, 16, 32).

    python -m scripts.plot_resolution_sweep <runs_root> --baseline <no_map_run>
    python -m scripts.plot_resolution_sweep <runs_root> --plots breakdown

--plots {both,metrics,breakdown} selects the views (default both). Only the
metrics view needs BlueSky + the noise/fuel metric fn.

The metrics view writes one PDF per metric *and* `metrics_grid_*.pdf`: the
reward/noise/fuel panels laid out as one matplotlib figure, the mode legend in
the cell the odd panel count leaves free, and the (a), (b), … captions above
the panels. That grid replaces the 2x2 block of `subfigure`s in the paper —
same reason as plot_trajectory_figure: the panels are laid out in inches here,
so they are guaranteed the same axes size and the same text size, and LaTeX
never rescales them. The `figure` environment to paste is printed at the end.

The breakdown stacks each bar by termination reason (success / failed approach /
max steps / out of bounds), so the success segment height is the arrival rate and
the segments above show *why* the rest failed.
"""

import re
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
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
    seed_legend, run_sweep_args_parser, collect_run_metrics, collect_baseline_metrics, add_reward,
    collect_breakdown_data, compute_baseline, collect_baseline_breakdown, collect_baseline_seed_rates,
)

METRIC_WIDTH, METRIC_HEIGHT = PLOT_TYPE_TO_SIZE["sweep_metric"]
BREAKDOWN_WIDTH, BREAKDOWN_HEIGHT = PLOT_TYPE_TO_SIZE["sweep_breakdown"]
LEGEND_STRIP_IN = 1.7

GRID_COLS = 2
GRID_WIDTH = W_FULL

BOX_OFFSET = 0.2
BOX_WIDTH = 0.35
BAR_WIDTH = 0.5
DOT_ALPHA = 0.8
DOT_SIZE = 40

# {sweep_N_}{forward|centered}_{resolution}_seed{NN}
PATTERN = re.compile(r"^(?:sweep_\d+_)?(?P<mode>forward|centered)_(?P<resolution>\d+)_seed(?P<seed>\d+)$")

MODE_TO_OFFSET = {
    "baseline": 0,
    "centered": -1 * BOX_OFFSET,
    "forward":  BOX_OFFSET,
}

MODE_TO_COLOR = {
    "baseline": BASELINE_COLOR,
    "centered": CENTERED_COLOR,
    "forward":  FORWARD_COLOR,
}

# ---------------------------------------------------------------------------- metrics

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
    resolutions = sorted(df["resolution"].dropna().unique())
    # baseline at 0, resolutions start at 1
    x_positions = {res: i + 1 for i, res in enumerate(resolutions)}
    rows: list[dict] = []

    for mode in MODE_TO_OFFSET:
        mode_df = df[df["mode"] == mode]
        if mode != "baseline":
            for res in resolutions:
                data = mode_df[mode_df["resolution"] == res][metric].values
                s = boxplot_stats(data)
                if report:
                    print(f"  {mode:>8}  {res:>3} km/px  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
                rows.append({"mode": mode, "resolution": res, "metric": metric, **s})
                if len(data) == 0:
                    continue
                draw_boxplot(ax, data, position=x_positions[res] + MODE_TO_OFFSET[mode], color=MODE_TO_COLOR[mode], box_width=BOX_WIDTH, alpha=BOXPLOT_ALPHA)
        else:
            draw_boxplot(ax, baseline_df[metric].values, position=MODE_TO_OFFSET["baseline"], color=MODE_TO_COLOR["baseline"], box_width=BOX_WIDTH)
            s = boxplot_stats(baseline_df[metric].values)
            if report:
                print(f"  {'baseline':>8}       N/A  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
            rows.append({"mode": "baseline", "resolution": float("nan"), "metric": metric, **s})

    ax.set_xticks([0] + list(range(1, len(resolutions) + 1)))
    ax.set_xticklabels(["No map"] + [f"{r}" for r in resolutions])
    ax.set_xlabel("Observation resolution [km/px]")
    ax.yaxis.set_inverted(METRIC_TO_AXIS_REVERS[metric])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    return rows


def mode_legend_handles() -> list:
    return [
        plt.Rectangle((0, 0), 1, 1, fc=MODE_TO_COLOR[mode], alpha=BOXPLOT_ALPHA,
                      label=mode.capitalize())
        for mode in MODE_TO_OFFSET
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


def save_mode_legend(output_dir: Path, runs_name: str, scenario: str) -> None:
    """Export the mode legend (Baseline / Centered / Forward) as a standalone PDF.

    Sized LEGEND_STRIP_IN wide by exactly the height of a metric panel, so it can
    be included next to one at its natural size: neither is rescaled, and the
    legend text matches the in-plot text size exactly.
    """
    fig = plt.figure(figsize=(LEGEND_STRIP_IN, METRIC_HEIGHT * TEXTWIDTH_IN))
    legend = fig.legend(handles=mode_legend_handles(), loc="center left", ncol=1)

    legend.get_frame().set_edgecolor('k')

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
    """One figure holding every grid metric, with the mode legend in the spare cell.

    Replaces the 2x2 block of ``subfigure``s in the paper: laying the panels out
    here means they are guaranteed the same axes size and the same text size,
    and the legend costs nothing extra because it goes in the cell the odd
    number of metrics leaves empty.
    """
    fig, panel_axes, legend_ax = metric_grid(len(metrics), ncols=ncols, width=width)

    rows: list[dict] = []
    for ax, letter, (metric, caption) in zip(panel_axes, PANEL_LETTERS, metrics.items()):
        rows.extend(draw_metric_boxplot(ax, df, baseline_df, metric, METRICS[metric], report=False))
        grid_caption(ax, letter, caption)
    legend_in_cell(fig, legend_ax, mode_legend_handles())

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"metrics_grid_{runs_name}_{scenario}.pdf"
    save(fig, out_path)
    plt.close(fig)
    print("\n" + grid_latex_snippet(out_path, list(metrics.values()), width) + "\n")
    return out_path


def plot_metrics(run_metrics, baseline_metrics, runs_root, scenario, output_dir):
    all_rows: list[dict] = []
    for metric, ylabel in METRICS.items():
        all_rows.extend(plot_metric_boxplot(
            run_metrics, baseline_metrics, metric, ylabel,
            scenario, runs_root.name, output_dir,
        ))
    csv_path = output_dir / f"boxplot_stats_{runs_root.name}_{scenario}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")
    save_mode_legend(output_dir, runs_root.name, scenario)
    plot_metric_grid(run_metrics, baseline_metrics, runs_root.name, scenario, output_dir)


def print_success_rates(breakdown: pd.DataFrame, baseline_breakdown=None, baseline_seed_rates=None) -> None:
    if baseline_seed_rates:
        mean_bl = sum(baseline_seed_rates.values()) / len(baseline_seed_rates)
        print(f"  {'baseline':>8}       N/A  success_rate={mean_bl:.1%}  (seeds: {', '.join(f'{v:.1%}' for v in baseline_seed_rates.values())})")
    for mode in ("centered", "forward"):
        mode_df = breakdown[breakdown["mode"] == mode]
        if mode_df.empty:
            continue
        for res in sorted(mode_df["resolution"].unique()):
            rates = mode_df[mode_df["resolution"] == res]["success_rate"].values
            print(f"  {mode:>8}  {res:>3} km/px  success_rate={rates.mean():.1%}  (seeds: {', '.join(f'{r:.1%}' for r in rates)})")


def plot_breakdown(breakdown, baseline_breakdown, baseline_seed_rates, runs_root, scenario, output_dir):
    print_success_rates(breakdown, baseline_breakdown, baseline_seed_rates)
    resolutions = sorted(breakdown["resolution"].unique())
    # baseline at 0, resolutions start at 1 — mirrors the metric boxplot layout
    x = np.arange(1, len(resolutions) + 1)
    fig, ax = paper_axes(BREAKDOWN_WIDTH, BREAKDOWN_HEIGHT, right=LEGEND_STRIP_IN)

    seen_reasons: set = set()

    def _bar(ax_, x_, h, bottom_, color, reason):
        hatch = REASON_HATCH.get(reason, "")
        if reason in FILLED_REASONS:
            ax_.bar(x_, h, width=BOX_WIDTH, bottom=bottom_, color=color,
                    alpha=BOXPLOT_ALPHA, hatch=hatch, edgecolor="black", linewidth=0.5)
        else:
            ax_.bar(x_, h, width=BOX_WIDTH, bottom=bottom_, facecolor="none",
                    hatch=hatch, edgecolor="black", linewidth=0.5)

    # --- baseline bar at x=0 ---
    if baseline_breakdown is not None:
        bottom = 0.0
        for reason in [SUCCESS_REASON] + [r for r in baseline_breakdown.index if r != SUCCESS_REASON]:
            frac = baseline_breakdown.get(reason, 0.0)
            if frac <= 0:
                continue
            _bar(ax, 0, frac, bottom, BASELINE_COLOR, reason)
            bottom += frac
            seen_reasons.add(reason)

    if baseline_seed_rates:
        seeds = sorted(baseline_seed_rates)
        jitter = np.linspace(-0.06, 0.06, len(seeds))
        for jit, seed in zip(jitter, seeds):
            ax.scatter(jit, baseline_seed_rates[seed],
                       color='black', s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.5)

    # --- per-mode stacked bars ---
    min_seed_rates = 1.0
    for mode in MODE_TO_OFFSET:
        mode_df = breakdown[breakdown["mode"] == mode]
        if mode_df.empty:
            continue

        mode_resolutions = sorted(mode_df["resolution"].unique())
        xi = np.array([resolutions.index(r) + 1 for r in mode_resolutions])
        ordered, means = mean_breakdowns(mode_df, mode_resolutions)
        bottom = np.zeros(len(mode_resolutions))
        for reason in ordered:
            _bar(ax, xi + MODE_TO_OFFSET[mode], means[reason], bottom, MODE_TO_COLOR[mode], reason)
            bottom += means[reason]
            seen_reasons.add(reason)
        for res in mode_resolutions:
            xi_base = resolutions.index(res) + 1 + MODE_TO_OFFSET[mode]
            seed_rates = {row["seed"]: row["success_rate"]
                          for _, row in mode_df[mode_df["resolution"] == res].iterrows()}
            seeds = sorted(seed_rates)
            jitter = np.linspace(-0.06, 0.06, len(seeds))
            min_seed_rates = min([min_seed_rates, *seed_rates.values()])
            for jit, seed in zip(jitter, seeds):
                ax.scatter(xi_base + jit, seed_rates[seed],
                           color='black', s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                           edgecolors="white", linewidths=0.8)

    ax.set_xticks([0] + list(x))
    ax.set_xticklabels(["No map"] + [f"{r}" for r in resolutions])
    ax.set_xlabel("Observation resolution [km/px]")
    ax.set_ylabel("Episode outcome fraction")
    ax.grid(axis="y")
    outcome_ylim(ax, min_seed_rates)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    legend_handles = []
    if baseline_breakdown is not None:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=BOXPLOT_ALPHA, label="No-map baseline"))
    legend_handles += [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=BOXPLOT_ALPHA, label=mode.capitalize())
        for mode, color in MODE_TO_COLOR.items()
        if not breakdown[breakdown["mode"] == mode].empty
    ]
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
        cache_path = runs_root / f"cached_metrics_{args.scenario}.csv"
        baseline_cache_path = runs_root / f"cached_baseline_metrics_{args.scenario}.csv"
        cached_metrics = args.cache and cache_path.exists()
        cached_baseline = args.cache and baseline_cache_path.exists()
        calculate_metrics = None
        if not cached_metrics or (args.baseline and not cached_baseline):
            import bluesky as bs
            from bluesky_gym.maps.map_sources import TiffMapSourceConfig
            from bluesky_gym.metrics.evaluation_metrics import build_metric_fn, make_pop_samplers

            bs.init()
            samplers = make_pop_samplers(
                TiffMapSourceConfig(file_path=args.map_path), bounds=None,
                clip_percentile=args.noise_clip_percentile, train_resampling="cubic_spline", true_resampling="average")
            calculate_metrics = build_metric_fn(samplers)

        if cached_metrics:
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
            if cached_baseline:
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
