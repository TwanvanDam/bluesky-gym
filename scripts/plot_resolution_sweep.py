"""
Plot the forward/centered resolution sweep: noise/fuel metric boxplots and/or
episode-outcome breakdown for runs named {forward|centered}_{resolution}_seed{NN}
(resolution in km/pixel: 1, 2, 4, 8, 16, 32).

    python -m scripts.plot_resolution_sweep <runs_root> --baseline <no_map_run>
    python -m scripts.plot_resolution_sweep <runs_root> --plots breakdown

--plots {both,metrics,breakdown} selects the views (default both). Only the
metrics view needs BlueSky + the noise/fuel metric fn.

The breakdown stacks each bar by termination reason (success / failed approach /
max steps / out of bounds), so the success segment height is the arrival rate and
the segments above show *why* the rest failed.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.colors import *
from scripts.common.sweep_plotting import (
    REASON_LABELS,
    SUCCESS_REASON,
    boxplot_stats,
    draw_boxplot,
    mean_breakdowns,
    seed_legend, run_sweep_args_parser, collect_run_metrics, collect_baseline_metrics, add_reward,
    collect_breakdown_data, compute_baseline, collect_baseline_breakdown, collect_baseline_seed_rates,
)

plt.rcParams["font.size"] = 12

# Source width of every metric figure, in inches. Figures are included at
# \textwidth in LaTeX, so the legend is exported at this same width and included
# at \textwidth too — both scale by the same factor, keeping text sizes matched.
PLOT_WIDTH_IN = 3.16

BOX_OFFSET = 0.2
BOX_WIDTH = 0.35
BAR_WIDTH = 0.5
DOT_ALPHA = 0.8
DOT_SIZE = 40

# {sweep_N_}{forward|centered}_{resolution}_seed{NN}
PATTERN = re.compile(r"^(?:sweep_\d+_)?(?P<mode>forward|centered)_(?P<resolution>\d+)_seed(?P<seed>\d+)$")

METRICS = [
    ("fuel", "fuel [kg]"),
    ("noise", "noise [W·s]"),
    ("normalized_fuel", "normalized fuel"),
    ("normalized_noise", "normalized noise"),
    ("combined", "normalized fuel + noise"),
    ("reward", "reward"),
    ("reward_unclipped", "reward (no noise clipping)"),
]

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

REASON_HATCH = {
    "success":        "",
    "failed_approach": "////",
    "max_steps":       "....",
    "out_of_bounds":   "xxxx",
}

# ---------------------------------------------------------------------------- metrics

def plot_metric_boxplot(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    metric: str,
    ylabel: str,
    scenario: str,
    runs_name: str,
    output_dir: Path,
) -> list[dict]:
    resolutions = sorted(df["resolution"].dropna().unique())
    # baseline at 0, resolutions start at 1
    x_positions = {res: i + 1 for i, res in enumerate(resolutions)}

    fig, ax = plt.subplots(figsize=(0.49 * TEXTWIDTH_IN, 0.49 * TEXTWIDTH_IN * 0.78), constrained_layout=True)
    rows: list[dict] = []

    for mode in MODE_TO_OFFSET:
        mode_df = df[df["mode"] == mode]
        if mode != "baseline":
            for res in resolutions:
                data = mode_df[mode_df["resolution"] == res][metric].values
                s = boxplot_stats(data)
                print(f"  {mode:>8}  {res:>3} km/px  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
                rows.append({"mode": mode, "resolution": res, "metric": metric, **s})
                if len(data) == 0:
                    continue
                draw_boxplot(ax, data, position=x_positions[res] + MODE_TO_OFFSET[mode], color=MODE_TO_COLOR[mode], box_width=BOX_WIDTH, alpha=BOXPLOT_ALPHA)
        else:
            draw_boxplot(ax, baseline_df[metric].values, position=MODE_TO_OFFSET["baseline"], color=MODE_TO_COLOR["baseline"], box_width=BOX_WIDTH)
            s = boxplot_stats(baseline_df[metric].values)
            print(f"  {'baseline':>8}       N/A  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
            rows.append({"mode": "baseline", "resolution": float("nan"), "metric": metric, **s})

    ax.set_xticks([0] + list(range(1, len(resolutions) + 1)))
    ax.set_xticklabels(["No map"] + [f"{r}" for r in resolutions])
    ax.set_xlabel("Observation resolution [km/px]")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_{runs_name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)
    return rows


def save_mode_legend(output_dir: Path, runs_name: str, scenario: str) -> None:
    """Export the mode legend (Baseline / Centered / Forward) as a standalone PDF.

    Rendered at PLOT_WIDTH_IN — the same source width as every metric figure — so
    that when both are included at \\textwidth in LaTeX they scale by the same
    factor and the legend text matches the in-plot text size exactly. The save box
    keeps the full figure width but is cropped tight in height.
    """
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=MODE_TO_COLOR[mode], alpha=BOXPLOT_ALPHA,
                      label=mode.capitalize())
        for mode in MODE_TO_OFFSET
    ]
    fig = plt.figure(figsize=(1, 0.49 * TEXTWIDTH_IN * 0.6))
    legend = fig.legend(handles=handles, loc="center left", ncol=1)

    legend.get_frame().set_edgecolor('k')

    out_path = output_dir / f"legend_modes_{runs_name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


def plot_metrics(run_metrics, baseline_metrics, runs_root, scenario, output_dir):
    all_rows: list[dict] = []
    for metric, ylabel in METRICS:
        all_rows.extend(plot_metric_boxplot(
            run_metrics, baseline_metrics, metric, ylabel,
            scenario, runs_root.name, output_dir,
        ))
    csv_path = output_dir / f"boxplot_stats_{runs_root.name}_{scenario}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")
    save_mode_legend(output_dir, runs_root.name, scenario)


# --------------------------------------------------------------------------- breakdown

def _draw_baseline(ax, value: float | None, label: str) -> None:
    if value is not None:
        ax.axhline(value, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline ({label})", zorder=3)
        ax.legend(frameon=False)

def plot_breakdown(breakdown, baseline_breakdown, baseline_seed_rates, runs_root, scenario, output_dir):
    resolutions = sorted(breakdown["resolution"].unique())
    # baseline at 0, resolutions start at 1 — mirrors the metric boxplot layout
    x = np.arange(1, len(resolutions) + 1)
    textwidth = 469
    plot_width_in = textwidth / 72.7
    fig, ax = plt.subplots(figsize=(plot_width_in, 0.4 * plot_width_in), constrained_layout=True)


    seen_reasons: set = set()

    def _bar(ax_, x_, h, bottom_, color, reason):
        hatch = REASON_HATCH.get(reason, "")
        ax_.bar(x_, h, width=BOX_WIDTH, bottom=bottom_, color=color,
                alpha=BOXPLOT_ALPHA, hatch=hatch, edgecolor="black", linewidth=0.5)

    # --- baseline bar at x=0 ---
    if baseline_breakdown is not None:
        bottom = 0.0
        for reason in [SUCCESS_REASON] + [r for r in baseline_breakdown if r != SUCCESS_REASON]:
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
            for jit, seed in zip(jitter, seeds):
                ax.scatter(xi_base + jit, seed_rates[seed],
                           color='black', s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                           edgecolors="white", linewidths=0.8)

    ax.set_xticks([0] + list(x))
    ax.set_xticklabels(["No map"] + [f"{r}" for r in resolutions])
    ax.set_xlabel("Observation resolution [km/px]")
    ax.set_ylabel("Episode outcome fraction")
    ax.grid(axis="y")
    ax.set_ylim(0.87, 1.01)
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
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, fc="lightgray", hatch=REASON_HATCH[reason], edgecolor="black",
            label=REASON_LABELS.get(reason, reason)))
    legend_handles.append(plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="black",
        markersize=8, label="Per-seed success rate"))
    ax.legend(handles=legend_handles, frameon=True, edgecolor="k", loc="center left", bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    out_path = output_dir / f"episode_success_{runs_root.name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
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
