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

from scripts.common.colors import (
    BASELINE_COLOR,
    FALLBACK_REASON_COLOR,
    MODE_COLORS,
    REASON_COLORS,
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

plt.rcParams["font.size"] = 12

BOX_OFFSET = 0.2
BOX_WIDTH = 0.35
BAR_WIDTH = 0.5
DOT_ALPHA = 0.8
DOT_SIZE = 60
BAR_ALPHA = 0.6

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

    fig, ax = plt.subplots(figsize=(8, 5))
    legend_handles = []
    rows: list[dict] = []

    if baseline_df is not None and not baseline_df.empty:
        draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=0.6, label="No-map baseline"))
        s = boxplot_stats(baseline_df[metric].values)
        print(f"  {'baseline':>8}       N/A  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
        rows.append({"mode": "baseline", "resolution": float("nan"), "metric": metric, **s})

    mode_config = [
        ("centered", MODE_COLORS["centered"], -BOX_OFFSET),
        ("forward",  MODE_COLORS["forward"],  +BOX_OFFSET),
    ]
    for mode, color, offset in mode_config:
        mode_df = df[df["mode"] == mode]
        for res in resolutions:
            data = mode_df[mode_df["resolution"] == res][metric].values
            s = boxplot_stats(data)
            print(f"  {mode:>8}  {res:>3} km/px  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
            rows.append({"mode": mode, "resolution": res, "metric": metric, **s})
            if len(data) == 0:
                continue
            draw_boxplot(ax, data, position=x_positions[res] + offset, color=color, box_width=BOX_WIDTH)
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6, label=mode.capitalize()))

    ax.set_xticks([0] + list(range(1, len(resolutions) + 1)))
    ax.set_xticklabels(["No map"] + [f"{r} km/px" for r in resolutions])
    ax.set_xlabel("Observation resolution")
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend_handles, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_{runs_name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)
    return rows


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

    fig, ax = plt.subplots(figsize=(max(8, (len(resolutions) + 1) * 1.5), 5))

    mode_config = [
        ("centered", MODE_COLORS["centered"], -BOX_OFFSET),
        ("forward",  MODE_COLORS["forward"],  +BOX_OFFSET),
    ]

    reason_handles: dict = {}

    # --- baseline bar at x=0 ---
    if baseline_breakdown is not None:
        bottom = 0.0
        success_frac = baseline_breakdown.get(SUCCESS_REASON, 0.0)
        ax.bar(0, success_frac, width=BOX_WIDTH, bottom=bottom,
               color=BASELINE_COLOR, alpha=BAR_ALPHA)
        bottom += success_frac
        for reason, frac in baseline_breakdown.items():
            if reason == SUCCESS_REASON or frac <= 0:
                continue
            bar_color = REASON_COLORS.get(reason, FALLBACK_REASON_COLOR)
            ax.bar(0, frac, width=BOX_WIDTH, bottom=bottom, color=bar_color, alpha=BAR_ALPHA)
            bottom += frac
            if reason not in reason_handles:
                reason_handles[reason] = plt.Rectangle(
                    (0, 0), 1, 1, fc=bar_color, alpha=BAR_ALPHA,
                    label=REASON_LABELS.get(reason, reason))

    if baseline_seed_rates:
        seeds = sorted(baseline_seed_rates)
        jitter = np.linspace(-0.06, 0.06, len(seeds))
        for jit, seed in zip(jitter, seeds):
            ax.scatter(jit, baseline_seed_rates[seed],
                       color='black', s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.8)

    # --- per-mode stacked bars ---
    for mode, color, offset in mode_config:
        mode_df = breakdown[breakdown["mode"] == mode]
        if mode_df.empty:
            continue

        mode_resolutions = sorted(mode_df["resolution"].unique())
        xi = np.array([resolutions.index(r) + 1 for r in mode_resolutions])
        ordered, means = mean_breakdowns(mode_df, mode_resolutions)
        bottom = np.zeros(len(mode_resolutions))
        for reason in ordered:
            bar_color = color if reason == SUCCESS_REASON else REASON_COLORS.get(reason, FALLBACK_REASON_COLOR)
            ax.bar(xi + offset, means[reason], width=BOX_WIDTH, bottom=bottom,
                   color=bar_color, alpha=BAR_ALPHA)
            bottom += means[reason]
            if reason != SUCCESS_REASON and reason not in reason_handles:
                reason_handles[reason] = plt.Rectangle(
                    (0, 0), 1, 1, fc=bar_color, alpha=BAR_ALPHA,
                    label=REASON_LABELS.get(reason, reason))

        for res in mode_resolutions:
            xi_base = resolutions.index(res) + 1 + offset
            seed_rates = {row["seed"]: row["success_rate"]
                          for _, row in mode_df[mode_df["resolution"] == res].iterrows()}
            seeds = sorted(seed_rates)
            jitter = np.linspace(-0.06, 0.06, len(seeds))
            for jit, seed in zip(jitter, seeds):
                ax.scatter(xi_base + jit, seed_rates[seed],
                           color='black', s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                           edgecolors="white", linewidths=0.8)

    ax.set_xticks([0] + list(x))
    ax.set_xticklabels(["No map"] + [f"{r} km/px" for r in resolutions])
    ax.set_xlabel("Observation resolution")
    ax.set_ylabel("Episode outcome fraction")
    ax.grid(axis="y")
    ax.set_ylim(0.8, 1.01)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = []
    if baseline_breakdown is not None:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=BAR_ALPHA, label="No-map baseline"))
    legend_handles += [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=BAR_ALPHA, label=mode.capitalize())
        for mode, color, _ in mode_config
        if not breakdown[breakdown["mode"] == mode].empty
    ]
    legend_handles.extend(reason_handles.values())
    legend_handles.append(plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="black",
        markersize=8, label="Per-seed success rate"))
    ax.legend(handles=legend_handles, frameon=True, framealpha=1.0, edgecolor="none")

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
            baseline_metrics = collect_baseline_metrics(
                list(args.baseline), args.scenario, calculate_metrics, args.mean_episode_length)

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
