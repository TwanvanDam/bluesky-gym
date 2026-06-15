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
    draw_boxplot,
    mean_breakdowns,
    seed_color_map,
    seed_legend, run_sweep_args_parser, collect_run_metrics, collect_baseline_metrics, add_reward,
    collect_breakdown_data, compute_baseline,
)

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
    ("reward_unclipped", "reward (no noise clipping"),
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
) -> None:
    resolutions = sorted(df["resolution"].dropna().unique())
    # baseline at 0, resolutions start at 1
    x_positions = {res: i + 1 for i, res in enumerate(resolutions)}

    fig, ax = plt.subplots(figsize=(8, 5))
    legend_handles = []

    if baseline_df is not None and not baseline_df.empty:
        draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=0.6, label="No-map baseline"))
        q1, median, q3 = baseline_df[metric].quantile([0.25, 0.5, 0.75])
        for val, ls in [(median, "--"), (q1, ":"), (q3, ":")]:
            ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=0.6)

    mode_config = [
        ("centered", MODE_COLORS["centered"], -BOX_OFFSET),
        ("forward",  MODE_COLORS["forward"],  +BOX_OFFSET),
    ]
    for mode, color, offset in mode_config:
        mode_df = df[df["mode"] == mode]
        for res in resolutions:
            data = mode_df[mode_df["resolution"] == res][metric].values
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

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_{runs_name}_{scenario}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def plot_metrics(run_metrics, baseline_metrics, runs_root, scenario, output_dir):
    for metric, ylabel in METRICS:
        plot_metric_boxplot(
            run_metrics, baseline_metrics, metric, ylabel,
            scenario, runs_root.name, output_dir,
        )


# --------------------------------------------------------------------------- breakdown

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


def plot_breakdown(breakdown, baseline_rate, baseline_length, runs_root, scenario, output_dir):
    for mode in ("forward", "centered"):
        mode_df = breakdown[breakdown["mode"] == mode]
        if mode_df.empty:
            print(f"No data for mode '{mode}', skipping.")
            continue

        color = MODE_COLORS[mode]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_episode_success(ax, mode_df, mode, color, baseline=baseline_rate)
        fig.tight_layout()
        out_path = output_dir / f"episode_success_{runs_root.name}_{mode}_{scenario}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_mode_length(ax, mode_df, mode, color, baseline=baseline_length)
        fig.tight_layout()
        out_path = output_dir / f"episode_length_{runs_root.name}_{mode}_{scenario}.png"
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
            baseline_rate = baseline_length = None
            if args.baseline:
                baseline_rate, baseline_length = compute_baseline(args.baseline[0], args.scenario)
                if baseline_rate is None:
                    print(f"Baseline — no usable trajectory data in {args.baseline[0]} (plotting without baseline)")
                else:
                    print(f"Baseline — success rate: {baseline_rate:.1%}, mean length: {baseline_length:.1f} s")
            plot_breakdown(breakdown, baseline_rate, baseline_length, runs_root, args.scenario, output_dir)
        else:
            print("No breakdown data found. Run generate_trajectories.py on the sweep runs first.")
