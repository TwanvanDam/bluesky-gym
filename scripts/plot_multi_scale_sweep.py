"""
Plot the multi-scale observation sweep: noise/fuel metric boxplots and/or
episode-outcome breakdown for runs named multi_scale_{group}{variant}_seed{NN}
(group 1-5, variant a/b).

    python -m scripts.plot_multi_scale_sweep <runs_root> --baseline <C4_run>
    python -m scripts.plot_multi_scale_sweep <runs_root> --plots breakdown

--plots {both,metrics,breakdown} selects the views (default both). Only the
metrics view needs BlueSky + the noise/fuel metric fn.
"""

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
    draw_boxplot,
    mean_breakdowns,
    seed_color_map, compute_baseline, collect_breakdown_data, add_reward, collect_baseline_metrics, collect_run_metrics,
    run_sweep_args_parser,
)

BOX_WIDTH = 0.6
BAR_WIDTH = 0.7
GROUP_GAP = 0.5  # extra x-space inserted between groups
DOT_ALPHA = 0.8
DOT_SIZE = 60
BAR_ALPHA = 0.6

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

# One color per group; variant b gets a lighter shade.
_GROUP_BASE_COLORS = {g: qual(i) for i, g in enumerate(range(1, 6))}

METRICS = [
    ("fuel", "fuel [kg]"),
    ("noise", "noise [W·s]"),
    ("normalized_fuel", "normalized fuel"),
    ("normalized_noise", "normalized noise"),
    ("combined", "normalized fuel + noise"),
    ("reward", "reward"),
]


def _group_color(group_num: int, variant: str) -> tuple:
    base = _GROUP_BASE_COLORS[group_num]
    if variant == "b":
        return tuple(c + (1 - c) * 0.45 for c in base[:3]) + (base[3],)
    return base


def _config_ids(df: pd.DataFrame) -> list[str]:
    return sorted(df["config_id"].dropna().unique(), key=lambda c: (int(c[:-1]), c[-1]))


def _add_config_id(df: pd.DataFrame) -> pd.DataFrame:
    """config_id is the composite of the group_num / variant named groups."""
    df["config_id"] = df["group_num"].astype(str) + df["variant"]
    return df


def _tick_labels(config_ids: list[str]) -> list[str]:
    return [f"{cid}\n{VARIANT_TO_OBSERVATION.get(cid, cid)}" for cid in config_ids]


# ---------------------------------------------------------------------------- metrics

def _config_x_positions(config_ids: list[str]) -> dict[str, float]:
    """Assign x positions with a gap between each group pair."""
    positions = {}
    x = 1.0
    prev_group = None
    for cid in config_ids:
        group = int(cid[:-1])
        if prev_group is not None and group != prev_group:
            x += GROUP_GAP
        positions[cid] = x
        x += 1.0
        prev_group = group
    return positions


def plot_metric_boxplot(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    metric: str,
    ylabel: str,
    scenario: str,
    runs_name: str,
    output_dir: Path,
) -> None:
    config_ids = _config_ids(df)
    x_pos = _config_x_positions(config_ids)

    fig, ax = plt.subplots(figsize=(14, 5))
    legend_handles = []

    # Baseline box + reference lines
    if baseline_df is not None and not baseline_df.empty:
        draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=0.6, label="Baseline (C4)")
        )
        q1, median, q3 = baseline_df[metric].quantile([0.25, 0.5, 0.75])
        for val, ls in [(median, "--"), (q1, ":"), (q3, ":")]:
            ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=0.6)

    # Per-config boxes
    prev_group = None
    for cid in config_ids:
        group_num = int(cid[:-1])
        variant = cid[-1]
        xp = x_pos[cid]

        # Vertical divider between groups
        if prev_group is not None and group_num != prev_group:
            ax.axvline(xp - (1.0 + GROUP_GAP) / 2, color="#cccccc", linewidth=0.8, zorder=0)
        prev_group = group_num

        data = df[df["config_id"] == cid][metric].values
        if len(data) == 0:
            continue
        draw_boxplot(ax, data, position=xp, color=_group_color(group_num, variant), box_width=BOX_WIDTH)

    tick_x = [0] + [x_pos[cid] for cid in config_ids]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(["Baseline\n(C4)"] + _tick_labels(config_ids), fontsize=8)
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
    _add_config_id(run_metrics)
    for metric, ylabel in METRICS:
        plot_metric_boxplot(
            run_metrics, baseline_metrics, metric, ylabel,
            scenario, runs_root.name, output_dir,
        )


# --------------------------------------------------------------------------- breakdown

def plot_episode_success(ax, df: pd.DataFrame, baseline: float | None = None) -> None:
    config_ids = _config_ids(df)
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

    ax.set_xticks(x)
    ax.set_xticklabels(_tick_labels(config_ids), fontsize=8)
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
    config_ids = _config_ids(df)
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
            ax.bar(x[i], np.mean(all_lengths), width=BAR_WIDTH,
                   color=_group_color(int(cid[:-1]), cid[-1]), alpha=BAR_ALPHA)

    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline ({baseline:.0f} s)", zorder=3)
        ax.legend(frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels(_tick_labels(config_ids), fontsize=8)
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


def plot_breakdown(breakdown, baseline_rate, baseline_length, runs_root, scenario, output_dir):
    _add_config_id(breakdown)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_episode_success(ax, breakdown, baseline=baseline_rate)
    fig.tight_layout()
    out_path = output_dir / f"episode_success_{runs_root.name}_{scenario}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_episode_length(ax, breakdown, baseline=baseline_length)
    fig.tight_layout()
    out_path = output_dir / f"episode_length_{runs_root.name}_{scenario}.png"
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
