import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bluesky as bs
from tqdm import tqdm

from bluesky_gym.metrics.evaluation_metrics import build_metric_fn
from scripts.common.colors import BASELINE_COLOR, SEED_COLORS, qual
from scripts.common.sweep_plotting import (
    add_reward,
    compute_episode_metrics,
    draw_boxplot,
    find_csv,
    find_run_dirs,
)

BOX_WIDTH = 0.6
DOT_ALPHA = 0.8
DOT_SIZE = 40
GROUP_GAP = 0.5  # extra x-space inserted between groups

VARIANT_TO_OBSERVATION = {
    "baseline": "C4",
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

_GROUP_BASE_COLORS = {g: qual(i) for i, g in enumerate(range(1, 6))}


def _group_color(group_num: int, variant: str) -> tuple:
    base = _GROUP_BASE_COLORS[group_num]
    if variant == "b":
        return tuple(c + (1 - c) * 0.45 for c in base[:3]) + (base[3],)
    return base


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


def collect_metrics(
    runs_root: Path,
    runway: str,
    calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame],
    mean_episode_length: float,
) -> pd.DataFrame:
    frames = []
    for run_dir in tqdm(list(find_run_dirs(None, runs_root))):
        match = RUN_PATTERN.search(run_dir.name)
        if not match:
            continue
        group_num, variant, seed = int(match.group(1)), match.group(2), int(match.group(3))

        csv = find_csv(run_dir, runway)
        if not csv:
            print(f"  Skipping {run_dir.name}: no CSV found")
            continue
        df = pd.read_csv(csv)
        df = calculate_metrics(df)
        metrics = compute_episode_metrics(df, mean_episode_length)
        metrics["config_id"] = f"{group_num}{variant}"
        metrics["group_num"] = group_num
        metrics["variant"] = variant
        metrics["seed"] = seed
        frames.append(metrics)
    return pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()


def collect_baseline_metrics(
    baseline_runs: list[Path],
    runway: str,
    calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame],
    mean_episode_length: float,
) -> pd.DataFrame:
    frames = []
    for baseline_run in baseline_runs:
        seed_match = re.search(r"seed(\d+)", baseline_run.name)
        seed = int(seed_match.group(1)) if seed_match else 0

        csv = find_csv(baseline_run, runway)
        if not csv:
            print(f"  Skipping {baseline_run.name}: no CSV found")
            continue
        df = pd.read_csv(csv)
        df = calculate_metrics(df)
        metrics = compute_episode_metrics(df, mean_episode_length)
        metrics["config_id"] = "baseline"
        metrics["seed"] = seed
        frames.append(metrics)
    return pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()


def plot_metric_boxplot(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    metric: str,
    ylabel: str,
    runway: str,
    runs_name: str,
    output_dir: Path = Path("./plots"),
) -> None:
    config_ids = sorted(df["config_id"].dropna().unique(), key=lambda c: (int(c[:-1]), c[-1]))
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

    # Per-config boxes + seed dots
    prev_group = None
    for cid in config_ids:
        group_num = int(cid[:-1])
        variant = cid[-1]
        color = _group_color(group_num, variant)
        xp = x_pos[cid]

        # Vertical divider between groups
        if prev_group is not None and (group_num != prev_group) or prev_group == "baseline":
            ax.axvline(xp - (1.0 + GROUP_GAP) / 2, color="#cccccc", linewidth=0.8, zorder=0)
        prev_group = group_num

        cid_df = df[df["config_id"] == cid]
        data = cid_df[metric].values
        if len(data) == 0:
            continue
        draw_boxplot(ax, data, position=xp, color=color, box_width=BOX_WIDTH)

        # seeds = sorted(cid_df["seed"].unique())
        # jitter = np.linspace(-0.1, 0.1, len(seeds)) if len(seeds) > 1 else [0.0]
        # for xi, seed in zip(jitter, seeds):
            # seed_mean = cid_df[cid_df["seed"] == seed][metric].mean()
            # ax.scatter(xp + xi, seed_mean,
            #            color=SEED_COLORS[seed % len(SEED_COLORS)],
            #            s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
            #            edgecolors="white", linewidths=0.8)

    tick_x = [0] + [x_pos[cid] for cid in config_ids]
    tick_labels = ["Baseline\n(C4)"] + [
        f"{cid}\n{VARIANT_TO_OBSERVATION.get(cid, cid)}" for cid in config_ids
    ]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend_handles, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_{runs_name}_{runway}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Plot multi-scale sweep metrics")
    arg_parser.add_argument("runs_root", type=str, help="path to runs root")
    arg_parser.add_argument("--baseline_run", nargs="+", type=str, help="path to C4 baseline run directory")
    arg_parser.add_argument("--runway", type=str, default="EHAM_RW27", help="runway to use")
    arg_parser.add_argument("--map-path", type=str, default="./scripts/population_maps/europe_3035_1km.tif")
    arg_parser.add_argument("--cache", action="store_true", default=False)
    arg_parser.add_argument("--noise_clip_percentile", type=float, default=99.9)
    arg_parser.add_argument("--mean_episode_length", type=float, default=1400.0)
    arg_parser.add_argument("--output_dir", type=Path, default=Path("plots/sweep_overview_plots"))
    args = arg_parser.parse_args()

    RUN_PATTERN = re.compile(r"^(?:multi_scale_)?(\d)([ab])_seed(\d+)$")

    bs.init()
    calculate_metrics = build_metric_fn(Path(args.map_path), args.noise_clip_percentile)

    runs_root = Path(args.runs_root)
    output_dir = args.output_dir / runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = runs_root / f"cached_metrics_{args.runway}.csv"
    if cache_path.exists() and args.cache:
        print("Using the cached metrics...")
        run_metrics = pd.read_csv(cache_path)
    else:
        run_metrics = collect_metrics(runs_root, args.runway, calculate_metrics, args.mean_episode_length)
        if args.cache:
            print(f"Saving results to {cache_path} ...")
            run_metrics.to_csv(cache_path)

    if args.baseline_run:
        baseline_metrics = collect_baseline_metrics(
            [Path(p) for p in args.baseline_run], args.runway, calculate_metrics, args.mean_episode_length
        )
    else:
        baseline_metrics = None

    run_metrics["combined"] = run_metrics["normalized_fuel"] + run_metrics["normalized_noise"]
    if baseline_metrics is not None and not baseline_metrics.empty:
        baseline_metrics["combined"] = baseline_metrics["normalized_fuel"] + baseline_metrics["normalized_noise"]

    add_reward(run_metrics)
    if baseline_metrics is not None and not baseline_metrics.empty:
        add_reward(baseline_metrics)

    for metric, ylabel in [
        ("fuel", "fuel [kg]"),
        ("noise", "noise [W·s]"),
        ("normalized_fuel", "normalized fuel"),
        ("normalized_noise", "normalized noise"),
        ("combined", "normalized fuel + noise"),
        ("reward", "reward"),
    ]:
        plot_metric_boxplot(
            run_metrics, baseline_metrics, metric, ylabel,
            args.runway, runs_root.name, output_dir,
        )
