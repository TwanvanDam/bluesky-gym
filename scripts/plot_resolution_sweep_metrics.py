import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import bluesky as bs
from tqdm import tqdm

from bluesky_gym.metrics.evaluation_metrics import build_metric_fn
from scripts.common.colors import BASELINE_COLOR, MODE_COLORS
from scripts.common.sweep_plotting import (
    add_reward,
    compute_episode_metrics,
    draw_boxplot,
    find_csv,
    find_run_dirs,
)

BOX_OFFSET = 0.2
BOX_WIDTH = 0.35


def collect_metrics(runs_root: Path, runway: str, run_pattern: None | list[str], calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame], mean_episode_length: float) -> pd.DataFrame:
    run_dirs = find_run_dirs(run_pattern, runs_root)
    frames = []
    for run_dir in tqdm(list(run_dirs)):
        match = RUN_PATTERN.search(run_dir.name)
        if not match:
            print(f"  Skipping {run_dir.name}: cannot parse mode/resolution/seed")
            continue
        mode, resolution, seed = match.group(1), int(match.group(2)), int(match.group(3))

        csv = find_csv(run_dir, runway)
        if not csv:
            print(f"  Skipping {run_dir.name}: no CSV found")
            continue
        df = pd.read_csv(csv)
        df = calculate_metrics(df)
        metrics = compute_episode_metrics(df, mean_episode_length)
        metrics["mode"] = mode
        metrics["resolution"] = resolution
        metrics["seed"] = seed
        frames.append(metrics)
    return pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()

def collect_baseline_metrics(baseline_run: Path, runway: str, calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame], mean_episode_length: float) -> pd.DataFrame:
    seed_match = re.search(r"seed(\d+)", baseline_run.name)
    seed = int(seed_match.group(1)) if seed_match else 0

    csv = find_csv(baseline_run, runway)
    if not csv:
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = calculate_metrics(df)
    metrics = compute_episode_metrics(df, mean_episode_length)
    metrics["mode"] = "no_map"
    metrics["resolution"] = None
    metrics["seed"] = seed
    return metrics.reset_index(drop=True)

def plot_metric_boxplot(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    metric: str,
    ylabel: str,
    runway: str,
    output_dir: Path = Path("./plots"),
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
        # mode_df = mode_df[mode_df["success"] == True]
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
    out_path = output_dir / f"{metric}_{runs_root.name}_{runway}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description="Plot resolution sweep metrics")
    arg_parser.add_argument("runs_root", type=str, help="path to runs root for comparison")
    arg_parser.add_argument("--baseline_run", nargs="+", type=str, help="path to baseline runs root")
    arg_parser.add_argument("--runway", type=str, default="EHAM_RW27", help="runway to use for comparison")
    arg_parser.add_argument("--map-path", type=str, default="./scripts/population_maps/europe_3035_1km.tif", help="path to map source")
    arg_parser.add_argument("--cache", action="store_true", default=False, help="whether to cache results")
    arg_parser.add_argument("--noise_clip_percentile", type=float, default=99.9, help="noise clip percentile")
    arg_parser.add_argument("--mean_episode_length", type=float, default=1400.0, help="Used to normalize the reward values")
    arg_parser.add_argument("--output_dir", type=Path, default=Path("plots/sweep_overview_plots"), help=f"Output directory for the plots")
    args = arg_parser.parse_args()

    RUN_PATTERN = re.compile(r"(forward|centered)_(\d+)_seed(\d+)")

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
        run_metrics = collect_metrics(runs_root, args.runway, ["forward", "centered"], calculate_metrics, args.mean_episode_length)
        if args.cache:
            print(f"Saving results to {cache_path} ...")
            run_metrics.to_csv(cache_path)

    if args.baseline_run:
        baseline_metrics = collect_baseline_metrics(Path(args.baseline_run[0]), args.runway, calculate_metrics, args.mean_episode_length)
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

