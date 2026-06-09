import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import bluesky as bs
from tqdm import tqdm

from bluesky_gym.metrics.evaluation_metrics import build_metric_fn
from scripts.common.colors import BASELINE_COLOR, qual
from scripts.common.sweep_plotting import (
    add_reward,
    compute_episode_metrics,
    draw_boxplot,
    find_csv,
    find_run_dirs,
)

BOX_WIDTH = 0.6

# Caption per transform variant. Key order also defines the left-to-right
# plot order of the boxes.
VARIANT_TO_CAPTION = {
    "baseline": "Baseline",
    "scale": "Scale [1, 7.6]",
    "power": "Power [0.52, 0.70]",
    "floor": "Floor [0, 40.2]",
    "zoom": "Zoom [1x - 2x]",
    "flip": "Flip",
    "flip_zoom": "Flip + Zoom",
    "power_flip": "Power + Flip",
    "power_zoom": "Power + Zoom",
    "power_flip_zoom": "Power + Flip + Zoom",
}

VARIANT_ORDER = list(VARIANT_TO_CAPTION)

# transformed_{variant}_seed{N}, e.g. transformed_power_flip_zoom_seed2
RUN_PATTERN = re.compile(r"^transformed_(.+)_seed(\d+)$")


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
        variant, seed = match.group(1), int(match.group(2))

        csv = find_csv(run_dir, runway)
        if not csv:
            print(f"  Skipping {run_dir.name}: no CSV found")
            continue
        df = pd.read_csv(csv)
        df = calculate_metrics(df)
        metrics = compute_episode_metrics(df, mean_episode_length)
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
        metrics["variant"] = "baseline_ref"
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
    present = set(df["variant"].dropna().unique())
    variants = [v for v in VARIANT_ORDER if v in present]
    # Any variants not in VARIANT_TO_CAPTION get appended so nothing is dropped.
    variants += sorted(present - set(VARIANT_TO_CAPTION))

    fig, ax = plt.subplots(figsize=(14, 5))
    legend_handles = []

    # Reference baseline box + quartile lines spanning the plot for comparison.
    has_baseline = baseline_df is not None and not baseline_df.empty
    if has_baseline:
        draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=0.6, label="Baseline (C4)")
        )
        q1, median, q3 = baseline_df[metric].quantile([0.25, 0.5, 0.75])
        for val, ls in [(median, "--"), (q1, ":"), (q3, ":")]:
            ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=0.6)

    # One box per transform variant.
    for i, variant in enumerate(variants):
        data = df[df["variant"] == variant][metric].values
        if len(data) == 0:
            continue
        draw_boxplot(ax, data, position=i + 1, color=qual(i), box_width=BOX_WIDTH)

    tick_x = ([0] if has_baseline else []) + [i + 1 for i in range(len(variants))]
    tick_labels = (["Baseline\n(C4)"] if has_baseline else []) + [
        VARIANT_TO_CAPTION.get(v, v) for v in variants
    ]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if legend_handles:
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

    arg_parser = argparse.ArgumentParser(description="Plot transform sweep metrics")
    arg_parser.add_argument("runs_root", type=str, help="path to runs root (e.g. runs/transforms)")
    arg_parser.add_argument("--baseline_run", nargs="+", type=str, help="path to C4 baseline run directory")
    arg_parser.add_argument("--runway", type=str, default="EHAM_RW27", help="runway to use")
    arg_parser.add_argument("--map-path", type=str, default="./scripts/population_maps/europe_3035_1km.tif")
    arg_parser.add_argument("--cache", action="store_true", default=False)
    arg_parser.add_argument("--noise_clip_percentile", type=float, default=99.9)
    arg_parser.add_argument("--mean_episode_length", type=float, default=1400.0)
    arg_parser.add_argument("--output_dir", type=Path, default=Path("plots/sweep_overview_plots"))
    args = arg_parser.parse_args()

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
