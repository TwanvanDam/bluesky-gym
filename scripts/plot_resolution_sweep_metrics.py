import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator

import matplotlib.pyplot as plt
import pandas as pd
import bluesky as bs
from tqdm import tqdm

from bluesky_gym.metrics.evaluation_metrics import build_metric_fn
from scripts.common.colors import BASELINE_COLOR, MODE_COLORS

@dataclass
class Record:
    mode: str
    fuel: float
    noise: float
    normalized_fuel: float
    normalized_noise: float
    normalized_noise_clipped: float
    success: bool
    resolution: float | None
    seed: int

def find_run_dirs(run_pattern: None | list[str], runs_root: Path) -> Generator:
    for run_dir in sorted(runs_root.iterdir()):
        if not run_pattern:
            pass
        elif not any(pattern in run_dir.name for pattern in run_pattern):
            continue
        yield run_dir

def find_csv(run_dir: Path, runway: str) -> None | Path:
    csvs = list(run_dir.glob(f"trajectories/*{runway}_map*/trajectories.csv"))
    if not csvs:
        return None
    if len(csvs) == 1:
        return csvs[0]
    else:
        print("More than one csv found for " + runway)
        return csvs[0]

def collect_metrics(runs_root: Path, runway: str, run_pattern: None| list[str], calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame], mean_episode_length: float) -> pd.DataFrame:
    run_dirs = find_run_dirs(run_pattern, runs_root)
    records = []
    for run_dir in tqdm(list(run_dirs)):
        m = RUN_PATTERN.search(run_dir.name)
        if not m:
            print(f"  Skipping {run_dir.name}: cannot parse mode/resolution/seed")
            continue
        mode, resolution, seed = m.group(1), int(m.group(2)), int(m.group(3))

        csv = find_csv(run_dir, runway)
        print(csv)
        if not csv:
            continue
        df = pd.read_csv(csv)
        df = calculate_metrics(df)
        df_grouped = df.groupby("start_angle")
        fuel_summed = df_grouped["calculated_fuel"].sum()
        noise_summed = df_grouped["calculated_noise"].sum()
        noise_clipped_summed = df_grouped["calculated_noise_clipped"].sum()
        success_per_episode = df_grouped["termination_reason"].last() == SUCCESS_REASON
        mean_noise_ref = df_grouped["mean_reference_noise"].first() * mean_episode_length
        normalized_fuel = fuel_summed / (df_grouped["mean_fuel_flow"].first() * mean_episode_length)
        normalized_noise = noise_summed / mean_noise_ref
        normalized_noise_clipped = noise_clipped_summed / mean_noise_ref

        for start_angle in fuel_summed.index:
            records.append(
                Record(
                    mode=mode,
                    resolution=resolution,
                    fuel=fuel_summed[start_angle],
                    noise=noise_summed[start_angle],
                    normalized_fuel=normalized_fuel[start_angle],
                    normalized_noise=normalized_noise[start_angle],
                    normalized_noise_clipped=normalized_noise_clipped[start_angle],
                    success=success_per_episode[start_angle],
                    seed=seed
                )
            )
    return pd.DataFrame(records)

def collect_baseline_metrics(baseline_run: Path, runway: str, calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame], mean_episode_length: float) -> pd.DataFrame:
    seed_match = re.search(r"seed(\d+)", baseline_run.name)
    seed = int(seed_match.group(1)) if seed_match else 0

    csv = find_csv(baseline_run, runway)
    if not csv:
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = calculate_metrics(df)
    if "termination_reason" not in df.columns:
        df["termination_reason"] = "success"
    df_grouped = df.groupby("start_angle")
    fuel_summed = df_grouped["calculated_fuel"].sum()
    noise_summed = df_grouped["calculated_noise"].sum()
    noise_clipped_summed = df_grouped["calculated_noise_clipped"].sum()
    success_per_episode = df_grouped["termination_reason"].last() == SUCCESS_REASON
    mean_noise_ref = df_grouped["mean_reference_noise"].first() * mean_episode_length
    normalized_fuel = fuel_summed / (df_grouped["mean_fuel_flow"].first() * mean_episode_length)
    normalized_noise = noise_summed / mean_noise_ref
    normalized_noise_clipped = noise_clipped_summed / mean_noise_ref
    combined = normalized_fuel + normalized_noise
    records = []
    for start_angle in fuel_summed.index:
        records.append(
            Record(
                mode="no_map",
                resolution=None,
                fuel=fuel_summed[start_angle],
                noise=noise_summed[start_angle],
                normalized_fuel=normalized_fuel[start_angle],
                normalized_noise=normalized_noise[start_angle],
                normalized_noise_clipped=normalized_noise_clipped[start_angle],
                success=success_per_episode[start_angle],
                seed=seed,
            )
        )
    return pd.DataFrame(records)

def add_reward(df: pd.DataFrame) -> None:
    """Add a per-episode reward column in place.

    Reward = (+5 if success else -1) with normalized fuel and noise entering
    negatively, since they are costs the agent is penalized for. The noise term
    uses the clipped variant so it matches the training reward (clip_noise_reward).
    """
    success_bonus = 5.0
    failure_penalty = -1.0
    success_term = df["success"].map({True: success_bonus, False: failure_penalty})
    df["reward"] = success_term - df["normalized_fuel"] - df["normalized_noise_clipped"]


def _draw_boxplot(ax, data, position, color):
    ax.boxplot(
        data,
        positions=[position],
        widths=BOX_WIDTH,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor=color, alpha=0.6),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        flierprops=dict(marker="o", color=color, alpha=0.4, markersize=3),
    )


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
        _draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR)
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
            _draw_boxplot(ax, data, position=x_positions[res] + offset, color=color)
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
    SUCCESS_REASON = "success"

    # rendering defaults
    BOX_OFFSET = 0.2
    BOX_WIDTH = 0.35

    bs.init()
    calculate_metrics = build_metric_fn(Path(args.map_path), args.noise_clip_percentile)

    runs_root = Path(args.runs_root)
    output_dir = args.output_dir / runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = args.baseline

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

    plot_metric_boxplot(run_metrics, baseline_metrics, "fuel", "fuel [kg]", args.runway, output_dir)
    plot_metric_boxplot(run_metrics, baseline_metrics, "noise", "noise [W·s]", args.runway, output_dir)
    plot_metric_boxplot(run_metrics, baseline_metrics, "normalized_fuel", "normalized fuel", args.runway, output_dir)
    plot_metric_boxplot(run_metrics, baseline_metrics, "normalized_noise", "normalized noise", args.runway, output_dir)
    plot_metric_boxplot(run_metrics, baseline_metrics, "combined", "normalized fuel + noise", args.runway, output_dir)
    plot_metric_boxplot(run_metrics, baseline_metrics, "reward", "reward", args.runway, output_dir)
