import re
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from scripts.common.colors import SEED_COLORS

SUCCESS_REASON = "success"
# Stack order (bottom → top) and styling for the outcome breakdown. "success"
REASON_ORDER = ["success", "failed_approach", "max_steps", "out_of_bounds"]
REASON_LABELS = {
    "success": "Success",
    "failed_approach": "Failed approach",
    "max_steps": "Max steps",
    "out_of_bounds": "Out of bounds",
}

def find_csvs(run_dir: Path, runway: str) -> list[Path]:
    """All trajectory CSVs matching *{runway}_map* under run_dir."""
    return list(run_dir.glob(f"trajectories/*{runway}_map*/trajectories.csv"))


def compute_episode_metrics(df: pd.DataFrame, mean_episode_length: float) -> pd.DataFrame:
    """Aggregate per-episode metrics from a trajectory dataframe.

    df must already have metric columns from build_metric_fn applied.
    Returns a DataFrame indexed by start_angle with columns:
    fuel, noise, normalized_fuel, normalized_noise, normalized_noise_clipped, success.
    """
    if "termination_reason" not in df.columns:
        df = df.copy()
        df["termination_reason"] = SUCCESS_REASON
    g = df.groupby("start_angle")
    fuel = g["calculated_fuel"].sum()
    noise = g["calculated_noise"].sum()
    noise_clipped = g["calculated_noise_clipped"].sum()
    success = g["termination_reason"].last() == SUCCESS_REASON
    mean_noise_ref = g["mean_reference_noise"].first() * mean_episode_length
    norm_fuel = fuel / (g["mean_fuel_flow"].first() * mean_episode_length)
    norm_noise = noise / mean_noise_ref
    norm_noise_clipped = noise_clipped / mean_noise_ref
    return pd.DataFrame({
        "fuel": fuel,
        "noise": noise,
        "normalized_fuel": norm_fuel,
        "normalized_noise": norm_noise,
        "normalized_noise_clipped": norm_noise_clipped,
        "success": success,
    })

def add_reward(df: pd.DataFrame, fuel_weight: float = 0.5) -> None:
    """Add a per-episode reward column in place.

    Reward = (+5 if success else -1) with normalized fuel and noise entering
    negatively, since they are costs the agent is penalized for. The noise term
    uses the clipped variant so it matches the training reward (clip_noise_reward).
    """
    success_bonus = 5.0
    failure_penalty = -1.0
    success_term = df["success"].map({True: success_bonus, False: failure_penalty})
    df["reward"] = success_term - (fuel_weight * df["normalized_fuel"]) - ((1 - fuel_weight) * df["normalized_noise_clipped"])


def draw_boxplot(ax, data, position, color, box_width) -> None:
    ax.boxplot(
        data,
        positions=[position],
        widths=box_width,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor=color, alpha=0.6),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        flierprops=dict(marker="o", color=color, alpha=0.4, markersize=3),
    )



def per_episode_reasons(run_dir: Path, runway: str) -> pd.Series | None:
    """Per-episode termination_reason pooled across all *_map CSVs in a run."""
    csvs = find_csvs(run_dir, runway)
    if not csvs:
        return None
    episodes = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        if "termination_reason" not in df.columns:
            print(f"  Warning: no termination_reason column in {csv_path}, skipping")
            continue
        per_episode = df.groupby("start_angle")["termination_reason"].last()
        episodes.append(per_episode)
    return pd.concat(episodes) if episodes else None


def compute_success_rate(run_dir: Path, runway: str) -> float | None:
    """Mean success rate across all *_map trajectory CSVs in a run."""
    reasons = per_episode_reasons(run_dir, runway)
    if reasons is None:
        return None
    return (reasons == SUCCESS_REASON).mean()


def compute_termination_breakdown(run_dir: Path, runway: str) -> pd.Series | None:
    """Fraction of episodes per termination reason, pooled across *_map CSVs."""
    reasons = per_episode_reasons(run_dir, runway)
    if reasons is None:
        return None
    return reasons.value_counts(normalize=True)


def seed_color_map(df: pd.DataFrame) -> dict:
    all_seeds = sorted(df["seed"].unique())
    return {seed: SEED_COLORS[i % len(SEED_COLORS)] for i, seed in enumerate(all_seeds)}


def seed_legend(ax, color_map: dict) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, label=f"Seed {s}")
        for s, c in color_map.items()
    ]
    ax.legend(handles=handles, frameon=False)

def find_csv(run_dir: Path, runway: str) -> Path | None:
    """Return the preferred trajectory CSV for runway, or None.

    When multiple CSVs match, prefers the exact {runway}_map_best directory
    over variants such as {runway}_map_best_modified.
    """
    csvs = find_csvs(run_dir, runway)
    if not csvs:
        return None
    if len(csvs) == 1:
        return csvs[0]

    print(f"  [{run_dir.name}] Multiple CSVs for '{runway}':")
    for csv in sorted(csvs):
        print(f"    {csv.parent.name}/trajectories.csv")

    preferred_name = f"{runway}_map_best"
    exact = [csv for csv in csvs if csv.parent.name == preferred_name]
    chosen = exact[0] if exact else sorted(csvs)[0]
    print(f"  → Using: {chosen.parent.name}/trajectories.csv")
    return chosen


def find_run_dirs(run_pattern: None | list[str], runs_root: Path) -> Generator:
    """Yield sorted run directories, optionally filtered by name patterns."""
    for run_dir in sorted(runs_root.iterdir()):
        if not run_pattern:
            pass
        elif not any(pattern in run_dir.name for pattern in run_pattern):
            continue
        yield run_dir


def compute_episode_length(run_dir: Path, runway: str) -> pd.Series | None:
    """Episode lengths (seconds) across all *_map trajectory CSVs in a run."""
    csvs = find_csvs(run_dir, runway)
    if not csvs:
        return None
    episodes = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        per_episode = df.groupby("start_angle")["sim_dt"].sum()
        episodes.append(per_episode)
    return pd.concat(episodes) if episodes else None


def compute_baseline(baseline_dir: Path, runway: str) -> tuple[float | None, float | None]:
    """Return (success_rate, mean_episode_length_s) for a baseline run directory."""
    lengths = compute_episode_length(baseline_dir, runway)
    return compute_success_rate(baseline_dir, runway), (lengths.mean() if lengths is not None else None)


def mean_breakdowns(df: pd.DataFrame, positions: list, pos_col: str = "resolution") -> tuple[list, dict]:
    """Mean outcome fractions per position value, averaged across seeds.

    df must have a "breakdown" column of pd.Series (from compute_termination_breakdown).
    pos_col is the df column whose values are given in positions (e.g. "resolution").
    Returns (ordered_reasons, {reason: np.array over positions}).
    """
    present_reasons = set()
    for termination_reason_breakdown in df["breakdown"]:
        if termination_reason_breakdown is not None:
            present_reasons.update(termination_reason_breakdown.index)
    present_reasons.discard("none")
    ordered_reasons = [reason for reason in REASON_ORDER if reason in present_reasons] + sorted(present_reasons - set(REASON_ORDER))

    means = {reason: np.zeros(len(positions)) for reason in ordered_reasons}
    for i, position in enumerate(positions):
        reason_breakdowns = [breakdown for breakdown in df[df[pos_col] == position]["breakdown"] if breakdown is not None]
        if not reason_breakdowns:
            continue
        mean_breakdown = pd.concat(reason_breakdowns, axis=1).fillna(0.0).mean(axis=1)
        for reason in ordered_reasons:
            means[reason][i] = mean_breakdown.get(reason, 0.0)
    return ordered_reasons, means


# =====================================================================================
# Generic, regex-driven data collection
#
# A "sweep" is just the set of runs whose directory name matches one regex. The regex's
# named groups become DataFrame columns: e.g.
#   r"^transformed_(?P<variant>.+)_seed(?P<seed>\d+)$"
# yields columns "variant" and "seed". All-digit groups are cast to int. Everything
# cosmetic — ordering, colors, x-layout, labels, savefig — lives in the per-sweep
# plotting script, NOT here.
# =====================================================================================


def _coerce(value: str | None):
    """Cast an all-digit regex group to int; leave everything else as-is."""
    return int(value) if value is not None and value.isdigit() else value


def collect_run_metrics(
    runs_root: Path,
    pattern: re.Pattern,
    runway: str,
    calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame],
    mean_episode_length: float,
) -> pd.DataFrame:
    """Per-episode metrics for every run whose name matches `pattern`.

    Each named group of `pattern` is added as a column (all-digit groups become ints).
    """
    frames = []
    for run_dir in tqdm(sorted(runs_root.iterdir())):
        match = pattern.search(run_dir.name)
        if not match:
            continue
        csv = find_csv(run_dir, runway)
        if not csv:
            print(f"  Skipping {run_dir.name}: no CSV found")
            continue
        df = calculate_metrics(pd.read_csv(csv))
        metrics = compute_episode_metrics(df, mean_episode_length)
        for key, value in match.groupdict().items():
            metrics[key] = _coerce(value)
        frames.append(metrics)
    return pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()


def collect_baseline_metrics(
    baseline_runs: list[Path],
    runway: str,
    calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame],
    mean_episode_length: float,
) -> pd.DataFrame:
    """Per-episode metrics pooled across the reference baseline run(s)."""
    frames = []
    for baseline_run in baseline_runs:
        seed_match = re.search(r"seed(\d+)", baseline_run.name)
        csv = find_csv(baseline_run, runway)
        if not csv:
            print(f"  Skipping {baseline_run.name}: no CSV found")
            continue
        df = calculate_metrics(pd.read_csv(csv))
        metrics = compute_episode_metrics(df, mean_episode_length)
        metrics["seed"] = int(seed_match.group(1)) if seed_match else 0
        frames.append(metrics)
    return pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()


def collect_breakdown_data(runs_root: Path, pattern: re.Pattern, runway: str) -> pd.DataFrame:
    """Success rate, termination breakdown and episode lengths per matching run.

    Named groups of `pattern` become columns, same as collect_run_metrics. Does not
    need the metric fn (reads only termination_reason / sim_dt), so it is cheap to call
    straight from a plotting script that wants the outcome / length panels.
    """
    records = []
    for run_dir in sorted(runs_root.iterdir()):
        match = pattern.search(run_dir.name)
        if not match:
            continue
        rate = compute_success_rate(run_dir, runway)
        if rate is None:
            print(f"  Skipping {run_dir.name}: no usable trajectory data")
            continue
        record = {key: _coerce(value) for key, value in match.groupdict().items()}
        record["success_rate"] = rate
        record["breakdown"] = compute_termination_breakdown(run_dir, runway)
        record["length"] = compute_episode_length(run_dir, runway)
        records.append(record)
    return pd.DataFrame(records)


def run_sweep_cli(
    pattern: re.Pattern,
    *,
    plot_metrics: Callable | None = None,
    plot_breakdown: Callable | None = None,
) -> None:
    """Shared CLI for a sweep that has a metrics view and/or an outcome-breakdown view.

    `--plots {both,metrics,breakdown}` (default both) selects which to draw. Only the
    metrics path needs BlueSky + the noise/fuel metric fn, so a breakdown-only run is
    dependency-free and fast. Each per-sweep script supplies the cosmetics through:

        plot_metrics(run_metrics, baseline_metrics, runs_root, runway, output_dir)
        plot_breakdown(breakdown, baseline_rate, baseline_length, runs_root, runway, output_dir)

    A view the script doesn't provide is silently skipped. `--baseline` is shared: the
    full list is pooled into the metric reference boxes, its first entry supplies the
    breakdown success-rate / episode-length reference lines.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Plot sweep metrics and/or outcome breakdown")
    parser.add_argument("runs_root", type=str, help="path to runs root (e.g. runs/transforms)")
    parser.add_argument("--plots", choices=["both", "metrics", "breakdown"], default="both",
                        help="which views to draw (default: both)")
    parser.add_argument("--baseline", nargs="+", type=Path, default=None,
                        help="baseline run directory/-ies (pooled for the metric boxes; "
                             "the first is used for the breakdown reference lines)")
    parser.add_argument("--runway", type=str, default="EHAM_RW27")
    parser.add_argument("--map-path", type=str, default="./scripts/population_maps/europe_3035_1km.tif")
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--noise_clip_percentile", type=float, default=99.9)
    parser.add_argument("--mean_episode_length", type=float, default=1400.0)
    parser.add_argument("--output_dir", type=Path, default=Path("plots/sweep_overview_plots"))
    args = parser.parse_args()

    selected = {"metrics", "breakdown"} if args.plots == "both" else {args.plots}
    if "metrics" in selected and plot_metrics is None:
        print("This sweep has no metrics view; skipping it.")
        selected.discard("metrics")
    if "breakdown" in selected and plot_breakdown is None:
        print("This sweep has no breakdown view; skipping it.")
        selected.discard("breakdown")
    if not selected:
        return

    runs_root = Path(args.runs_root)
    output_dir = args.output_dir / runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if "metrics" in selected:
        import bluesky as bs
        from bluesky_gym.metrics.evaluation_metrics import build_metric_fn

        bs.init()
        calculate_metrics = build_metric_fn(Path(args.map_path), args.noise_clip_percentile)

        cache_path = runs_root / f"cached_metrics_{args.runway}.csv"
        if args.cache and cache_path.exists():
            print("Using cached metrics...")
            run_metrics = pd.read_csv(cache_path)
        else:
            run_metrics = collect_run_metrics(
                runs_root, pattern, args.runway, calculate_metrics, args.mean_episode_length)
            if args.cache:
                print(f"Saving metrics to {cache_path} ...")
                run_metrics.to_csv(cache_path, index=False)

        baseline_metrics = None
        if args.baseline:
            baseline_metrics = collect_baseline_metrics(
                list(args.baseline), args.runway, calculate_metrics, args.mean_episode_length)

        for frame in (run_metrics, baseline_metrics):
            if frame is not None and not frame.empty:
                frame["combined"] = frame["normalized_fuel"] + frame["normalized_noise"]
                add_reward(frame)

        plot_metrics(run_metrics, baseline_metrics, runs_root, args.runway, output_dir)

    if "breakdown" in selected:
        breakdown = collect_breakdown_data(runs_root, pattern, args.runway)
        if breakdown.empty:
            print("No breakdown data found. Run generate_trajectories.py on the sweep runs first.")
            return
        baseline_rate = baseline_length = None
        if args.baseline:
            baseline_rate, baseline_length = compute_baseline(args.baseline[0], args.runway)
            if baseline_rate is None:
                print(f"Baseline — no usable trajectory data in {args.baseline[0]} (plotting without baseline)")
            else:
                print(f"Baseline — success rate: {baseline_rate:.1%}, mean length: {baseline_length:.1f} s")
        plot_breakdown(breakdown, baseline_rate, baseline_length, runs_root, args.runway, output_dir)


