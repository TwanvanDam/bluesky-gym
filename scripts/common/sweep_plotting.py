import re
import warnings
from argparse import Namespace
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

def find_csv(run_dir: Path, scenario: str) -> Path | None:
    """Trajectory CSV for one exact evaluation scenario, or None.

    `scenario` is the trajectory subdir name written by generate_trajectories,
    i.e. {runway}_{label} (e.g. "EHAM_RW27",
    "EHAM_RW18R_scaling"). Selection is exact: the scenario label is the
    key, so multiple scenarios for the same runway never collide.
    """
    csv = run_dir / "trajectories" / scenario / "trajectories.csv"
    return csv if csv.exists() else None


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
    reason = g["termination_reason"].last()
    success = reason == SUCCESS_REASON
    mean_noise_ref = g["mean_reference_noise"].first() * mean_episode_length
    norm_fuel = fuel / (g["mean_fuel_flow"].first() * mean_episode_length)
    norm_noise = noise / mean_noise_ref
    norm_noise_clipped = noise_clipped / mean_noise_ref
    return pd.DataFrame({
        "fuel": fuel,
        "noise": noise,
        "noise_clipped": noise_clipped,
        "normalized_fuel": norm_fuel,
        "normalized_noise": norm_noise,
        "normalized_noise_clipped": norm_noise_clipped,
        "termination_reason": reason,
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
    df["reward_unclipped"] = success_term - (fuel_weight * df["normalized_fuel"]) - ((1 - fuel_weight) * df["normalized_noise"])


def draw_boxplot(ax, data, position, color, box_width,alpha=0.6, showfliers=True) -> None:
    ax.boxplot(
        data,
        positions=[position],
        widths=box_width,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor=color, alpha=alpha),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        showfliers=showfliers,
        flierprops=dict(marker="o", color=color, alpha=0.4, markersize=3),
    )


def boxplot_stats(data) -> dict:
    """Box statistics matching matplotlib's default boxplot (whis=1.5).

    Returns the quartiles, the IQR, and the whisker caps — the most extreme data
    points still within [Q1 - 1.5*IQR, Q3 + 1.5*IQR], exactly where draw_boxplot
    renders the whiskers. NaNs are dropped; an all-empty input yields all-NaN.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return {"q10": np.nan,"q25": np.nan, "q50": np.nan, "q75": np.nan, "q90": np.nan,
                "iqr": np.nan, "whisker_lo": np.nan, "whisker_hi": np.nan}
    q10, q25, q50, q75, q90 = np.percentile(data, [10, 25, 50, 75, 90])
    iqr = q75 - q25
    whisker_lo = data[data >= q25 - 1.5 * iqr].min()
    whisker_hi = data[data <= q75 + 1.5 * iqr].max()
    return {"q10": q10, "q25": q25, "q50": q50, "q75": q75, "q90" : q90,
            "iqr": iqr, "whisker_lo": whisker_lo, "whisker_hi": whisker_hi}



def add_scenario_id(df: pd.DataFrame, group_cols: list[str]) -> bool:
    """Add a `scenario_id` column aligning the same evaluation scenario across runs.

    Every run evaluates the same ordered set of bearings (start_angle), so the i-th
    episode of one run is the same scenario as the i-th episode of any other. We use
    that positional index as the scenario id: it works whether or not the frame still
    carries an explicit `start_angle` column (older cached CSVs do not), which is what
    lets matched-scenario plots reuse an existing cache.

    `group_cols` identifies one run (e.g. ["config", "seed"], or ["seed"] for a pooled
    baseline). When `start_angle` is present it is used to order within a run for safety;
    otherwise the existing row order (ascending start_angle from compute_episode_metrics)
    is used. Returns False (and warns, leaving no column) if runs have unequal episode
    counts, since positional alignment would then be meaningless — callers fall back to
    unconditioned (all-successful) plotting.
    """
    sizes = df.groupby(group_cols).size()
    if sizes.nunique() != 1:
        warnings.warn(
            f"Runs have unequal episode counts {sorted(sizes.unique())}; cannot align "
            "scenarios positionally. Falling back to all-successful (no matched set).")
        return False
    sort_cols = list(group_cols) + (["start_angle"] if "start_angle" in df.columns else [])
    ordered = df.sort_values(sort_cols, kind="stable")
    df["scenario_id"] = ordered.groupby(group_cols).cumcount().reindex(df.index)
    return True


def per_episode_reasons(run_dir: Path, scenario: str) -> pd.Series | None:
    """Per-episode termination_reason for the run's `scenario` trajectory CSV."""
    csv_path = find_csv(run_dir, scenario)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    if "termination_reason" not in df.columns:
        # Older trajectory CSVs predate termination logging; treat every episode as a
        # success, matching compute_episode_metrics' default for the metric path so such
        # runs still appear (instead of being dropped) in the outcome breakdown.
        return pd.Series(SUCCESS_REASON, index=df.groupby("start_angle").size().index,
                         name="termination_reason")
    return df.groupby("start_angle")["termination_reason"].last()


def compute_success_rate(run_dir: Path, scenario: str) -> float | None:
    """Mean success rate for the run's `scenario` trajectory CSV."""
    reasons = per_episode_reasons(run_dir, scenario)
    if reasons is None:
        return None
    return (reasons == SUCCESS_REASON).mean()


def compute_termination_breakdown(run_dir: Path, scenario: str) -> pd.Series | None:
    """Fraction of episodes per termination reason for the run's `scenario` CSV."""
    reasons = per_episode_reasons(run_dir, scenario)
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

def find_run_dirs(run_pattern: None | list[str], runs_root: Path) -> Generator:
    """Yield sorted run directories, optionally filtered by name patterns."""
    for run_dir in sorted(runs_root.iterdir()):
        if not run_pattern:
            pass
        elif not any(pattern in run_dir.name for pattern in run_pattern):
            continue
        yield run_dir


def compute_episode_length(run_dir: Path, scenario: str) -> pd.Series | None:
    """Episode lengths (seconds) for the run's `scenario` trajectory CSV."""
    csv_path = find_csv(run_dir, scenario)
    if csv_path is None:
        return None
    return pd.read_csv(csv_path).groupby("start_angle")["sim_dt"].sum()

def compute_baseline(baseline_dirs: list[Path], scenario: str) -> tuple[float | None, float | None]:
    """Return (success_rate, mean_episode_length_s) pooled across all baseline run directories."""
    all_reasons = []
    all_lengths = []
    for d in baseline_dirs:
        reasons = per_episode_reasons(d, scenario)
        if reasons is not None:
            all_reasons.append(reasons)
        lengths = compute_episode_length(d, scenario)
        if lengths is not None:
            all_lengths.append(lengths)
    if not all_reasons:
        return None, None
    pooled_reasons = pd.concat(all_reasons)
    pooled_lengths = pd.concat(all_lengths) if all_lengths else None
    return (pooled_reasons == SUCCESS_REASON).mean(), (pooled_lengths.mean() if pooled_lengths is not None else None)


def collect_baseline_breakdown(baseline_dirs: list[Path], scenario: str) -> pd.Series | None:
    """Pooled termination-reason fractions across all baseline run directories."""
    all_reasons = []
    for d in baseline_dirs:
        reasons = per_episode_reasons(d, scenario)
        if reasons is not None:
            all_reasons.append(reasons)
    if not all_reasons:
        return None
    return pd.concat(all_reasons).value_counts(normalize=True)


def collect_baseline_seed_rates(baseline_dirs: list[Path], scenario: str) -> dict[int, float]:
    """Per-seed success rate for each baseline run directory."""
    result = {}
    for d in baseline_dirs:
        seed_match = re.search(r"seed(\d+)", d.name)
        seed = int(seed_match.group(1)) if seed_match else 0
        rate = compute_success_rate(d, scenario)
        if rate is not None:
            result[seed] = rate
    return result


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
    scenario: str,
    calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame],
    mean_episode_length: float,
) -> pd.DataFrame:
    """Per-episode metrics for every run whose name matches `pattern`.

    Each named group of `pattern` is added as a column (all-digit groups become ints).
    """
    frames = []
    pbar = tqdm(sorted(runs_root.iterdir()))
    for run_dir in pbar:
        pbar.set_description(f"Processing {run_dir.name}")
        match = pattern.search(run_dir.name)
        if not match:
            continue
        csv = find_csv(run_dir, scenario)
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
    scenario: str,
    calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame],
    mean_episode_length: float,
) -> pd.DataFrame:
    """Per-episode metrics pooled across the reference baseline run(s)."""
    frames = []
    pbar = tqdm(baseline_runs)
    for baseline_run in pbar:
        pbar.set_description(f"Processing {baseline_run.name}")
        seed_match = re.search(r"seed(\d+)", baseline_run.name)
        csv = find_csv(baseline_run, scenario)
        if not csv:
            print(f"  Skipping {baseline_run.name}: no CSV found")
            continue
        df = calculate_metrics(pd.read_csv(csv))
        metrics = compute_episode_metrics(df, mean_episode_length)
        metrics["seed"] = int(seed_match.group(1)) if seed_match else 0
        frames.append(metrics)
    return pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()


def collect_breakdown_data(runs_root: Path, pattern: re.Pattern, scenario: str) -> pd.DataFrame:
    """Success rate and termination breakdown per matching run.

    Named groups of `pattern` become columns, same as collect_run_metrics. Does not
    need the metric fn (reads only termination_reason), so it is cheap to call
    straight from a plotting script that wants the outcome panel.
    """
    records = []
    for run_dir in sorted(runs_root.iterdir()):
        match = pattern.search(run_dir.name)
        if not match:
            continue
        rate = compute_success_rate(run_dir, scenario)
        if rate is None:
            print(f"  Skipping {run_dir.name}: no usable trajectory data")
            continue
        record = {key: _coerce(value) for key, value in match.groupdict().items()}
        record["success_rate"] = rate
        record["breakdown"] = compute_termination_breakdown(run_dir, scenario)
        records.append(record)
    return pd.DataFrame(records)

def run_sweep_args_parser() -> tuple[Namespace, set]:
    """Shared CLI for a sweep that has a metrics view and/or an outcome-breakdown view.

    `--plots {both,metrics,breakdown}` (default both) selects which to draw. Only the
    metrics path needs BlueSky + the noise/fuel metric fn, so a breakdown-only run is
    dependency-free and fast. Each per-sweep script supplies the cosmetics through its
    own `plot_metrics(...)` / `plot_breakdown(...)` functions; their exact signatures
    (e.g. how the baseline is threaded through) differ per script, so check the sweep
    script itself rather than this docstring.

    A view the script doesn't provide is silently skipped. `--baseline` is shared: the
    full list is pooled into the metric reference boxes AND into the breakdown
    success-rate reference line(s).

    `--scenario` is the exact evaluation-scenario subdir to read, i.e. the
    {runway}_{label}_{model} folder generate_trajectories writes under each run's
    trajectories/. It is the sole selection key, so multiple scenarios for the same
    runway (e.g. a density-scaled map) stay separate, and it is woven into the output
    filenames so they never overwrite each other.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Plot sweep metrics and/or outcome breakdown")
    parser.add_argument("runs_root", type=str, help="path to runs root (e.g. runs/transforms)")
    parser.add_argument("--plots", choices=["both", "metrics", "breakdown"], default="both",
                        help="which views to draw (default: both)")
    parser.add_argument("--baseline", nargs="+", type=Path, default=None,
                        help="baseline run directory/-ies (pooled for both metric boxes "
                             "and breakdown success-rate / episode-length reference lines)")
    parser.add_argument("--scenario", type=str, default="EHAM_RW27_map_best",
                        help="evaluation-scenario trajectory subdir to read, exactly as "
                             "named by generate_trajectories: {runway}_{label}_{model} "
                             "(e.g. EHAM_RW27_map_best, EHAM_RW18R_scaling_best)")
    parser.add_argument("--map-path", type=str, default="./scripts/population_maps/europe_3035_1km.tif")
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--noise_clip_percentile", type=float, default=99.9)
    parser.add_argument("--mean_episode_length", type=float, default=1400.0)
    parser.add_argument("--output_dir", type=Path, default=Path("plots/sweep_overview_plots"))
    args = parser.parse_args()

    selected = {"metrics", "breakdown"} if args.plots == "both" else {args.plots}
    return args, selected




