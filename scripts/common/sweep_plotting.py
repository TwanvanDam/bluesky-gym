from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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




