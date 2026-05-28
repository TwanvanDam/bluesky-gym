"""
Plot success rate vs. observation resolution for the forward/centered sweep.

Expects runs under runs/PopulationWrapper-v0/ named:
  {forward|centered}_{resolution}_seed{NN}

where {resolution} is km/pixel (1, 2, 4, 8, 16, 32).

Success rate is computed from trajectory CSVs (requires generate_trajectories.py
to have been run first with the termination_reason column present).
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RUNS_ROOT = Path(__file__).parent.parent / "runs" / "resolution_sweep_2"
DEFAULT_BASELINE_NAME = "sweep_2_no_map_seed00"
DEFAULT_RUNWAY = "EDDF_RW25R"
RUN_PATTERN = re.compile(r"^(forward|centered)_(\d+)_seed(\d+)$")
SUCCESS_REASON = "success"

FORWARD_COLOR = "#2196F3"
CENTERED_COLOR = "#FF6B35"
DOT_ALPHA = 0.8
DOT_SIZE = 60
BAR_ALPHA = 0.6
BAR_WIDTH = 0.5
SEED_COLORS = ["#4CAF50", "#9C27B0", "#FF9800", "#E91E63", "#00BCD4", "#795548"]
BASELINE_COLOR = "#555555"


def compute_success_rate(run_dir: Path, runway: str) -> float | None:
    """Mean success rate across all *_map trajectory CSVs in a run."""
    traj_root = run_dir / "trajectories"
    if not traj_root.exists():
        return None

    map_csvs = list(traj_root.glob(f"*{runway}_map*/trajectories.csv"))
    if not map_csvs:
        return None

    episodes = []
    for csv_path in map_csvs:
        df = pd.read_csv(csv_path)
        if "termination_reason" not in df.columns:
            print(f"  Warning: no termination_reason column in {csv_path}, skipping")
            continue
        # One row per timestep; termination_reason is constant within an episode.
        per_episode = df.groupby("start_angle")["termination_reason"].last()
        episodes.append(per_episode)

    if not episodes:
        return None

    all_episodes = pd.concat(episodes)
    return (all_episodes == SUCCESS_REASON).mean()


def compute_episode_length(run_dir: Path, runway: str) -> pd.Series | None:
    """Episode lengths (seconds) across all *_map trajectory CSVs in a run."""
    traj_root = run_dir / "trajectories"
    if not traj_root.exists():
        return None

    map_csvs = list(traj_root.glob(f"*{runway}_map*/trajectories.csv"))
    if not map_csvs:
        return None

    episodes = []
    for csv_path in map_csvs:
        df = pd.read_csv(csv_path)
        # One row per timestep; termination_reason is constant within an episode.
        per_episode = df.groupby("start_angle")["sim_dt"].sum()
        episodes.append(per_episode)

    if not episodes:
        return None

    return pd.concat(episodes)


def collect_data(runs_root: Path, runway: str) -> pd.DataFrame:
    records = []
    for run_dir in sorted(runs_root.iterdir()):
        m = RUN_PATTERN.match(run_dir.name)
        if not m:
            continue
        mode, resolution, seed = m.group(1), int(m.group(2)), int(m.group(3))
        rate = compute_success_rate(run_dir, runway)
        if rate is None:
            print(f"  Skipping {run_dir.name}: no usable trajectory data")
            continue
        records.append({
            "mode": mode,
            "resolution": resolution,
            "seed": seed,
            "success_rate": rate,
            "length": compute_episode_length(run_dir, runway),
        })
    return pd.DataFrame(records)


def compute_baseline(baseline_dir: Path, runway: str) -> tuple[float | None, float | None]:
    """Return (success_rate, mean_episode_length_s) for the baseline run."""
    lengths = compute_episode_length(baseline_dir, runway)
    return compute_success_rate(baseline_dir, runway), (lengths.mean() if lengths is not None else None)


def _seed_color_map(df: pd.DataFrame) -> dict:
    all_seeds = sorted(df["seed"].unique())
    return {seed: SEED_COLORS[i % len(SEED_COLORS)] for i, seed in enumerate(all_seeds)}


def _seed_legend(ax, color_map: dict) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, label=f"Seed {s}")
        for s, c in color_map.items()
    ]
    ax.legend(handles=handles, frameon=False)


def _draw_baseline(ax, value: float | None, label: str) -> None:
    if value is not None:
        ax.axhline(value, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline ({label})", zorder=3)
        ax.legend(frameon=False)


def plot_mode(ax, df: pd.DataFrame, mode: str, color: str, baseline: float | None = None) -> None:
    resolutions = sorted(df["resolution"].unique())
    x = np.arange(len(resolutions))
    colors = _seed_color_map(df)

    for i, res in enumerate(resolutions):
        res_df = df[df["resolution"] == res]
        seed_rates = {row["seed"]: row["success_rate"] for _, row in res_df.iterrows()}

        ax.bar(x[i], np.mean(list(seed_rates.values())), width=BAR_WIDTH, color=color, alpha=BAR_ALPHA)

        seeds = sorted(seed_rates)
        jitter = np.linspace(-0.08, 0.08, len(seeds))
        for xi, seed in zip(jitter, seeds):
            ax.scatter(x[i] + xi, seed_rates[seed],
                       color=colors[seed], s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} km/px" for r in resolutions])
    ax.set_xlabel("Observation resolution")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{mode.capitalize()} observation window")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _seed_legend(ax, colors)
    _draw_baseline(ax, baseline, f"{baseline:.0%}" if baseline is not None else "")


def plot_mode_length(ax, df: pd.DataFrame, mode: str, color: str, baseline: float | None = None) -> None:
    resolutions = sorted(df["resolution"].unique())
    x = np.arange(len(resolutions))
    colors = _seed_color_map(df)

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
    _seed_legend(ax, colors)
    _draw_baseline(ax, baseline, f"{baseline:.0f} s" if baseline is not None else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot resolution sweep results.")
    parser.add_argument(
        "runs_root", nargs="?", type=Path, default=DEFAULT_RUNS_ROOT,
        help=f"Directory containing sweep run folders (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help=f"Baseline run directory (default: <runs_root>/{DEFAULT_BASELINE_NAME})",
    )
    parser.add_argument(
        "--runway", default=DEFAULT_RUNWAY,
        help=f"Runway identifier used to filter trajectory CSVs (default: {DEFAULT_RUNWAY})",
    )
    args = parser.parse_args()

    runs_root: Path = args.runs_root
    baseline_dir: Path = args.baseline or runs_root / DEFAULT_BASELINE_NAME
    runway: str = args.runway

    df = collect_data(runs_root, runway)
    if df.empty:
        print("No data found. Run generate_trajectories.py on the sweep runs first.")
        return

    baseline_rate, baseline_length = compute_baseline(baseline_dir, runway)
    print(f"Baseline — success rate: {baseline_rate:.1%}, mean episode length: {baseline_length:.1f} s")

    for mode in ("forward", "centered"):
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            print(f"No data for mode '{mode}', skipping.")
            continue

        color = FORWARD_COLOR if mode == "forward" else CENTERED_COLOR

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_mode(ax, mode_df, mode, color, baseline=baseline_rate)
        fig.tight_layout()
        out_path = Path(__file__).parent / f"{runs_root.name}_{mode}_{runway}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_mode_length(ax, mode_df, mode, color, baseline=baseline_length)
        fig.tight_layout()
        out_path = Path(__file__).parent / f"{runs_root.name}_{mode}_length_{runway}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.show()


if __name__ == "__main__":
    main()
