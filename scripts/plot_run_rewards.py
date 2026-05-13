"""
Plot rollout/ep_rew_mean for runs.
Reads tensorboard event files directly; no pre-generated CSVs needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scripts.common.run_paths import resolve_run, RunPaths


TAG = "rollout/ep_rew_mean"


def load_scalar(run: RunPaths, tag: str) -> pd.DataFrame | None:
    tb_dir = run.tensorboard_dir
    event_dirs = sorted(tb_dir.iterdir()) if tb_dir.exists() else []
    if not event_dirs:
        print(f"  No tensorboard data for {run.run_name}")
        return None

    ea = EventAccumulator(str(event_dirs[0]))
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        print(f"  Tag '{tag}' not found for {run.run_name}")
        return None

    events = ea.Scalars(tag)
    return pd.DataFrame({"step": [e.step for e in events], "value": [e.value for e in events]})


def main(runs: list[RunPaths], seeds: list[int], smoothing: int, plot_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    cmap = plt.colormaps["Dark2"]

    for i, (run, seed) in enumerate(zip(runs, seeds)):
        df = load_scalar(run, TAG)
        if df is None:
            continue
        color = cmap(i)
        ax.plot(df["step"], df["value"], color=color, alpha=0.3, linewidth=1)
        df["smoothed"] = df["value"].rolling(smoothing).mean()
        ax.plot(df["step"], df["smoothed"], color=color, linewidth=1.5, label=f"seed {seed}")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title(f"{plot_name} — training reward ({len(runs)} seeds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    out_path = f"plots/{plot_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot mean reward for trained run(s).")
    parser.add_argument("run_refs", nargs="+", help="Run reference(s) or path to a trajectories.csv")
    parser.add_argument("--smoothing", type=int, default=100)
    args = parser.parse_args()

    runs = [resolve_run(r) for r in args.run_refs]
    smoothing = args.smoothing

    if not all("seed" in run.run_name for run in runs):
        parser.error("All runs must contain 'seed' in their name.")

    bases = [run.run_name.split("seed")[0].rstrip("_") for run in runs]
    if len(set(bases)) > 1:
        parser.error(f"Runs differ in more than seed: {set(bases)}")

    seeds = [int(run.run_name.split("seed")[-1]) for run in runs]
    plot_name = bases[0]

    main(runs, seeds, smoothing, plot_name)