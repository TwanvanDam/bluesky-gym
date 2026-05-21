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


def _differing_parts(names: list[str]) -> tuple[list[str], str]:
    """Return (labels, plot_name) using the part of each name that differs from the rest."""
    if len(names) == 1:
        return [names[0]], names[0]

    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break

    suffix = names[0]
    for name in names[1:]:
        while not name.endswith(suffix):
            suffix = suffix[1:]
        if not suffix:
            break

    suffix_len = len(suffix)
    labels = [name[len(prefix):-suffix_len if suffix_len else None] for name in names]
    plot_name = prefix.rstrip("_-")
    return labels, plot_name


def main(runs: list[RunPaths], labels: list[str], smoothing: int, plot_name: str, legend_title: str = "runs") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    cmap = plt.colormaps["Dark2"]

    for i, (run, label) in enumerate(zip(runs, labels)):
        df = load_scalar(run, TAG)
        if df is None:
            continue
        color = cmap(i)
        ax.plot(df["step"], df["value"], color=color, alpha=0.3, linewidth=1)
        df["smoothed"] = df["value"].rolling(smoothing).mean()
        ax.plot(df["step"], df["smoothed"], color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title(f"{plot_name} — training reward ({len(runs)} {legend_title})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    group_slug = runs[0].env_name.replace("_", "-").lower()
    out_dir = Path("plots/reward-plots") / group_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{plot_name}.png"
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
    run_names = [run.run_name for run in runs]

    if all("seed" in name for name in run_names):
        bases = [name.split("seed")[0].rstrip("_") for name in run_names]
        if len(set(bases)) == 1:
            seeds = [int(name.split("seed")[-1]) for name in run_names]
            labels = [f"seed {s}" for s in seeds]
            plot_name = bases[0]
            legend_title = "seeds"
        else:
            labels, plot_name = _differing_parts(run_names)
            legend_title = "runs"
    else:
        labels, plot_name = _differing_parts(run_names)
        legend_title = "runs"

    main(runs, labels, smoothing, plot_name, legend_title)