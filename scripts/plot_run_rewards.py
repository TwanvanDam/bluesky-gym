"""
Plot rollout/ep_rew_mean for runs.
Reads tensorboard event files directly; no pre-generated CSVs needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

from scripts.common.colors import qual
from scripts.common.run_paths import resolve_run, RunPaths


TAG = "rollout/ep_rew_mean"
TAG_NOISE = "episode/total_episode_noise_reward"
TAG_FUEL = "episode/total_episode_fuel_reward"


RunData = dict[str, pd.DataFrame | None]


def load_run_scalars(run: RunPaths, tags: list[str]) -> RunData:
    tb_dir = run.tensorboard_dir
    event_dirs = sorted(tb_dir.iterdir()) if tb_dir.exists() else []
    if not event_dirs:
        print(f"  No tensorboard data for {run.run_name}")
        return {tag: None for tag in tags}

    ea = EventAccumulator(str(event_dirs[0]))
    ea.Reload()
    available = ea.Tags().get("scalars", [])

    result: RunData = {}
    for tag in tags:
        if tag not in available:
            result[tag] = None
        else:
            events = ea.Scalars(tag)
            result[tag] = pd.DataFrame({"step": [e.step for e in events], "value": [e.value for e in events]})
    return result


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


def _plot_and_save(
    run_data: list[RunData],
    labels: list[str],
    smoothing: int,
    plot_name: str,
    legend_title: str,
    group_slug: str,
    tag: str,
    ylabel: str,
    title_suffix: str,
    file_suffix: str,
    output_dir: Path,
    limits: list | None
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (data, label) in enumerate(zip(run_data, labels)):
        df = data.get(tag)
        if df is None:
            continue
        color = qual(i)
        ax.plot(df["step"], df["value"], color=color, alpha=0.1, linewidth=1)
        df["smoothed"] = df["value"].rolling(smoothing).mean()
        ax.plot(df["step"], df["smoothed"], color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{plot_name} — {title_suffix} ({len(run_data)} {legend_title})")
    if limits:
        ax.set_ylim(*limits)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    ax.grid()

    fig.tight_layout()
    out_dir = output_dir / group_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{plot_name}{file_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _plot_sum_and_save(
    run_data: list[RunData],
    labels: list[str],
    smoothing: int,
    plot_name: str,
    legend_title: str,
    group_slug: str,
    tags: list[str],
    ylabel: str,
    title_suffix: str,
    file_suffix: str,
    output_dir: Path,
    limits: list | None
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (data, label) in enumerate(zip(run_data, labels)):
        dfs = [data.get(t) for t in tags]
        if any(d is None for d in dfs):
            continue
        merged = dfs[0].rename(columns={"value": "v0"})
        for j, d in enumerate(dfs[1:], 1):
            merged = merged.merge(d.rename(columns={"value": f"v{j}"}), on="step", how="inner")
        merged["value"] = sum(merged[f"v{j}"] for j in range(len(tags)))
        color = qual(i)
        ax.plot(merged["step"], merged["value"], color=color, alpha=0.1, linewidth=1)
        merged["smoothed"] = merged["value"].rolling(smoothing).mean()
        ax.plot(merged["step"], merged["smoothed"], color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{plot_name} — {title_suffix} ({len(run_data)} {legend_title})")
    if limits:
        ax.set_ylim(*limits)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    ax.grid()

    fig.tight_layout()
    out_dir = output_dir / group_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{plot_name}{file_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main(runs: list[RunPaths], labels: list[str], smoothing: int, plot_name: str, output_dir: Path, legend_title: str = "runs") -> None:
    all_tags = [TAG, TAG_NOISE, TAG_FUEL]
    run_data = [
        load_run_scalars(run, all_tags)
        for run in tqdm(runs, desc="Loading tensorboard logs", unit="run")
    ]
    group_slug = runs[0].env_name.replace("_", "-").lower()

    plots = [
        dict(tag=TAG, ylabel="Mean episode reward", title_suffix="training reward", file_suffix="", limits=None),
        dict(tag=TAG_NOISE, ylabel="Mean noise reward", title_suffix="noise reward", file_suffix="_noise", limits=[-1, 0]),
        dict(tag=TAG_FUEL, ylabel="Mean fuel reward", title_suffix="fuel reward", file_suffix="_fuel", limits=[-1, 0]),
    ]
    for kwargs in tqdm(plots, desc="Rendering figures", unit="fig"):
        _plot_and_save(run_data, labels, smoothing, plot_name, legend_title, group_slug, output_dir=output_dir, **kwargs)
    tqdm.write("Rendering fuel+noise figure...")
    _plot_sum_and_save(run_data, labels, smoothing, plot_name, legend_title, group_slug,
                       tags=[TAG_FUEL, TAG_NOISE], ylabel="Mean fuel + noise reward",
                       title_suffix="fuel + noise reward", file_suffix="_fuel_noise", output_dir=output_dir, limits=[-2,0])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot mean reward for trained run(s).")
    parser.add_argument("run_refs", nargs="+", help="Run reference(s) or path to a trajectories.csv")
    parser.add_argument("--smoothing", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=Path("plots/reward-plots"), help=f"Output directory for the plots")
    args = parser.parse_args()

    runs = [resolve_run(r) for r in args.run_refs]
    smoothing = args.smoothing
    run_names = [run.run_name for run in runs]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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

    main(runs, labels, smoothing, plot_name, output_dir, legend_title)