"""
Plot rollout/ep_rew_mean for runs.
Reads tensorboard event files directly; no pre-generated CSVs needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

from scripts.common.colors import *
from scripts.common.run_paths import RunPaths


TAG = "rollout/ep_rew_mean"
TAG_NOISE = "episode/total_episode_noise_reward"
TAG_FUEL = "episode/total_episode_fuel_reward"

LEGEND_FILE = "name_to_legend.txt"

figure_size = (0.75 * TEXTWIDTH_IN, 0.4 * TEXTWIDTH_IN)

RunData = dict[str, pd.DataFrame | None]


def load_legend_overrides(runs: list[RunPaths]) -> dict[str, str]:
    """Read `name_to_legend.txt` files sitting alongside the run dirs.

    Returns a {run_name: legend_entry} mapping. Each run's parent directory is
    checked for the file; format is one `filename, legend_entry` pair per line
    with an optional `filename, legend_entry` header row.
    """
    mapping: dict[str, str] = {}
    for parent in {run.root.parent for run in runs}:
        legend_path = parent / LEGEND_FILE
        if not legend_path.exists():
            continue
        for line in legend_path.read_text().splitlines():
            line = line.strip()
            if not line or "," not in line:
                continue
            name, legend = (part.strip() for part in line.split(",", 1))
            if name == "filename" and legend == "legend_entry":
                continue  # header row
            mapping[name] = legend
    return mapping


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
    fig, ax = plt.subplots(figsize=figure_size)
    _max = []
    for i, (data, label) in enumerate(zip(run_data, labels)):
        df = data.get(tag)
        if df is None:
            continue
        color = SEED_COLORS[i]
        ax.plot(df["step"], df["value"], color=color, alpha=0.1, linewidth=1)
        df["smoothed"] = df["value"].rolling(smoothing).mean()
        ax.plot(df["step"], df["smoothed"], color=color, linewidth=1.5, label=label)
        _max += [df["step"].max()]
    ax.set_xlim([0, round(max(_max) / 100_000) * 100_000])
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    if limits:
        ax.set_ylim(*limits)
    legend = ax.legend(frameon=True, edgecolor="k", loc="center left",bbox_to_anchor=(1,0.5))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_dir = output_dir / group_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{plot_name}{file_suffix}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", bbox_extra_artists=[legend])
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
    fig, ax = plt.subplots(figsize=figure_size)

    for i, (data, label) in enumerate(zip(run_data, labels)):
        dfs = [data.get(t) for t in tags]
        if any(d is None for d in dfs):
            continue
        merged = dfs[0].rename(columns={"value": "v0"})
        for j, d in enumerate(dfs[1:], 1):
            merged = merged.merge(d.rename(columns={"value": f"v{j}"}), on="step", how="inner")
        merged["value"] = sum(merged[f"v{j}"] for j in range(len(tags)))
        color = SEED_COLORS[i]
        ax.plot(merged["step"], merged["value"], color=color, alpha=0.1, linewidth=1)
        merged["smoothed"] = merged["value"].rolling(smoothing).mean()
        ax.plot(merged["step"], merged["smoothed"], color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    if limits:
        ax.set_ylim(*limits)
    legend = ax.legend(frameon=True, edgecolor="k", loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_dir = output_dir / group_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{plot_name}{file_suffix}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", bbox_extra_artists=[legend])
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
    parser = argparse.ArgumentParser(description="Plot mean reward for the run(s) in a folder.")
    parser.add_argument("folder", type=Path, help="Folder containing run subdirectories (e.g. runs/appendix/centered_16)")
    parser.add_argument("--smoothing", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=Path("plots/reward-plots"), help=f"Output directory for the plots")
    args = parser.parse_args()

    folder = args.folder
    if not folder.is_dir():
        parser.error(f"Not a directory: {folder}")
    runs = [
        RunPaths.from_run_dir(d)
        for d in sorted(folder.iterdir(), key=lambda p: p.name)
        if d.is_dir() and d.name != "slurm"
    ]
    if not runs:
        parser.error(f"No run subdirectories found in {folder}")
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

    overrides = load_legend_overrides(runs)
    if overrides:
        labels = [overrides.get(run.run_name, label) for run, label in zip(runs, labels)]

    main(runs, labels, smoothing, plot_name, output_dir, legend_title)