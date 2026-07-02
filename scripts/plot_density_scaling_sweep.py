"""Density-scaling ("implicit fuel-weight") sweep on an in-distribution airport.

For each config and each density-scale factor alpha, the policy flies with its OBSERVED
population map multiplied by alpha (generate_trajectories.py --scale_density), while fuel
and noise are measured against the TRUE (unscaled) density. Sweeping alpha traces each
config's fuel-noise frontier. Configs that cannot read density magnitude appear as a single
point: the no-map config (no map input) and the legacy Groot benchmark (scaling unsupported,
alpha=1 anchor only).

Each run directory name must match PATTERN (a config name plus an optional `_seedNN`). Each
alpha is read from the trajectory subdir `{runway}_scale_{alpha}` written by
generate_density_scaling.sh / generate_trajectories.py.

Usage:
    python -m scripts.plot_density_scaling_sweep runs/generalization \
        --runway EDDF_RW25R \
        --alphas 0.1 0.25 0.5 1 2 4 10

Filtering (max-steps fairness): a starting bearing that does NOT complete (success or
failed-approach) in *every* config/seed/alpha is dropped from all of them, so every frontier
point is evaluated on an identical bearing set and is not deflated by truncated, loitering
episodes. Disable with --no-match.
"""

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.common.colors import *
from scripts.common.sweep_plotting import compute_episode_metrics, find_csv

# Extracts config + optional seed; handles both "name_seed00" and bare "name" forms.
PATTERN = re.compile(r"^(?P<config>.+?)(?:_seed(?P<seed>\d+))?$")

# Episodes that completed and therefore carry valid fuel/noise (matches the generalization
# analysis): success and failed-approach are kept; max_steps / out_of_bounds are dropped.
KEEP_REASONS = {"success", "failed_approach"}

ANCHOR_ALPHA = 1.0  # the trained operating point

# CLI defaults (kept as module constants so the sweep's canonical settings live in one place).
DEFAULT_RUNWAY = "EDDF_RW25R"
DEFAULT_ALPHAS = ["0.1", "0.25", "0.5", "1", "2", "4", "10"]
DEFAULT_MAP_PATH = "./scripts/population_maps/europe_3035_1km.tif"
DEFAULT_MEAN_EPISODE_LENGTH = 1400.0
DEFAULT_NOISE_CLIP_PERCENTILE = 99.9
DEFAULT_OUTPUT_DIR = Path("plots/sweep_overview_plots")

# Alphas shown in the marker-size legend and used as failure-rate x-ticks.
LEGEND_ALPHAS = (0.25, 0.5, 1, 2, 4)

WIDTH = 0.85 * TEXTWIDTH_IN
AXES_ASPECT = 0.78  # figure height / width, shared by both plots
CONFIG_COLOR_RULES = {
    "no_map" : BASELINE_COLOR,
    "centered": CENTERED_COLOR,
    "forward": FORWARD_COLOR,
    "multi_scale": MULTI_SCALE_COLOR,
    "transformed_baseline": BASELINE_COLOR,
    "transformed": TRANSFORMS_COLOR,
}

def config_color(config: str) -> str:
    for needle, color in CONFIG_COLOR_RULES.items():
        if needle in config:
            return color
    warnings.warn(f"No color rule matches config {config!r}; using UNKNOWN_COLOR "
                  "(add a CONFIG_COLOR_RULES entry to give it its own color).")
    return UNKNOWN_COLOR

# ----------------------------------------------------------------------------- collection

def _seed_of(name: str):
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def collect_scaling_metrics(
    runs_root: Path, runway: str, alphas: list[str],
    calculate_metrics, mean_episode_length: float,
) -> pd.DataFrame:
    """Long per-bearing dataframe: one row per (config, seed, alpha, start_angle).

    Keeps start_angle and termination_reason (which collect_run_metrics discards) so the
    bearing-level matched filter below can run.
    """
    frames = []
    run_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())
    for run_dir in tqdm(run_dirs, desc="Collecting scaling metrics", unit="run"):
        match = PATTERN.search(run_dir.name)
        if not match:
            continue
        config, seed = match.group("config"), _seed_of(run_dir.name)
        for alpha in alphas:
            csv = find_csv(run_dir, f"{runway}_scale_{alpha}")
            if csv is None:
                continue
            raw = pd.read_csv(csv)
            metrics = compute_episode_metrics(calculate_metrics(raw), mean_episode_length)
            metrics["termination_reason"] = raw.groupby("start_angle")["termination_reason"].last()
            metrics = metrics.reset_index()  # start_angle becomes a column
            metrics["config"] = config
            metrics["seed"] = seed
            metrics["alpha"] = float(alpha)
            frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def matched_filter(df: pd.DataFrame, keep_reasons: set) -> pd.DataFrame:
    """Drop any start_angle that fails to complete in even one config/seed/alpha."""
    keep = df["termination_reason"].isin(keep_reasons)
    bearing_ok = keep.groupby(df["start_angle"]).transform("all")
    return df[bearing_ok]


def frontier_points(df: pd.DataFrame) -> pd.DataFrame:
    """Median normalized fuel/noise per (config, alpha), pooled over bearings and seeds."""
    g = (
        df.groupby(["config", "alpha"])
        .agg(
            fuel=("normalized_fuel", "median"),
            noise=("normalized_noise", "median"),
            fuel_q1=("normalized_fuel", lambda s: s.quantile(0.25)),
            fuel_q3=("normalized_fuel", lambda s: s.quantile(0.75)),
            noise_q1=("normalized_noise", lambda s: s.quantile(0.25)),
            noise_q3=("normalized_noise", lambda s: s.quantile(0.75)),
            n_bearings=("start_angle", "nunique"),
        )
        .reset_index()
        .sort_values(["config", "alpha"])
    )
    return g


def failure_rate_points(df: pd.DataFrame, keep_reasons: set) -> pd.DataFrame:
    """Fraction of episodes that did NOT complete per (config, alpha).

    "Not successful" means the episode's termination_reason is outside keep_reasons
    (i.e. max_steps / out_of_bounds), pooled over bearings and seeds. Computed on the
    unmatched dataframe, since the matched filter would otherwise drop these episodes.
    """
    df = df.copy()
    df["failed"] = ~df["termination_reason"].isin(keep_reasons)
    g = (
        df.groupby(["config", "alpha"])
        .agg(failure_rate=("failed", "mean"), n_episodes=("failed", "size"))
        .reset_index()
        .sort_values(["config", "alpha"])
    )
    g["failure_pct"] = 100.0 * g["failure_rate"]
    return g


# -------------------------------------------------------------------------------- plotting

# Marker area (points^2) as a function of alpha. Grows on a log2 scale so the
# multiplicative density factor reads linearly, and stays strictly positive for
# alpha < 1 (log2 would go negative). alpha=0.25 -> ~20, alpha=1 -> ~100, alpha=4 -> ~180.
SIZE_INTERCEPT = 18.0
SIZE_SLOPE = 26.0
SIZE_LOG_OFFSET = 2.0  # shifts log2(min expected alpha=0.25) to 0
MARKER_EDGE_WIDTH = 0.8  # black outline thickness, shared by every marker


def alpha_to_size(alpha) -> np.ndarray:
    return SIZE_INTERCEPT + SIZE_SLOPE * (np.log2(alpha) + SIZE_LOG_OFFSET)


def plot_frontier(pts: pd.DataFrame, runway: str, runs_name: str, output_dir: Path) -> Path:
    configs = sorted(pts["config"].unique())

    fig, ax = plt.subplots(figsize=(WIDTH, AXES_ASPECT * WIDTH))

    # Solid line is the default; a config whose color has already been used gets a
    # dashed line so overlapping-color series stay distinguishable.
    color_use_count: dict = {}
    config_handles = []

    for c in configs:
        sub = pts[pts["config"] == c].sort_values("alpha")
        color = config_color(c)

        # A real frontier: line + square markers sized by alpha.
        seen = color_use_count.get(color, 0)
        color_use_count[color] = seen + 1
        linestyle = "-" if seen == 0 else "--"

        ax.plot(sub["fuel"], sub["noise"], linestyle=linestyle, color=color, zorder=3)
        for fuel, noise, alpha in zip(sub["fuel"], sub["noise"], sub["alpha"]):
            if alpha == 1:
                 facecolor = "white"
                 outline = color
            else:
                facecolor = color
                outline = "k"
            ax.scatter(fuel, noise, marker="o", s=alpha_to_size(alpha),
                       facecolors=facecolor, edgecolors=outline, linewidths=MARKER_EDGE_WIDTH, zorder=4)

        config_handles.append(
            plt.Line2D([0], [0], color=color, linestyle=linestyle, marker="o", markeredgecolor="k",
                       markeredgewidth=MARKER_EDGE_WIDTH, markersize=7, label=c))

    ax.set_xlabel("normalized fuel (median over bearings)")
    ax.set_ylabel("normalized noise (median over bearings)")
    ax.grid(True, alpha=0.3)

    # Both legends sit outside the axes on the right, stacked, and share the framed
    # black-edged style used across the other sweep plots.
    legend_main = ax.legend(handles=config_handles, frameon=True, edgecolor="k",
                            loc="upper left", bbox_to_anchor=(1, 1))
    ax.add_artist(legend_main)

    # Secondary legend: marker size <-> alpha.
    size_alphas = [a for a in LEGEND_ALPHAS if a in set(pts["alpha"])]
    if size_alphas:
        size_handles = [
            plt.Line2D([0], [0], linestyle="none", marker="o", markeredgecolor="k" if a != 1 else "0.5",
                       markeredgewidth=MARKER_EDGE_WIDTH,
                       markersize=np.sqrt(alpha_to_size(a)), markerfacecolor="0.5" if a != 1 else "white",
                       label=r"$\alpha = $" + f"{a:g}")
            for a in size_alphas
        ]
        ax.legend(handles=size_handles,
                  frameon=True, edgecolor="k", loc="lower left", bbox_to_anchor=(1, 0),
                  ncol=1, handletextpad=0.2)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"frontier_{runs_name}_{runway}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_failure_rate(rates: pd.DataFrame, runway: str, runs_name: str,
                      output_dir: Path) -> Path:
    """Percentage of non-completing episodes vs. density-scale alpha, per config."""
    configs = sorted(rates["config"].unique())

    fig, ax = plt.subplots(figsize=(WIDTH, AXES_ASPECT * WIDTH))
    for c in configs:
        sub = rates[rates["config"] == c].sort_values("alpha")
        col = config_color(c)
        if len(sub) >= 2:
            ax.plot(sub["alpha"], sub["failure_pct"], "-o", color=col, label=c,
                    markersize=5)
        else:  # single point: no-map / legacy benchmark
            ax.scatter(sub["alpha"], sub["failure_pct"], marker="*", s=240,
                       facecolors=col, edgecolors="black", linewidths=1.0,
                       label=f"{c} (fixed)")

    ax.set_xscale("log")
    ax.set_xticks(list(LEGEND_ALPHAS), labels=[str(a) for a in LEGEND_ALPHAS])
    ax.set_xlabel(r"density-scale factor $\alpha$")
    ax.set_ylabel("non-completing episodes (%)")
    ax.set_title(f"Unsuccessful-run rate under density scaling — {runway}")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, edgecolor="k", loc="upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"failure_rate_{runs_name}_{runway}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------------------ main

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=Path, help="folder of run dirs (e.g. runs/generalization)")
    parser.add_argument("--runway", default=DEFAULT_RUNWAY,
                        help=f"runway label as it appears in the scenario subdir (default: {DEFAULT_RUNWAY})")
    parser.add_argument("--alphas", nargs="+", default=DEFAULT_ALPHAS,
                        help="density-scale factors; must match the labels used at generation time")
    parser.add_argument("--map-path", type=str, default=DEFAULT_MAP_PATH)
    parser.add_argument("--mean_episode_length", type=float, default=DEFAULT_MEAN_EPISODE_LENGTH)
    parser.add_argument("--noise_clip_percentile", type=float, default=DEFAULT_NOISE_CLIP_PERCENTILE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--use-cache", action="store_true",
                        help="load frontier points from the saved CSV instead of recomputing")
    return parser.parse_args()


def _expected_configs(runs_root: Path) -> set[str]:
    """Config names implied by the run dirs under `runs_root` (seed suffix stripped)."""
    configs = set()
    for run_dir in runs_root.iterdir():
        match = PATTERN.search(run_dir.name) if run_dir.is_dir() else None
        if match:
            configs.add(match.group("config"))
    return configs


def _check_cache_complete(df: pd.DataFrame, runs_root: Path, requested: list[float]) -> None:
    """Fail loudly if the cache is missing any run-dir config or requested alpha."""
    missing_configs = _expected_configs(runs_root) - set(df["config"].unique())
    if missing_configs:
        raise KeyError(
            f"Cache is missing config(s) {sorted(missing_configs)} present under {runs_root}. "
            "Regenerate without --use-cache to include them.")
    missing_alphas = set(requested) - set(df["alpha"].unique())
    if missing_alphas:
        raise KeyError(
            f"Cache is missing alpha(s) {sorted(missing_alphas)}. "
            "Regenerate without --use-cache to include them.")


def _load_cached(csv_path: Path, failure_csv_path: Path, alphas: list[str],
                 runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload previously-saved frontier points and failure rates, filtered to `alphas`.

    Errors if the cache lacks any config found under `runs_root` or any requested alpha.
    """
    requested = [float(a) for a in alphas]
    for path in (csv_path, failure_csv_path):
        if not path.exists():
            raise FileNotFoundError(f"Cache not found: {path}")

    pts = pd.read_csv(csv_path)
    _check_cache_complete(pts, runs_root, requested)
    pts = pts[pts["alpha"].isin(requested)]
    print(f"Loaded frontier points from cache → {csv_path}")
    print(pts.to_string(index=False))

    rates = pd.read_csv(failure_csv_path)
    _check_cache_complete(rates, runs_root, requested)
    rates = rates[rates["alpha"].isin(requested)]
    print(f"Loaded failure rates from cache → {failure_csv_path}")
    return pts, rates


def _compute_frontier(args: argparse.Namespace, csv_path: Path,
                      failure_csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect per-bearing metrics from trajectory CSVs, then derive frontier + failure rates."""
    import bluesky as bs
    from bluesky_gym.maps.map_sources import TransformedTiffMapSourceConfig
    from bluesky_gym.metrics.evaluation_metrics import (
        bounds_from_df, build_metric_fn, make_pop_samplers,
    )

    bs.init()

    # Build the metric fn on the TRUE (unscaled) density, exactly like plot_generalization_sweep:
    # bounds are taken from every scenario CSV so the sampler covers all flown trajectories.
    all_csvs = [
        find_csv(run_dir, f"{args.runway}_scale_{a}")
        for run_dir in args.runs_root.iterdir() if run_dir.is_dir()
        for a in args.alphas
    ]
    all_csvs = [p for p in all_csvs if p is not None]
    if not all_csvs:
        raise FileNotFoundError(
            f"No trajectory CSVs under {args.runs_root} for runway '{args.runway}' "
            f"and alphas {args.alphas}. Run generate_density_scaling.sh first.")
    combined = pd.concat([pd.read_csv(p) for p in all_csvs], ignore_index=True)
    samplers = make_pop_samplers(
        TransformedTiffMapSourceConfig(file_path=args.map_path), bounds=bounds_from_df(combined),
        clip_percentile=args.noise_clip_percentile, train_resampling="average",
        true_resampling="average")
    calculate_metrics = build_metric_fn(samplers)

    df = collect_scaling_metrics(args.runs_root, args.runway, args.alphas,
                                 calculate_metrics, args.mean_episode_length)
    if df.empty:
        raise SystemExit("No per-bearing metrics collected.")

    # Failure rate is computed on the unmatched df, since matched_filter drops the
    # very (non-completing) episodes the failure-rate plot is meant to show.
    rates = failure_rate_points(df, KEEP_REASONS)
    rates.to_csv(failure_csv_path, index=False)
    print(f"Saved failure rates → {failure_csv_path}")

    pts = frontier_points(df)
    pts.to_csv(csv_path, index=False)
    print(f"Saved frontier points → {csv_path}")
    print(pts.to_string(index=False))
    return pts, rates


def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir / args.runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Frontier/failure caches live alongside the runs they summarize (runs/<sweep>/), not
    # with the plot images, so the recomputed data stays next to its source trajectories.
    csv_path = args.runs_root / f"frontier_{args.runs_root.name}_{args.runway}.csv"
    failure_csv_path = args.runs_root / f"failure_rate_{args.runs_root.name}_{args.runway}.csv"

    if args.use_cache:
        pts, rates = _load_cached(csv_path, failure_csv_path, args.alphas, args.runs_root)
    else:
        pts, rates = _compute_frontier(args, csv_path, failure_csv_path)

    out_path = plot_frontier(pts, args.runway, args.runs_root.name, output_dir)
    print(f"Saved plot → {out_path}")
    failure_path = plot_failure_rate(rates, args.runway, args.runs_root.name, output_dir)
    print(f"Saved plot → {failure_path}")


if __name__ == "__main__":
    main()
