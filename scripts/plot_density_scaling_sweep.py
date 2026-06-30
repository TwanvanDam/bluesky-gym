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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.common.colors import qual
from scripts.common.sweep_plotting import compute_episode_metrics, find_csv

# Extracts config + optional seed; handles both "name_seed00" and bare "name" forms.
PATTERN = re.compile(r"^(?P<config>.+?)(?:_seed(?P<seed>\d+))?$")

# Episodes that completed and therefore carry valid fuel/noise (matches the generalization
# analysis): success and failed-approach are kept; max_steps / out_of_bounds are dropped.
KEEP_REASONS = {"success", "failed_approach"}

ANCHOR_ALPHA = 1.0  # the trained operating point


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

def plot_frontier(pts: pd.DataFrame, runway: str, runs_name: str, output_dir: Path,
                  errorbars: bool) -> Path:
    configs = sorted(pts["config"].unique())
    color = {c: qual(i) for i, c in enumerate(configs)}

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c in configs:
        sub = pts[pts["config"] == c].sort_values("alpha")
        col = color[c]
        if len(sub) >= 2:  # a real frontier
            ax.plot(sub["fuel"], sub["noise"], "-o", color=col, label=c, zorder=3,
                    markersize=5)
            if errorbars:
                ax.errorbar(sub["fuel"], sub["noise"],
                            xerr=[sub["fuel"] - sub["fuel_q1"], sub["fuel_q3"] - sub["fuel"]],
                            yerr=[sub["noise"] - sub["noise_q1"], sub["noise_q3"] - sub["noise"]],
                            fmt="none", ecolor=col, alpha=0.3, zorder=2, capsize=2)
            for _, r in sub.iterrows():
                ax.annotate(f"{r['alpha']:g}", (r["fuel"], r["noise"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=6, color=col)
            anchor = sub[np.isclose(sub["alpha"], ANCHOR_ALPHA)]
            ax.scatter(anchor["fuel"], anchor["noise"], s=150, facecolors=col,
                       edgecolors="black", linewidths=1.3, zorder=4)
        else:  # single point: no-map / legacy benchmark
            ax.scatter(sub["fuel"], sub["noise"], marker="*", s=240, facecolors=col,
                       edgecolors="black", linewidths=1.0, label=f"{c} (fixed)", zorder=5)

    ax.scatter([], [], s=150, facecolors="white", edgecolors="black", linewidths=1.3,
               label=f"$\\alpha={ANCHOR_ALPHA:g}$ (trained)")
    ax.set_xlabel("normalized fuel (median over bearings)")
    ax.set_ylabel("normalized noise (median over bearings)")
    ax.set_title(f"Fuel-noise frontier under density scaling — {runway}")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"frontier_{runs_name}_{runway}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_failure_rate(rates: pd.DataFrame, runway: str, runs_name: str,
                      output_dir: Path) -> Path:
    """Percentage of non-completing episodes vs. density-scale alpha, per config."""
    configs = sorted(rates["config"].unique())
    color = {c: qual(i) for i, c in enumerate(configs)}

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c in configs:
        sub = rates[rates["config"] == c].sort_values("alpha")
        col = color[c]
        if len(sub) >= 2:
            ax.plot(sub["alpha"], sub["failure_pct"], "-o", color=col, label=c,
                    markersize=5)
        else:  # single point: no-map / legacy benchmark
            ax.scatter(sub["alpha"], sub["failure_pct"], marker="*", s=240,
                       facecolors=col, edgecolors="black", linewidths=1.0,
                       label=f"{c} (fixed)")

    ax.set_xscale("log")
    ax.set_xlabel(r"density-scale factor $\alpha$")
    ax.set_ylabel("non-completing episodes (%)")
    ax.set_title(f"Unsuccessful-run rate under density scaling — {runway}")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"failure_rate_{runs_name}_{runway}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=Path, help="folder of run dirs (e.g. runs/generalization)")
    parser.add_argument("--runway", default="EDDF_RW25R",
                        help="runway label as it appears in the scenario subdir (default: EDDF_RW25R)")
    parser.add_argument("--alphas", nargs="+", default=["0.1", "0.25", "0.5", "1", "2", "4", "10"],
                        help="density-scale factors; must match the labels used at generation time")
    parser.add_argument("--map-path", type=str,
                        default="./scripts/population_maps/europe_3035_1km.tif")
    parser.add_argument("--mean_episode_length", type=float, default=1400.0)
    parser.add_argument("--noise_clip_percentile", type=float, default=99.9)
    parser.add_argument("--no-match", action="store_true",
                        help="disable the matched-bearing filter (keep every bearing as-is)")
    parser.add_argument("--errorbars", action="store_true",
                        help="draw IQR error bars on each frontier point")
    parser.add_argument("--output_dir", type=Path, default=Path("plots/sweep_overview_plots"))
    parser.add_argument("--use-cache", action="store_true",
                        help="load frontier points from the saved CSV instead of recomputing")
    args = parser.parse_args()

    output_dir = args.output_dir / args.runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"frontier_{args.runs_root.name}_{args.runway}.csv"
    failure_csv_path = output_dir / f"failure_rate_{args.runs_root.name}_{args.runway}.csv"

    if args.use_cache:
        if not csv_path.exists():
            raise FileNotFoundError(f"Cache not found: {csv_path}")
        pts = pd.read_csv(csv_path)
        requested = [float(a) for a in args.alphas]
        pts = pts[pts["alpha"].isin(requested)]
        print(f"Loaded frontier points from cache → {csv_path}")
        print(pts.to_string(index=False))
        if not failure_csv_path.exists():
            raise FileNotFoundError(f"Cache not found: {failure_csv_path}")
        rates = pd.read_csv(failure_csv_path)
        rates = rates[rates["alpha"].isin(requested)]
        print(f"Loaded failure rates from cache → {failure_csv_path}")
    else:
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

        n_before = df["start_angle"].nunique()
        if not args.no_match:
            df = matched_filter(df, KEEP_REASONS)
        n_after = df["start_angle"].nunique()
        print(f"Bearings: {n_after}/{n_before} kept after matched filtering "
              f"({'disabled' if args.no_match else 'success/failed-approach, matched across all'})")
        pts = frontier_points(df)
        pts.to_csv(csv_path, index=False)
        print(f"Saved frontier points → {csv_path}")
        print(pts.to_string(index=False))

    out_path = plot_frontier(pts, args.runway, args.runs_root.name, output_dir, args.errorbars)
    print(f"Saved plot → {out_path}")
    failure_path = plot_failure_rate(rates, args.runway, args.runs_root.name, output_dir)
    print(f"Saved plot → {failure_path}")


if __name__ == "__main__":
    main()
