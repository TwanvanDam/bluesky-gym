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

Filtering (max-steps fairness): non-completing (max_steps / out_of_bounds) episodes carry
truncation-length fuel/noise, so the frontier is emitted in three variants:
  - unfiltered: all episodes, no filtering (legacy behavior);
  - omit_incomplete: every (config, alpha) point with at least one non-completing episode
    is dropped entirely — the failure table documents the gaps;
  - matched_bearings: every bearing that fails to complete in any config/seed/alpha is
    dropped from all of them, so each point is evaluated on an identical bearing set.
"""

import argparse
import re
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.common.colors import *
from scripts.common.figures import PLOT_TYPE_TO_SIZE, paper_axes, right_margin_x, save
from scripts.common.sweep_plotting import compute_episode_metrics, find_csv

# Extracts config + optional seed; handles both "name_seed00" and bare "name" forms.
PATTERN = re.compile(r"^(?P<config>.+?)(?:_seed(?P<seed>\d+))?$")

# "Successful" outcomes for the failure-rate plot and the failure table: anything else
# (failed_approach / max_steps / out_of_bounds) counts as a failure there.
KEEP_REASONS = {"success"} #, "failed_approach"}

# Episodes that complete a flight and therefore carry valid fuel/noise metrics; max_steps /
# out_of_bounds episodes are truncated and would inflate fuel. Used by the frontier filters.
COMPLETED_REASONS = {"success", "failed_approach"}

ANCHOR_ALPHA = 1.0  # the trained operating point

# CLI defaults (kept as module constants so the sweep's canonical settings live in one place).
DEFAULT_RUNWAY = "EDDF_RW25R"
DEFAULT_ALPHAS = ["0.25", "0.5", "1", "2", "4"]
DEFAULT_MAP_PATH = "./scripts/population_maps/europe_3035_1km.tif"
DEFAULT_MEAN_EPISODE_LENGTH = 1400.0
DEFAULT_NOISE_CLIP_PERCENTILE = 99.9
DEFAULT_OUTPUT_DIR = Path("plots/sweep_overview_plots")

# Alphas shown in the marker-size legend and used as failure-rate x-ticks.
LEGEND_ALPHAS = (0.25, 0.5, 1, 2, 4)
METRIC_REDUCTION = "mean"
# Figure geometry comes from common.figures, shared by both plots: the frontier
# carries two stacked legends (configs, marker size <-> alpha), so the right-hand
# strip has to be reserved up front instead of being discovered by a tight bbox.
FIGURE_WIDTH, FIGURE_HEIGHT = PLOT_TYPE_TO_SIZE["sweep_frontier"]
LEGEND_STRIP_IN = 1.35
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

# Exact-match display labels for legend text, one entry per config name produced by
# PATTERN (run dir name minus any trailing _seedNN suffix). Keeps the legend short
# enough to fit alongside the axes instead of being clipped by long raw config names.
CONFIG_DISPLAY_NAMES = {
    "sweep_2_no_map": "No-map",
    "sweep_2_centered_4": "C4-old",
    "multi_scale_3a": "3a (C4 + C16)",
    "transformed_baseline": "C4-new",
    "transformed_zoom": "Zoom",
    "transformed_scale": "Scale",
    "E_3_256-x1": "Groot et al.",
}

def post_process_metrics(df, mode: Literal["mean", "median", "iqr", "q1", "q3"]):
    match mode:
        case "mean":
            return df["fuel_mean"], df["noise_mean"]
        case "median":
            return df["fuel"], df["noise"]
        case "iqr":
            return df["fuel_q3"]-df["fuel_q1"], df["noise_q3"]-df["noise_q1"]
        case "q1":
            return df["fuel_q1"], df["noise_q1"]
        case "q3":
            return df["fuel_q3"], df["noise_q3"]
        case _:
            return None

def config_display_name(config: str) -> str:
    label = CONFIG_DISPLAY_NAMES.get(config)
    if label is None:
        warnings.warn(f"No display name for config {config!r}; using raw name "
                      "(add a CONFIG_DISPLAY_NAMES entry to give it a display label).")
        return config
    return label

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

    discarded_bearings = sorted(df.loc[~bearing_ok, "start_angle"].unique())
    if discarded_bearings:
        print(f"matched_bearings: discarding {len(discarded_bearings)} bearing(s) with a "
              f"non-completing episode in some config/seed/alpha: {[int(bear) for bear in discarded_bearings]}")

    return df[bearing_ok]


def drop_incomplete_points(df: pd.DataFrame, completed_reasons: set) -> pd.DataFrame:
    """Drop every (config, alpha) frontier point with at least one non-completing episode.

    Surviving points consist purely of completed episodes, so no further per-episode
    filtering is needed on top; the omitted points are documented by the failure table.
    """
    completed = df["termination_reason"].isin(completed_reasons)
    point_ok = completed.groupby([df["config"], df["alpha"]]).transform("all")
    return df[point_ok]


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
            fuel_mean=("normalized_fuel", lambda s: s.mean()),
            noise_mean=("normalized_noise", lambda s: s.mean()),
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


def config_linestyles(configs: list[str]) -> dict[str, str]:
    """Map each line-drawn config to its linestyle, shared by both plots so a config
    keeps the same colour+linestyle everywhere.

    Solid is the default; a config whose colour has already been used gets a dashed
    line so overlapping-colour series stay distinguishable. no_map configs are omitted
    (they render as a standalone point, not a line, and so don't consume a colour slot).
    """
    color_use_count: dict = {}
    styles: dict[str, str] = {}
    for c in configs:
        if "no_map" in c:
            continue
        color = config_color(c)
        seen = color_use_count.get(color, 0)
        color_use_count[color] = seen + 1
        styles[c] = "-" if seen == 0 else "--"
    return styles


def plot_frontier(pts: pd.DataFrame, runway: str, runs_name: str, output_dir: Path,
                  variant: str = "") -> Path:
    configs = sorted(pts["config"].unique())

    fig, ax = paper_axes(FIGURE_WIDTH, FIGURE_HEIGHT, right=LEGEND_STRIP_IN)

    linestyles = config_linestyles(configs)
    config_handles = []

    for c in configs:
        sub = pts[pts["config"] == c].sort_values("alpha")
        color = config_color(c)
        if "no_map" in c:
            continue
        # A real frontier: line + square markers sized by alpha.
        linestyle = linestyles[c]

        ax.plot(*post_process_metrics(sub, METRIC_REDUCTION), linestyle=linestyle, color=color, zorder=3)
        for fuel, noise, alpha in zip(*post_process_metrics(sub, METRIC_REDUCTION), sub["alpha"]):
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
                       markeredgewidth=MARKER_EDGE_WIDTH, markersize=7, label=config_display_name(c)))

    ax.set_xlabel(f"{METRICS["normalized_fuel"]} ({METRIC_REDUCTION} over bearings)")
    ax.set_ylabel(f"{METRICS["normalized_noise"]} ({METRIC_REDUCTION} over bearings)")
    ax.grid(True, alpha=0.3)

    # Both legends sit in the right-hand strip reserved by paper_axes, stacked at
    # the top and bottom of the axes, and share the framed black-edged style used
    # across the other sweep plots.
    margin_x = right_margin_x(ax)
    legend_main = ax.legend(handles=config_handles, frameon=True, edgecolor="k",
                            loc="upper left", bbox_to_anchor=(margin_x, 1))
    ax.add_artist(legend_main)

    # Secondary legend: marker size <-> alpha.
    size_alphas = [a for a in LEGEND_ALPHAS if a in set(pts["alpha"])]
    if size_alphas:
        size_handles = [
            plt.Line2D([0], [0], linestyle="none", marker="o", markeredgecolor="k" if a != 1 else "0.5",
                       markeredgewidth=MARKER_EDGE_WIDTH,
                       markersize=np.sqrt(alpha_to_size(a)), markerfacecolor="0.5" if a != 1 else "white",
                       label=r"$\kappa = $" + f"{a:g}")
            for a in size_alphas
        ]
        ax.legend(handles=size_handles,
                  frameon=True, edgecolor="k", loc="lower left", bbox_to_anchor=(margin_x, 0),
                  ncol=1, handletextpad=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{variant}" if variant else ""
    out_path = output_dir / f"frontier_{METRIC_REDUCTION}_{runs_name}_{runway}{suffix}.pdf"
    save(fig, out_path)
    plt.close(fig)
    return out_path


def plot_failure_rate(rates: pd.DataFrame, runway: str, runs_name: str,
                      output_dir: Path) -> Path:
    """Percentage of non-completing episodes vs. density-scale alpha, per config."""
    configs = sorted(rates["config"].unique())

    # Extra top room for the title, extra bottom for the math x-label's descender,
    # and a wider strip than the frontier's because the single-point configs carry
    # a "(fixed)" suffix in their legend label.
    fig, ax = paper_axes(FIGURE_WIDTH, FIGURE_HEIGHT,
                         right=LEGEND_STRIP_IN + 0.3, top=0.30, bottom=0.50)
    linestyles = config_linestyles(configs)
    for c in configs:
        sub = rates[rates["config"] == c].sort_values("alpha")
        col = config_color(c)
        if len(sub) >= 2:
            ax.plot(sub["alpha"], sub["failure_pct"], linestyle=linestyles[c], marker="o",
                    color=col, label=config_display_name(c), markersize=5)
        else:  # single point: no-map / legacy benchmark
            ax.scatter(sub["alpha"], sub["failure_pct"], marker="*", s=240,
                       facecolors=col, edgecolors="black", linewidths=1.0,
                       label=f"{config_display_name(c)} (fixed)")

    ax.set_xscale("log")
    ax.set_xticks(list(LEGEND_ALPHAS), labels=[str(a) for a in LEGEND_ALPHAS])
    ax.set_xlabel(r"density-scale factor $\alpha$")
    ax.set_ylabel("non-completing episodes (%)")
    ax.set_title(f"Unsuccessful-run rate under density scaling — {runway}")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, edgecolor="k", loc="upper left",
              bbox_to_anchor=(right_margin_x(ax), 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"failure_rate_{runs_name}_{runway}.pdf"
    save(fig, out_path)
    plt.close(fig)
    return out_path


# Order in which failure reasons appear inside a table cell; only reasons that actually
# occur in the data get a slot.
FAILURE_REASON_ORDER = ("failed_approach", "max_steps", "out_of_bounds")


def export_failure_table(df: pd.DataFrame, keep_reasons: set, runway: str,
                         runs_name: str, output_dir: Path) -> Path:
    r"""Write the failure counts as a LaTeX tabular (.tex) to \input into the paper.

    One row per config, one column per alpha. A cell holds the failure counts joined by
    "/" in FAILURE_REASON_ORDER, with "--" for a zero count, and is left empty where a
    config was not evaluated at that alpha (fixed-point configs). Computed on the
    unmatched dataframe, for the same reason as failure_rate_points. Emits only the
    tabular environment so caption/label/placement stay in the paper source.
    """
    failed = df[~df["termination_reason"].isin(keep_reasons)]
    reasons = [r for r in FAILURE_REASON_ORDER if r in set(failed["termination_reason"])]
    unknown = sorted(set(failed["termination_reason"]) - set(FAILURE_REASON_ORDER))
    if unknown:
        warnings.warn(f"Failure reason(s) {unknown} missing from FAILURE_REASON_ORDER; "
                      "appending them last.")
        reasons += unknown
    counts = (failed.groupby(["config", "alpha", "termination_reason"]).size()
              .unstack(fill_value=0).reindex(columns=reasons, fill_value=0))
    evaluated = df.groupby(["config", "alpha"]).size()
    n_unique = sorted(evaluated.unique())
    n_text = f"{n_unique[0]}" if len(n_unique) == 1 else f"{n_unique[0]}–{n_unique[-1]}"
    alphas = sorted(df["alpha"].unique())

    def cell(config: str, alpha: float) -> str:
        if (config, alpha) not in evaluated.index:
            return ""
        if (config, alpha) not in counts.index:
            return "--"
        vals = counts.loc[(config, alpha)]
        if not vals.any():
            return "--"
        return r"\,/\,".join(str(v) if v else "--" for v in vals)

    header = " & ".join([r"\textbf{Configuration}"]
                        + [rf"$\alpha={a:g}$" for a in alphas]) + r" \\"
    rows = [
        " & ".join([config_display_name(c)] + [cell(c, a) for a in alphas]) + r" \\"
        for c in sorted(df["config"].unique())
    ]
    lines = [
        f"% Auto-generated by plot_density_scaling_sweep.py ({runs_name}, {runway}); do not edit.",
        f"% Cell format: {' / '.join(reasons)} counts out of {n_text} episodes per cell;",
        "% -- = no failures, empty = config not evaluated at this alpha.",
        r"\begin{tabular}[c]{l" + "|c" * len(alphas) + "}",
        "\t\\hline",
        "\t" + header,
        "\t\\hline",
        *("\t" + row for row in rows),
        "\t\\hline",
        r"\end{tabular}",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"failure_table_{runs_name}_{runway}.tex"
    out_path.write_text("\n".join(lines) + "\n")
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
                        help="load the saved per-bearing metrics CSV instead of re-reading "
                             "every trajectory (reduction/filtering still runs on load)")
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


def _load_cached_metrics(cache_path: Path, alphas: list[str], runs_root: Path) -> pd.DataFrame:
    """Reload the previously-saved per-bearing metrics, filtered to `alphas`.

    This is the raw per-trajectory table (fuel/noise/termination_reason per
    config/seed/alpha/start_angle) — the same granularity the other sweeps cache — so the
    filtering + reduction (frontier_points / failure_rate_points) can be changed later
    without re-reading every trajectory CSV. Errors if the cache lacks any config found
    under `runs_root` or any requested alpha.
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    requested = [float(a) for a in alphas]
    df = pd.read_csv(cache_path)
    _check_cache_complete(df, runs_root, requested)
    df = df[df["alpha"].isin(requested)]
    print(f"Loaded per-bearing metrics from cache → {cache_path}")
    return df


def _collect_metrics(args: argparse.Namespace, cache_path: Path) -> pd.DataFrame:
    """Collect the long per-bearing metrics table from trajectory CSVs and cache it.

    Returns one row per (config, seed, alpha, start_angle) carrying fuel, noise and
    termination_reason (see collect_scaling_metrics). The reduction to frontier / failure
    points happens downstream, so caching this table lets the analysis change later.
    """
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

    df.to_csv(cache_path, index=False)
    print(f"Saved per-bearing metrics → {cache_path}")
    return df


def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir / args.runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # The per-bearing metrics cache lives alongside the runs it summarizes (runs/<sweep>/),
    # not with the plot images, so the recomputed data stays next to its source trajectories.
    cache_path = args.runs_root / f"cached_metrics_{args.runs_root.name}_{args.runway}.csv"

    if args.use_cache:
        df = _load_cached_metrics(cache_path, args.alphas, args.runs_root)
    else:
        df = _collect_metrics(args, cache_path)

    # Reduction happens here (not in the cache) so the filtering / data-reduction method can
    # be changed and replayed with --use-cache. Failure rate is computed on the unmatched df,
    # since a matched filter would drop the very (non-completing) episodes it is meant to show.
    rates = failure_rate_points(df, KEEP_REASONS)
    pts = frontier_points(df)
    print(pts.to_string(index=False))

    out_path = plot_frontier(pts, args.runway, args.runs_root.name, output_dir)
    print(f"Saved plot → {out_path}")
    # Completion-filtered variants (see module docstring): omit whole points vs. omit the
    # non-completing bearings everywhere.
    for variant, filtered in (
        ("omit_incomplete", drop_incomplete_points(df, COMPLETED_REASONS)),
        ("matched_bearings", matched_filter(df, COMPLETED_REASONS)),
    ):
        variant_path = plot_frontier(frontier_points(filtered), args.runway,
                                     args.runs_root.name, output_dir, variant=variant)
        print(f"Saved plot → {variant_path}")
    failure_path = plot_failure_rate(rates, args.runway, args.runs_root.name, output_dir)
    print(f"Saved plot → {failure_path}")
    table_path = export_failure_table(df, KEEP_REASONS, args.runway, args.runs_root.name, Path("~/Thesis-Paper---Twan-van-Dam/content/results/tables"))
    print(f"Saved table → {table_path}")


if __name__ == "__main__":
    main()
