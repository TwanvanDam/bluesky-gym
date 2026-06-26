"""
Cross-sweep "weird comparison": put a few hand-picked single-scale resolution
configs next to the two multi-scale configs that reuse the same resolutions, so
the multi-scale gain (or lack of it) is visible on one axis.

    python -m scripts.plot_weird_comparison [--scenario EHAM_RW27]

Reads the per-sweep metric caches written by plot_resolution_sweep /
plot_multi_scale_sweep (their `--cache` flag), so it needs no BlueSky and no
trajectory re-processing — just run those two scripts with --cache first.
"""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from pydantic import color

from scripts.common.colors import qual
from scripts.common.sweep_plotting import add_reward, boxplot_stats, draw_boxplot

BOX_WIDTH = 0.5

RESOLUTION_ROOT = Path("runs/resolution_sweep_2")
MULTI_SCALE_ROOT = Path("runs/multi-scale-sweep")

# Each comparison variant: (label, runs_root, row filter into that sweep's cache).
# c{N} are single-scale centered runs from the resolution sweep; 5a/5b are the
# multi-scale configs that combine those same resolutions.
VARIANT_SPECS: list[tuple[str, Path, dict]] = [
    ("c2",  RESOLUTION_ROOT,  {"mode": "centered", "resolution": 2}),
    ("c8",  RESOLUTION_ROOT,  {"mode": "centered", "resolution": 8}),
    ("c16", RESOLUTION_ROOT,  {"mode": "centered", "resolution": 16}),
    ("5a",  MULTI_SCALE_ROOT, {"group_num": 5, "variant": "a"}),
    ("5b",  MULTI_SCALE_ROOT, {"group_num": 5, "variant": "b"}),
]

variants = [label for label, _, _ in VARIANT_SPECS]

VARIANT_TO_CAPTION = {
    "c2":  "C2",
    "c8":  "C8",
    "c16": "C16",
    "5a":  "C2 + C16\n(5a)",
    "5b":  "C2 + C8\n(5b)",
}

VARIANT_TO_COLOR = {
    "c2" : "grey",
    "c8" : "grey",
    "c16" : "grey",
    "5a" : "green",
    "5b" : "green",
}

COMPARISONS = [
    ("c2", "5a", "c16"),
    ("c2", "5b", "c8")
]

METRICS = [
    ("fuel", "fuel [kg]"),
    ("noise", "noise [W·s]"),
    ("normalized_fuel", "normalized fuel"),
    ("normalized_noise", "normalized noise"),
    ("combined", "normalized fuel + noise"),
    ("reward", "reward"),
    ("reward_unclipped", "reward (no noise clipping"),
]


def _load_cache(runs_root: Path, scenario: str) -> pd.DataFrame:
    """Read the cached per-episode metric table a sweep script wrote with --cache."""
    cache_path = runs_root / f"cached_metrics_{scenario}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cache at {cache_path}. Run the corresponding sweep plot script "
            f"with --cache --scenario {scenario} first."
        )
    return pd.read_csv(cache_path)


def build_comparison_df(scenario: str) -> pd.DataFrame:
    """Pull each VARIANT_SPECS row out of its sweep cache into one tidy frame.

    The result has all the per-episode metric columns plus a single `variant`
    column whose values are the comparison labels (c2, c8, c16, 5a, 5b). The
    derived `combined` / `reward` / `reward_unclipped` columns are not cached, so
    they are recomputed here exactly as the sweep scripts do before plotting.
    """
    caches: dict[Path, pd.DataFrame] = {}
    frames = []
    for label, runs_root, row_filter in VARIANT_SPECS:
        if runs_root not in caches:
            caches[runs_root] = _load_cache(runs_root, scenario)
        df = caches[runs_root]
        mask = pd.Series(True, index=df.index)
        for column, value in row_filter.items():
            mask &= df[column] == value
        subset = df[mask].copy()
        if subset.empty:
            print(f"  Warning: no cached rows for variant {label} ({row_filter}) in {runs_root}")
            continue
        subset["variant"] = label
        frames.append(subset)

    out = pd.concat(frames, ignore_index=True)
    out["combined"] = out["normalized_fuel"] + out["normalized_noise"]
    add_reward(out)
    return out


def plot_metric_boxplot(
        df: pd.DataFrame,
        baseline_df: pd.DataFrame | None,
        metric: str,
        ylabel: str,
        scenario: str,
        runs_name: str,
        output_dir: Path,
) -> list[dict]:
    fig, ax = plt.subplots(figsize=(8, 5))
    legend_handles = []
    rows: list[dict] = []

    has_baseline = baseline_df is not None and not baseline_df.empty

    # # Reference baseline box + quartile lines spanning the plot for comparison.
    # if has_baseline:
    #     draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
    #     legend_handles.append(
    #         plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=0.6, label="Baseline (C4)")
    #     )
    #     s = boxplot_stats(baseline_df[metric].values)
    #     rows.append({"variant": "baseline", "metric": metric, **s})
    #     for val, ls in [(s["q50"], "--"), (s["q25"], ":"), (s["q75"], ":")]:
    #         ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=0.6)

    # One box per transform variant.
    tick_x = []
    tick_labels = []
    i = 0
    for comparison in COMPARISONS:
        for variant in comparison:
            data = df[df["variant"] == variant][metric].values
            if len(data) == 0:
                continue
            rows.append({"variant": variant, "metric": metric, **boxplot_stats(data)})
            draw_boxplot(ax, data, position=i + 1, color=VARIANT_TO_COLOR[variant], box_width=BOX_WIDTH)
            tick_x.append(i+1)
            tick_labels.append(VARIANT_TO_CAPTION[variant])
            if i == 2:
                ax.axvline(x = i + 1.5, color="grey", alpha=0.5)
            i += 1
    ax.grid(axis='y')
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_ylabel(ylabel)
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_{runs_name}_{scenario}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-sweep comparison from cached metrics")
    parser.add_argument("--scenario", type=str, default="EHAM_RW27",
                        help="evaluation-scenario cache to read (must exist in both sweeps)")
    parser.add_argument("--output_dir", type=Path, default=Path("plots/weird_comparison"))
    args = parser.parse_args()

    df = build_comparison_df(args.scenario)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for metric, ylabel in METRICS:
        all_rows.extend(plot_metric_boxplot(
            df, None, metric, ylabel, args.scenario, "weird_comparison", args.output_dir,
        ))

    csv_path = args.output_dir / f"boxplot_stats_weird_comparison_{args.scenario}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")
