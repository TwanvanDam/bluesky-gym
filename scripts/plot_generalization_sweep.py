"""
Plot all runs in a generalization folder: episode-outcome breakdown and/or cost metrics.
Config names are the x-axis; individual seeds shown as dots like the transform sweep.
Runs without a seed suffix (e.g. E_3_256-x1) are treated as a single-seed entry.

    python -m scripts.plot_generalization_sweep runs/generalization
    python -m scripts.plot_generalization_sweep runs/generalization --plots breakdown
    python -m scripts.plot_generalization_sweep runs/generalization --baseline \\
        runs/generalization/transformed_baseline_seed00 \\
        runs/generalization/transformed_baseline_seed01 \\
        runs/generalization/transformed_baseline_seed02

The metrics view writes one PDF per metric *and*, per filter mode,
`metrics_grid_*.pdf`: the reward/noise/fuel panels laid out as one matplotlib
figure, the config key in the cell the odd panel count leaves free, and the
(a), (b), … captions above the panels. That grid replaces the 2x2 block of
`subfigure`s in the paper — the panels are laid out in inches, so they are
guaranteed the same axes size and the same text size, and LaTeX never rescales
them. The `figure` environment to paste is printed at the end.
"""

import re
import warnings
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.colors import *
from scripts.common.figures import (
    PANEL_LETTERS, PLOT_TYPE_TO_SIZE, W_FULL,
    grid_caption, grid_latex_snippet, legend_in_cell, legend_right, metric_grid,
    outcome_ylim, paper_axes, save,
)
from scripts.common.sweep_plotting import (
    REASON_LABELS,
    add_reward,
    add_scenario_id,
    boxplot_stats,
    collect_baseline_metrics,
    collect_breakdown_data,
    collect_run_metrics,
    compute_baseline,
    draw_boxplot,
    find_csv,
    mean_breakdowns,
    run_sweep_args_parser,
)

# Figure geometry comes from common.figures: every panel is saved at exactly its
# LaTeX slot size, so nothing is rescaled on inclusion. The breakdown legend lives
# in a reserved right-hand strip; the config legend for the metric panels is a
# separate PDF, one panel tall so the two line up side by side. That legend is two
# columns wide (short code + full config name), hence its own strip width.
METRIC_WIDTH, METRIC_HEIGHT = PLOT_TYPE_TO_SIZE["sweep_metric"]
BREAKDOWN_WIDTH, BREAKDOWN_HEIGHT = PLOT_TYPE_TO_SIZE["sweep_breakdown"]
LEGEND_STRIP_IN = 1.7
CONFIG_LEGEND_STRIP_IN = 2.2

BOX_WIDTH = 0.6
BAR_WIDTH = 0.7
DOT_SIZE = 60
DOT_ALPHA = 0.8
BAR_ALPHA = 0.6

# Extracts config + optional seed; handles both "name_seed00" and bare "name" forms.
PATTERN = re.compile(r"^(?P<config>.+?)(?:_seed(?P<seed>\d+))?$")

# Panels of the combined figure (common.figures.metric_grid owns its geometry);
# the leftover cell holds the config key. One grid per filter mode. Which
# metrics get a panel, and every axis label, come from common.colors so this
# sweep's panels match the other sweeps'.
GRID_COLS = 2
GRID_WIDTH = W_FULL

REASON_HATCH = {
    "success":         "",
    "failed_approach": "////",
    "max_steps":       "....",
    "out_of_bounds":   "xxxx",
}

# success/failed_approach are filled with the config color (hatch drawn on top);
# the remaining failure modes are hatch-only so they don't compete visually with
# the arrival-rate segments.
FILLED_REASONS = {"success", "failed_approach"}

# Checked in order; first substring match wins. Config names come from PATTERN
# (e.g. "centered_16_all"), so exact-match lookup would miss most real configs.
CONFIG_COLOR_RULES = {
    "no_map": BASELINE_COLOR,
    "centered": CENTERED_COLOR,
    "forward": FORWARD_COLOR,
    "multi_scale": MULTI_SCALE_COLOR,
    "transformed_baseline": BASELINE_COLOR,
    "transformed": TRANSFORMS_COLOR,
    "E_3_256": HIGHLIGHT_COLOR,
}

def config_color(config: str) -> str:
    for needle, color in CONFIG_COLOR_RULES.items():
        if needle in config:
            return color
    warnings.warn(f"No color rule matches config {config!r}; using UNKNOWN_COLOR "
                   "(add a CONFIG_COLOR_RULES entry to give it its own color).")
    return UNKNOWN_COLOR

# Exact-match display labels for xtick text, one entry per config name produced by
# PATTERN. Keep in sync with runs/generalization (config = run dir name minus any
# trailing _seedNN suffix).
CONFIG_TICK_LABELS = {
    "sweep_2_no_map": "No-map",
    "sweep_2_centered_4": "C4-old",
    "multi_scale_3a": "3a (C4 + C16)",
    "transformed_baseline": "C4-new",
    "transformed_zoom": "Zoom",
    "transformed_scale": "Scale",
    "E_3_256-x1": "Groot et al.",
}

def config_tick_label(config: str) -> str:
    label = CONFIG_TICK_LABELS.get(config)
    if label is None:
        warnings.warn(f"No tick label for config {config!r}; using raw name "
                       "(add a CONFIG_TICK_LABELS entry to give it a display label).")
        return config
    return label

# Short x-axis codes. The legend panel (save_legend) expands these to full names,
# so the axis stays uncluttered while the panel carries the key. Keep in sync with
# CONFIG_TICK_LABELS.
CONFIG_SHORT_CODES = {
    "sweep_2_no_map":       "NM",
    "transformed_baseline": "C4n",
    "sweep_2_centered_4":   "C4o",
    "multi_scale_3a":       "3a",
    "transformed_zoom":     "Z",
    "transformed_scale":    "S",
    "E_3_256-x1":           "Bench",
}

def config_short_code(config: str) -> str:
    code = CONFIG_SHORT_CODES.get(config)
    if code is None:
        warnings.warn(f"No short code for config {config!r}; using tick label "
                       "(add a CONFIG_SHORT_CODES entry).")
        return config_tick_label(config)
    return code

# Left-to-right x-axis order. Configs not listed here are appended afterwards,
# alphabetically, with a warning (so a new run still plots instead of vanishing).
CONFIG_ORDER = [
    "E_3_256-x1",           # yellow  — benchmark
    "sweep_2_no_map",       # grey    — baseline
    "transformed_baseline", # grey    — baseline
    "sweep_2_centered_4",   # orange  — single-scale
    "multi_scale_3a",       # purple  — multi-scale
    "transformed_zoom",     # green   — domain randomisation
    "transformed_scale",    # green   — domain randomisation
]

def _ordered_configs(present: set[str]) -> list[str]:
    ordered = [c for c in CONFIG_ORDER if c in present]
    unknown = sorted(present - set(CONFIG_ORDER))
    if unknown:
        warnings.warn(f"Config(s) {unknown} not in CONFIG_ORDER; appending alphabetically "
                       "(add them to CONFIG_ORDER to control placement).")
    return ordered + unknown


def _normalize_seed(s):
    """Normalize a seed value: NaN (pandas None) → None for consistent dict keys."""
    return None if pd.isna(s) else s


def _seed_color_map(df: pd.DataFrame) -> dict:
    all_seeds = sorted({_normalize_seed(s) for s in df["seed"].unique()}, key=lambda s: s if s is not None else -1)
    return {seed: SEED_COLORS[i % len(SEED_COLORS)] for i, seed in enumerate(all_seeds)}


# ---------------------------------------------------------------------------- metrics

# Identifies one (config, seed) run. Shared by add_scenario_id (in plot_metrics) and
# filter_valid_perseed so the two can't silently drift onto different grouping keys.
SCENARIO_GROUP_COLS = ["config", "seed"]

# An episode reaching one of these termination_reasons flew the full approach, as
# opposed to being cut off early (max_steps / out_of_bounds) before any outcome
# was resolved. "Completed" says nothing about whether it landed successfully —
# see filter_full / add_reward for that distinction.
_COMPLETED_REASONS = {"success", "failed_approach"}


def _is_completed(df: pd.DataFrame) -> pd.Series:
    """True for episodes that flew the full approach (success or failed_approach).

    Cached metrics CSVs written before termination_reason was tracked only have a
    `success` column and no way to distinguish "cut off early" from "flew the full
    approach but failed" — for those, `success` is the closest available proxy and
    is used as-is (regenerate the cache to get the precise split; see class docs).
    """
    if "termination_reason" not in df.columns:
        return df["success"]
    return df["termination_reason"].isin(_COMPLETED_REASONS)


def _scenario_ids_completed_everywhere(completed: pd.Series, scenario_id: pd.Series) -> set:
    """scenario_ids where every row's `completed` value is True.

    The caller may pass one config, the whole sweep, or the sweep plus baseline —
    whatever rows are represented must ALL have completed a given bearing for its
    scenario_id to survive here, which is what makes callers directly comparable
    on that bearing.
    """
    completed_per_scenario = completed.groupby(scenario_id).all()
    return set(completed_per_scenario.index[completed_per_scenario])

def filter_full(df: pd.DataFrame) -> pd.DataFrame:
    """All episodes, including failures — reward's -1 failure penalty must count them."""
    return df

def filter_valid_perconfig(df: pd.DataFrame) -> pd.DataFrame:
    """Completed episodes only; each config judged on its own completed flights."""
    return df[_is_completed(df)]

def filter_valid_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Completed episodes, restricted to bearings every config/seed (baseline included) completed."""
    completed_mask = _is_completed(df)
    completed = df[completed_mask]
    if "scenario_id" not in df.columns:
        return completed
    common_ids = _scenario_ids_completed_everywhere(completed_mask, df["scenario_id"])
    if common_ids:
        completed = completed[completed["scenario_id"].isin(common_ids)]
    return completed

def filter_valid_perseed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every episode of a (config, seed) run if any one of its bearings failed to complete."""
    group_cols = [c for c in SCENARIO_GROUP_COLS if c in df.columns]
    if not group_cols:
        return df[_is_completed(df)]
    seed_fully_completed = _is_completed(df).groupby([df[c] for c in group_cols]).transform("all")
    return df[seed_fully_completed]

# Each mode pairs its filter function with the filename prefix it should produce.
FILTERS = {
    "full": (filter_full, "full"),
    "valid_perconfig": (filter_valid_perconfig, "per_config"),
    "valid_matched": (filter_valid_matched, "matched"),
    "valid_perseed": (filter_valid_perseed, "per_seed"),
}


def summarize_filter(metrics: pd.DataFrame, filtered: pd.DataFrame, mode: str) -> dict:
    """Episode/config/(config, seed) counts a filter mode removes.

    A config or seed whose episodes are ALL dropped by a mode never gets a tick in
    plot_metric_boxplot (it only iterates `filtered["config"].unique()`), so it
    otherwise disappears from that mode's plots with no visible indication.
    """
    before_configs = set(metrics["config"].dropna().unique())
    after_configs = set(filtered["config"].dropna().unique())
    before_pairs = set(map(tuple, metrics[SCENARIO_GROUP_COLS].drop_duplicates().to_numpy()))
    after_pairs = set(map(tuple, filtered[SCENARIO_GROUP_COLS].drop_duplicates().to_numpy()))
    return {
        "filter_mode": mode,
        "episodes_before": len(metrics),
        "episodes_after": len(filtered),
        "episodes_dropped": len(metrics) - len(filtered),
        "configs_total": len(before_configs),
        "configs_fully_dropped": len(before_configs - after_configs),
        "config_seed_total": len(before_pairs),
        "config_seed_fully_dropped": len(before_pairs - after_pairs),
    }


def draw_metric_boxplot(ax, df, metric, ylabel, filter_mode: str, report: bool = True) -> list[dict]:
    """Draw one metric's boxplot on `ax`. `df` must already be filtered by the caller.

    `df` holds both the swept configs and, if `is_baseline` is set on any row, the
    pooled baseline; the baseline gets position 0 and dashed reference lines,
    everything else is a regular tick. Split out of :func:`plot_metric_boxplot`
    so the standalone PDF and the combined grid draw the exact same panel;
    `report` is off for the grid so the stats are not printed a second time.
    """
    has_baseline = df["is_baseline"].any()
    configs = _ordered_configs(set(df.loc[~df["is_baseline"], "config"].dropna().unique()))
    rows: list[dict] = []

    if has_baseline:
        bvals = df[df["is_baseline"]][metric].values
        if len(bvals):
            draw_boxplot(ax, bvals, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
            s = boxplot_stats(bvals)
            if report:
                print(f"  [{filter_mode}] {'baseline':>18}  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}  "
                      f"IQR={s['iqr']:8.3f}  whiskers=[{s['whisker_lo']:8.3f}, {s['whisker_hi']:8.3f}]  mean={bvals.mean():8.3f}")
            rows.append({"config": "baseline", "metric": metric, "filter_mode": filter_mode, "mean": bvals.mean(), **s})
            for val, ls in [(s["q50"], "--"), (s["q25"], ":"), (s["q75"], ":")]:
                ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=BOXPLOT_ALPHA)

    offset = 1 if has_baseline else 0
    for i, config in enumerate(configs):
        data = df[df["config"] == config][metric].values
        if len(data):
            s = boxplot_stats(data)
            if report:
                print(f"  [{filter_mode}] {config:>18}  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}  "
                      f"IQR={s['iqr']:8.3f}  whiskers=[{s['whisker_lo']:8.3f}, {s['whisker_hi']:8.3f}]  mean={data.mean():8.3f}")
            rows.append({"config": config, "metric": metric, "filter_mode": filter_mode, "mean": data.mean(), **s})
            draw_boxplot(ax, data, position=i + offset, color=config_color(config), box_width=BOX_WIDTH)

    tick_x = ([0] if has_baseline else []) + [i + offset for i in range(len(configs))]
    tick_labels = (["Base"] if has_baseline else []) + [config_short_code(c) for c in configs]
    ax.grid(axis="y")
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=9, rotation=0, ha="center")
    ax.yaxis.set_inverted(METRIC_TO_AXIS_REVERS[metric])
    ax.set_ylabel(ylabel)
    return rows


def plot_metric_boxplot(df, metric, ylabel, scenario, runs_name, output_dir, filter_mode: str) -> list[dict]:
    fig, ax = paper_axes(METRIC_WIDTH, METRIC_HEIGHT)
    rows = draw_metric_boxplot(ax, df, metric, ylabel, filter_mode)

    _, prefix = FILTERS[filter_mode]
    out_path = output_dir / prefix / f"{metric}_{runs_name}_{scenario}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(fig, out_path)
    plt.close(fig)
    return rows

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def config_legend_entries() -> tuple[list, list[str]]:
    """One row per config: colour swatch + short code + full name.

    Returns ``(handles, labels)`` for a two-column legend. matplotlib fills
    legend columns TOP-TO-BOTTOM (column-major), so this is the whole first
    column (swatch + code) followed by the whole second column (full name).
    """
    phantom = Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)

    swatch_handles, code_labels = [], []
    name_handles,  name_labels  = [], []

    for c in _ordered_configs(set(CONFIG_SHORT_CODES)):
        swatch_handles.append(Rectangle((0, 0), 1, 1, fc=config_color(c), alpha=BOXPLOT_ALPHA))
        code_labels.append(config_short_code(c))
        name_handles.append(phantom)
        name_labels.append(config_tick_label(c))

    return swatch_handles + name_handles, code_labels + name_labels


def save_legend(output_dir: Path, runs_name: str, scenario: str) -> None:
    handles, labels = config_legend_entries()

    fig = plt.figure(figsize=(CONFIG_LEGEND_STRIP_IN, METRIC_HEIGHT * TEXTWIDTH_IN))
    legend = fig.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        ncol=2,
        handlelength=1.2,
        columnspacing=0.5,
        handletextpad=0.5,
    )
    legend.get_frame().set_edgecolor("k")

    out_path = output_dir / f"legend_modes_{runs_name}_{scenario}.pdf"
    save(fig, out_path)
    plt.close(fig)


def plot_metric_grid(df, runs_name, scenario, output_dir, filter_mode: str,
                     metrics: Mapping[str, str] = METRIC_TO_CAPTION,
                     width: float = GRID_WIDTH, ncols: int = GRID_COLS) -> Path:
    """One figure per filter mode holding every grid metric, config key in the spare cell.

    Replaces the 2x2 block of ``subfigure``s in the paper: laying the panels out
    here means they are guaranteed the same axes size and the same text size,
    and the legend costs nothing extra because it goes in the cell the odd
    number of metrics leaves empty.
    """
    fig, panel_axes, legend_ax = metric_grid(len(metrics), ncols=ncols, width=width)

    for ax, letter, (metric, caption) in zip(panel_axes, PANEL_LETTERS, metrics.items()):
        draw_metric_boxplot(ax, df, metric, METRICS[metric], filter_mode, report=False)
        grid_caption(ax, letter, caption)
    handles, labels = config_legend_entries()
    legend_in_cell(fig, legend_ax, handles, labels=labels, ncol=2,
                   handlelength=1.2, columnspacing=0.5, handletextpad=0.5)

    _, prefix = FILTERS[filter_mode]
    out_path = output_dir / prefix / f"metrics_grid_{runs_name}_{scenario}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(fig, out_path)
    plt.close(fig)
    print("\n" + grid_latex_snippet(out_path, list(metrics.values()), width) + "\n")
    return out_path


def plot_metrics(metrics, runs_root, scenario, output_dir) -> None:
    """`metrics` holds the swept configs and, if present, the pooled baseline (is_baseline column)."""
    # NaN seeds (runs without a seed suffix) are dropped by groupby; use -1 as sentinel.
    metrics["seed"] = metrics["seed"].fillna(-1)
    add_scenario_id(metrics, SCENARIO_GROUP_COLS)

    all_rows: list[dict] = []
    filter_summaries: list[dict] = []
    for mode, (filt, _) in FILTERS.items():
        filtered = filt(metrics)
        summary = summarize_filter(metrics, filtered, mode)
        filter_summaries.append(summary)
        print(f"[{mode}] episodes kept {summary['episodes_after']}/{summary['episodes_before']} "
              f"({summary['episodes_dropped']} dropped) — "
              f"configs fully dropped {summary['configs_fully_dropped']}/{summary['configs_total']}, "
              f"(config, seed) runs fully dropped {summary['config_seed_fully_dropped']}/{summary['config_seed_total']}")
        for metric, ylabel in METRICS.items():
            all_rows.extend(plot_metric_boxplot(filtered, metric, ylabel,
                                                scenario, runs_root.name, output_dir, mode))
        plot_metric_grid(filtered, runs_root.name, scenario, output_dir, mode)
    csv_path = output_dir / f"boxplot_stats_{runs_root.name}_{scenario}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    save_legend(output_dir, runs_root.name, scenario)

    summary_path = output_dir / f"filter_summary_{runs_root.name}_{scenario}.csv"
    pd.DataFrame(filter_summaries).to_csv(summary_path, index=False)
    print(f"Saved → {summary_path}")

# --------------------------------------------------------------------------- breakdown

def print_success_rates(breakdown: pd.DataFrame, baseline: float | None = None) -> list[dict]:
    rows: list[dict] = []
    if baseline is not None:
        print(f"  {'baseline':>18}  success_rate={baseline:.1%}")
        rows.append({"config": "baseline", "success_rate_mean": baseline})
    for config in _ordered_configs(set(breakdown["config"].dropna().unique())):
        rates = breakdown[breakdown["config"] == config]["success_rate"].values
        label = config_tick_label(config)
        print(f"  {label:>18}  success_rate={rates.mean():.1%}  (seeds: {', '.join(f'{r:.1%}' for r in rates)})")
        rows.append({"config": config, "success_rate_mean": rates.mean(), "success_rate_seeds": list(rates)})
    return rows


def plot_episode_success(ax, df: pd.DataFrame, baseline: float | None = None) -> None:
    configs = _ordered_configs(set(df["config"].dropna().unique()))
    x = np.arange(len(configs))
    seen_reasons: set = set()

    def _bar(xi, h, bottom_, color, reason):
        hatch = REASON_HATCH.get(reason, "")
        if reason in FILLED_REASONS:
            ax.bar(xi, h, width=BAR_WIDTH, bottom=bottom_, color=color,
                   alpha=BAR_ALPHA, hatch=hatch, edgecolor="black", linewidth=0.5)
        else:
            ax.bar(xi, h, width=BAR_WIDTH, bottom=bottom_, facecolor="none",
                   hatch=hatch, edgecolor="black", linewidth=0.5)

    # Each config keeps its sweep color; the termination reason is shown by hatch.
    ordered, means = mean_breakdowns(df, configs, pos_col="config")
    bottom = np.zeros(len(configs))
    for reason in ordered:
        for i, config in enumerate(configs):
            _bar(x[i], means[reason][i], bottom[i], config_color(config), reason)
        bottom += means[reason]
        seen_reasons.add(reason)

    min_seed_rates = 1.0
    for i, config in enumerate(configs):
        seed_rates = {_normalize_seed(row["seed"]): row["success_rate"] for _, row in df[df["config"] == config].iterrows()}
        seeds = sorted(seed_rates, key=lambda s: s if s is not None else -1)
        jitter = np.linspace(-0.08, 0.08, len(seeds))
        min_seed_rates = min([min_seed_rates, *seed_rates.values()])
        for xi, seed in zip(jitter, seeds):
            ax.scatter(x[i] + xi, seed_rates[seed],
                       color="black", s=DOT_SIZE, zorder=5,
                       alpha=DOT_ALPHA, edgecolors="white", linewidths=0.8)

    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline success ({baseline:.0%})", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([config_short_code(c) for c in configs], fontsize=9, rotation=0, ha="center")
    ax.set_ylabel("Episode outcome fraction")
    ax.grid(axis="y")
    outcome_ylim(ax, min_seed_rates)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    legend_handles = []
    if baseline is not None:
        legend_handles.append(plt.Line2D([0], [0], color=BASELINE_COLOR, linestyle="--",
                                         label=f"Baseline success ({baseline:.0%})"))
    for reason in [r for r in REASON_HATCH if r in seen_reasons]:
        fc = "lightgray" if reason in FILLED_REASONS else "none"
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, fc=fc, hatch=REASON_HATCH[reason], edgecolor="black",
            label=REASON_LABELS.get(reason, reason)))
    legend_handles.append(plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="black",
        markersize=8, label="Per-seed success rate"))
    legend_right(ax, handles=legend_handles, frameon=True, edgecolor="k")

def plot_breakdown(breakdown, baseline_rate, runs_root, scenario, output_dir):
    rows = print_success_rates(breakdown, baseline_rate)
    fig, ax = paper_axes(BREAKDOWN_WIDTH, BREAKDOWN_HEIGHT, right=LEGEND_STRIP_IN)
    plot_episode_success(ax, breakdown, baseline=baseline_rate)
    out_path = output_dir / f"episode_success_{runs_root.name}_{scenario}.pdf"
    save(fig, out_path)
    plt.close(fig)

    csv_path = output_dir / f"success_rates_{runs_root.name}_{scenario}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

if __name__ == "__main__":
    args, selected = run_sweep_args_parser()

    runs_root = Path(args.runs_root)
    output_dir = args.output_dir / runs_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if "metrics" in selected:
        import bluesky as bs
        from bluesky_gym.maps.map_sources import TransformedTiffMapSourceConfig
        from bluesky_gym.metrics.evaluation_metrics import (
            bounds_from_df,
            build_metric_fn,
            make_pop_samplers,
        )

        bs.init()
        all_csvs = [
            find_csv(run_dir, args.scenario)
            for run_dir in runs_root.iterdir()
            if run_dir.is_dir()
        ]
        all_csvs = [p for p in all_csvs if p is not None]
        if not all_csvs:
            raise FileNotFoundError(f"No trajectory CSVs found under {runs_root} for scenario '{args.scenario}'")
        combined_df = pd.concat([pd.read_csv(p) for p in all_csvs], ignore_index=True)
        bounds = bounds_from_df(combined_df)

        samplers = make_pop_samplers(
            TransformedTiffMapSourceConfig(file_path=args.map_path), bounds=bounds,
            clip_percentile=args.noise_clip_percentile, train_resampling="average", true_resampling="average")
        calculate_metrics = build_metric_fn(samplers)

        cache_path = runs_root / f"cached_metrics_{args.scenario}.csv"
        if args.cache and cache_path.exists():
            print("Using cached metrics...")
            run_metrics = pd.read_csv(cache_path)
        else:
            run_metrics = collect_run_metrics(
                runs_root, PATTERN, args.scenario, calculate_metrics, args.mean_episode_length)
            if args.cache:
                print(f"Saving metrics to {cache_path} ...")
                run_metrics.to_csv(cache_path, index=False)

        run_metrics["is_baseline"] = False
        metrics = run_metrics
        if args.baseline:
            baseline_metrics = collect_baseline_metrics(
                list(args.baseline), args.scenario, calculate_metrics, args.mean_episode_length)
            baseline_metrics["config"] = "baseline"
            baseline_metrics["is_baseline"] = True
            metrics = pd.concat([run_metrics, baseline_metrics], ignore_index=True)

        if not metrics.empty:
            metrics["combined"] = metrics["normalized_fuel"] + metrics["normalized_noise"]
            add_reward(metrics)

        plot_metrics(metrics, runs_root, args.scenario, output_dir)

    if "breakdown" in selected:
        breakdown = collect_breakdown_data(runs_root, PATTERN, args.scenario)
        if not breakdown.empty:
            baseline_rate = baseline_length = None
            if args.baseline:
                baseline_rate, baseline_length = compute_baseline(args.baseline, args.scenario)
                if baseline_rate is None:
                    print(f"Baseline — no usable trajectory data in {args.baseline} (plotting without baseline)")
                else:
                    print(f"Baseline — success rate: {baseline_rate:.1%}, mean length: {baseline_length:.1f} s")
            plot_breakdown(breakdown, baseline_rate, runs_root, args.scenario, output_dir)
        else:
            print("No breakdown data found. Run generate_trajectories.py on the sweep runs first.")
