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
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.colors import *
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

BOX_WIDTH = 0.6
BAR_WIDTH = 0.7
DOT_SIZE = 60
DOT_ALPHA = 0.8
BAR_ALPHA = 0.6

# Extracts config + optional seed; handles both "name_seed00" and bare "name" forms.
PATTERN = re.compile(r"^(?P<config>.+?)(?:_seed(?P<seed>\d+))?$")

METRICS = [
    ("fuel", "fuel [kg]"),
    ("noise", "noise [W·s]"),
    ("normalized_fuel", "normalized fuel"),
    ("normalized_noise", "normalized noise"),
    ("combined", "normalized fuel + noise"),
    ("reward", "reward"),
    ("reward_unclipped", "reward (no noise clipping)"),
]

METRIC_TO_AXIS_REVERS = {
    "fuel": True,
    "noise": True,
    "normalized_fuel": True,
    "normalized_noise": True,
    "combined": True,
    "reward": False,
    "reward_unclipped": False,
}

REASON_HATCH = {
    "success":         "",
    "failed_approach": "////",
    "max_steps":       "....",
    "out_of_bounds":   "xxxx",
}


def config_color(config: str):
    """Color a generalization config by the sweep it originates from.

    The generalization folder pools the single best run of each sweep, so each
    config is tinted with that sweep's semantic color (see scripts/common/colors).
    `transformed_baseline` is the no-transform reference, so it stays gray; the
    standalone best model (E_3_256-*) is highlighted to stand apart.
    """
    if "no_map" in config:
        return BASELINE_COLOR
    if "centered" in config:
        return CENTERED_COLOR
    if "forward" in config:
        return FORWARD_COLOR
    if "multi_scale" in config:
        return MULTI_SCALE_COLOR
    if "transformed_baseline" in config:
        return BASELINE_COLOR
    if "transformed" in config:
        return TRANSFORMS_COLOR
    if config.startswith("E_3_256"):
        return HIGHLIGHT_COLOR
    return BASELINE_COLOR


def _ordered_configs(present: set[str]) -> list[str]:
    return sorted(present)


def _ns(s):
    """Normalize a seed value: NaN (pandas None) → None for consistent dict keys."""
    return None if pd.isna(s) else s


def _seed_color_map(df: pd.DataFrame) -> dict:
    all_seeds = sorted({_ns(s) for s in df["seed"].unique()}, key=lambda s: s if s is not None else -1)
    return {seed: SEED_COLORS[i % len(SEED_COLORS)] for i, seed in enumerate(all_seeds)}


# ---------------------------------------------------------------------------- metrics

_KEEP_REASONS = {"success", "failed_approach"}


def _keep(df: pd.DataFrame) -> pd.Series:
    """True for episodes that completed (success or failed approach); excludes max_steps / out_of_bounds."""
    if "termination_reason" in df.columns:
        return df["termination_reason"].isin(_KEEP_REASONS)
    return df["success"]


def _matched_keep_ids(df: pd.DataFrame) -> set:
    """Scenario ids where _keep() is True for every (config, seed) combination.

    If a bearing fails _keep() in any one run it is excluded from all runs so
    every config is evaluated on an identical set of bearings.
    """
    if "scenario_id" not in df.columns:
        return set()
    keep_mask = _keep(df)
    per_scenario = keep_mask.groupby(df["scenario_id"]).all()
    return set(per_scenario.index[per_scenario])


def plot_metric_boxplot(run_df, baseline_df, metric, ylabel, scenario, runs_name, output_dir) -> list[dict]:
    configs = _ordered_configs(set(run_df["config"].dropna().unique()))
    has_baseline = baseline_df is not None and not baseline_df.empty

    matched_ids = _matched_keep_ids(run_df)

    fig, ax = plt.subplots(figsize=(0.49 * TEXTWIDTH_IN, 0.49 * 0.78 * TEXTWIDTH_IN))
    rows: list[dict] = []

    if has_baseline:
        bdf = baseline_df[_keep(baseline_df)]
        if matched_ids and "scenario_id" in bdf.columns:
            bdf = bdf[bdf["scenario_id"].isin(matched_ids)]
        bvals = bdf[metric].values
        if len(bvals):
            draw_boxplot(ax, bvals, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
            s = boxplot_stats(bvals)
            rows.append({"config": "baseline", "metric": metric, **s})
            for val, ls in [(s["q50"], "--"), (s["q25"], ":"), (s["q75"], ":")]:
                ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=BOXPLOT_ALPHA)

    offset = 1 if has_baseline else 0
    for i, config in enumerate(configs):
        sub = run_df[(run_df["config"] == config) & _keep(run_df)]
        if matched_ids and "scenario_id" in sub.columns:
            sub = sub[sub["scenario_id"].isin(matched_ids)]
        data = sub[metric].values
        if len(data):
            rows.append({"config": config, "metric": metric, **boxplot_stats(data)})
            draw_boxplot(ax, data, position=i + offset, color=config_color(config), box_width=BOX_WIDTH)

    tick_x = ([0] if has_baseline else []) + [i + offset for i in range(len(configs))]
    tick_labels = (["Baseline"] if has_baseline else []) + configs
    ax.grid(axis="y")
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=40, ha="right")
    ax.yaxis.set_inverted(METRIC_TO_AXIS_REVERS[metric])
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_{runs_name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)
    return rows


def plot_metric_delta_boxplot(run_df, baseline_df, metric, ylabel, scenario, runs_name, output_dir) -> None:
    if baseline_df is None or baseline_df.empty:
        return
    if "scenario_id" not in run_df.columns or "scenario_id" not in baseline_df.columns:
        print(f"  Skipping delta plot for {metric}: scenario_id not available")
        return

    matched_ids = _matched_keep_ids(run_df)
    ref = baseline_df.groupby("scenario_id")[metric].mean()
    configs = _ordered_configs(set(run_df["config"].dropna().unique()))

    fig, ax = plt.subplots(figsize=(0.49 * TEXTWIDTH_IN, 0.49 * 0.78 * TEXTWIDTH_IN))
    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.0,
               label="Baseline = 0", zorder=1)

    for i, config in enumerate(configs):
        sub = run_df[run_df["config"] == config]
        if matched_ids:
            sub = sub[sub["scenario_id"].isin(matched_ids)]
        deltas = sub[metric].values - sub["scenario_id"].map(ref).to_numpy(dtype=float)
        deltas = deltas[~np.isnan(deltas)]
        if len(deltas):
            draw_boxplot(ax, deltas, position=i, color=config_color(config), box_width=BOX_WIDTH)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel(f"Δ {ylabel}\nvs baseline")
    ax.grid(axis="y")
    ax.legend(handles=[plt.Line2D([0], [0], color=BASELINE_COLOR, linestyle="--", label="Baseline = 0")],
              frameon=False)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"delta_{metric}_{runs_name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def plot_metrics(run_metrics, baseline_metrics, runs_root, scenario, output_dir) -> None:
    has_baseline = baseline_metrics is not None and not baseline_metrics.empty
    # NaN seeds (runs without a seed suffix) are dropped by groupby; use -1 as sentinel.
    run_metrics["seed"] = run_metrics["seed"].fillna(-1)
    run_ok = add_scenario_id(run_metrics, ["config", "seed"])
    if has_baseline:
        add_scenario_id(baseline_metrics, ["seed"])

    all_rows: list[dict] = []
    for metric, ylabel in METRICS:
        all_rows.extend(plot_metric_boxplot(run_metrics, baseline_metrics, metric, ylabel,
                                            scenario, runs_root.name, output_dir))
        if run_ok and has_baseline:
            plot_metric_delta_boxplot(run_metrics, baseline_metrics, metric, ylabel,
                                      scenario, runs_root.name, output_dir)
    csv_path = output_dir / f"boxplot_stats_{runs_root.name}_{scenario}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")


# --------------------------------------------------------------------------- breakdown

def plot_episode_success(ax, df: pd.DataFrame, baseline: float | None = None) -> None:
    configs = _ordered_configs(set(df["config"].dropna().unique()))
    x = np.arange(len(configs))
    seen_reasons: set = set()

    def _bar(xi, h, bottom_, color, reason):
        hatch = REASON_HATCH.get(reason, "")
        ax.bar(xi, h, width=BAR_WIDTH, bottom=bottom_, color=color,
               alpha=BAR_ALPHA, hatch=hatch, edgecolor="black", linewidth=0.5)

    # Each config keeps its sweep color; the termination reason is shown by hatch.
    ordered, means = mean_breakdowns(df, configs, pos_col="config")
    bottom = np.zeros(len(configs))
    for reason in ordered:
        for i, config in enumerate(configs):
            _bar(x[i], means[reason][i], bottom[i], config_color(config), reason)
        bottom += means[reason]
        seen_reasons.add(reason)

    for i, config in enumerate(configs):
        seed_rates = {_ns(row["seed"]): row["success_rate"] for _, row in df[df["config"] == config].iterrows()}
        seeds = sorted(seed_rates, key=lambda s: s if s is not None else -1)
        jitter = np.linspace(-0.08, 0.08, len(seeds))
        for xi, seed in zip(jitter, seeds):
            ax.scatter(x[i] + xi, seed_rates[seed],
                       color="black", s=DOT_SIZE, zorder=5,
                       alpha=DOT_ALPHA, edgecolors="white", linewidths=0.8)

    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline success ({baseline:.0%})", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel("Episode outcome fraction")
    ax.grid(axis="y")
    ax.set_ylim(0.8, 1.01)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    legend_handles = []
    if baseline is not None:
        legend_handles.append(plt.Line2D([0], [0], color=BASELINE_COLOR, linestyle="--",
                                         label=f"Baseline success ({baseline:.0%})"))
    for reason in [r for r in REASON_HATCH if r in seen_reasons]:
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, fc="lightgray", hatch=REASON_HATCH[reason], edgecolor="black",
            label=REASON_LABELS.get(reason, reason)))
    legend_handles.append(plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="black",
        markersize=8, label="Per-seed success rate"))
    ax.legend(handles=legend_handles, frameon=True, edgecolor="k", loc="center left", bbox_to_anchor=(1, 0.5))


def plot_episode_length(ax, df: pd.DataFrame, baseline: float | None = None) -> None:
    configs = _ordered_configs(set(df["config"].dropna().unique()))
    x = np.arange(len(configs))
    seed_colors = _seed_color_map(df)

    for i, config in enumerate(configs):
        c_df = df[df["config"] == config]
        all_lengths = []
        seeds = sorted((_ns(row["seed"]) for _, row in c_df.iterrows() if row["length"] is not None),
                       key=lambda s: s if s is not None else -1)
        slot_width = BAR_WIDTH / max(len(seeds), 1)
        seed_centers = {s: x[i] - BAR_WIDTH / 2 + (j + 0.5) * slot_width for j, s in enumerate(seeds)}
        for _, row in c_df.iterrows():
            if row["length"] is None:
                continue
            lengths = row["length"].values
            all_lengths.extend(lengths)
            seed = _ns(row["seed"])
            rng_seed = int(seed) if seed is not None else i
            jitter = np.random.default_rng(rng_seed).uniform(
                -slot_width * 0.35, slot_width * 0.35, len(lengths))
            ax.scatter(seed_centers[seed] + jitter, lengths,
                       color=seed_colors.get(seed, config_color(config)), s=DOT_SIZE * 0.5,
                       zorder=5, alpha=DOT_ALPHA, edgecolors="none")
        if all_lengths:
            ax.bar(x[i], np.mean(all_lengths), width=BAR_WIDTH, color=config_color(config), alpha=BAR_ALPHA)

    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline ({baseline:.0f} s)", zorder=3)
        ax.legend(frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel("Mean episode length (s)")
    ax.grid(axis="y")

    seed_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8,
                   label=f"Seed {s}" if pd.notna(s) else "Single run")
        for s, c in seed_colors.items()
    ]
    ax.legend(handles=seed_handles, frameon=False, fontsize=8, title="Seed", loc="upper right")


def plot_breakdown(breakdown, baseline_rate, baseline_length, runs_root, scenario, output_dir):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 0.4 * TEXTWIDTH_IN), constrained_layout=True)
    plot_episode_success(ax, breakdown, baseline=baseline_rate)
    out_path = output_dir / f"episode_success_{runs_root.name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 0.4 * TEXTWIDTH_IN), constrained_layout=True)
    plot_episode_length(ax, breakdown, baseline=baseline_length)
    out_path = output_dir / f"episode_length_{runs_root.name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


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

        baseline_metrics = None
        if args.baseline:
            baseline_metrics = collect_baseline_metrics(
                list(args.baseline), args.scenario, calculate_metrics, args.mean_episode_length)

        for frame in (run_metrics, baseline_metrics):
            if frame is not None and not frame.empty:
                frame["combined"] = frame["normalized_fuel"] + frame["normalized_noise"]
                add_reward(frame)

        plot_metrics(run_metrics, baseline_metrics, runs_root, args.scenario, output_dir)

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
            plot_breakdown(breakdown, baseline_rate, baseline_length, runs_root, args.scenario, output_dir)
        else:
            print("No breakdown data found. Run generate_trajectories.py on the sweep runs first.")