"""
Plot the transform sweep: noise/fuel metric boxplots and/or episode-outcome
breakdown for runs named transformed_{variant}_seed{NN}.

    python -m scripts.plot_transform_sweep runs/transforms --baseline <C4_run>
    python -m scripts.plot_transform_sweep runs/transforms --plots breakdown

--plots {both,metrics,breakdown} selects the views (default both). Only the
metrics view needs BlueSky + the noise/fuel metric fn.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.colors import *
from scripts.common.sweep_plotting import (
    REASON_LABELS,
    SUCCESS_REASON,
    boxplot_stats,
    draw_boxplot,
    find_csv,
    mean_breakdowns,
    collect_breakdown_data, add_reward, collect_baseline_metrics, collect_run_metrics,
    collect_baseline_breakdown, collect_baseline_seed_rates,
    run_sweep_args_parser,
)

BOX_WIDTH = 0.6
BAR_WIDTH = 0.7
DOT_SIZE = 60
DOT_ALPHA = 0.8
BAR_ALPHA = 0.6

# transformed_{variant}_seed{N}, e.g. transformed_power_flip_zoom_seed2
PATTERN = re.compile(r"^transformed_(?P<variant>.+)_seed(?P<seed>\d+)$")

# Caption per transform variant. Key order also defines the left-to-right plot order.
VARIANT_TO_CAPTION = {
    "baseline": "Baseline",
    "scale": "Scale",
    "power": "Power",
    "floor": "Floor",
    "zoom": "Zoom",
    "flip": "Flip",
    "flip_zoom": "Flip + Zoom",
    "power_flip": "Power + Flip",
    "power_zoom": "Power + Zoom",
    "power_flip_zoom": "Power + Flip + Zoom",
}

VARIANT_TO_COLOR = {
    "baseline": BASELINE_COLOR,
    "scale": TRANSFORMS_COLOR,
    "power":TRANSFORMS_COLOR,
    "floor":TRANSFORMS_COLOR,
    "zoom": TRANSFORMS_COLOR,
    "flip": TRANSFORMS_COLOR,
    "flip_zoom": TRANSFORMS_COLOR,
    "power_flip": TRANSFORMS_COLOR,
    "power_zoom": TRANSFORMS_COLOR,
    "power_flip_zoom": TRANSFORMS_COLOR,
}
VARIANT_ORDER = list(VARIANT_TO_CAPTION)

def _ordered_variants(present: set[str]) -> list[str]:
    """Captioned variants first (in caption order), then any extras alphabetically."""
    return [v for v in VARIANT_ORDER if v in present] + sorted(present - set(VARIANT_TO_CAPTION))


# ---------------------------------------------------------------------------- metrics

def plot_metric_boxplot(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    metric: str,
    ylabel: str,
    scenario: str,
    runs_name: str,
    output_dir: Path,
) -> list[dict]:
    variants = _ordered_variants(set(df["variant"].dropna().unique()))

    fig, ax = plt.subplots(figsize=(0.49 * TEXTWIDTH_IN, 0.49 * 0.78 * TEXTWIDTH_IN))
    legend_handles = []
    rows: list[dict] = []

    # Reference baseline box + quartile lines spanning the plot for comparison.
    has_baseline = baseline_df is not None and not baseline_df.empty
    if has_baseline:
        # draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
        # legend_handles.append(
        #     plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=BOXPLOT_ALPHA, label="Baseline (C4)")
        # )
        s = boxplot_stats(baseline_df[metric].values)
        rows.append({"variant": "baseline", "metric": metric, **s})
        # for val, ls in [(s["q50"], "--"), (s["q25"], ":"), (s["q75"], ":")]:
        #     ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=0.6)

    # One box per transform variant.
    for i, variant in enumerate(variants):
        data = df[df["variant"] == variant][metric].values
        if len(data) == 0:
            continue
        rows.append({"variant": variant, "metric": metric, **boxplot_stats(data)})
        draw_boxplot(ax, data, position=i, color=VARIANT_TO_COLOR[variant], box_width=BOX_WIDTH)

    tick_x = list(range(len(variants)))
    tick_labels = [
        VARIANT_TO_CAPTION.get(v, v) for v in variants
    ]
    ax.grid(axis='y')
    ax.set_xticks(tick_x)
    print(list(zip(tick_x, tick_labels)))
    ax.yaxis.set_inverted(METRIC_TO_AXIS_REVERS[metric])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Domain Randomization ID")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_{runs_name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)
    return rows


def plot_metrics(run_metrics, baseline_metrics, runs_root, scenario, output_dir):
    all_rows: list[dict] = []
    for metric, ylabel in METRICS:
        all_rows.extend(plot_metric_boxplot(
            run_metrics, baseline_metrics, metric, ylabel,
            scenario, runs_root.name, output_dir,
        ))
    csv_path = output_dir / f"boxplot_stats_{runs_root.name}_{scenario}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")


# --------------------------------------------------------------------------- breakdown

REASON_HATCH = {
    "success":         "",
    "failed_approach": "////",
    "max_steps":       "....",
    "out_of_bounds":   "xxxx",
}


def print_success_rates(breakdown: pd.DataFrame, baseline_seed_rates=None) -> None:
    if baseline_seed_rates:
        mean_bl = sum(baseline_seed_rates.values()) / len(baseline_seed_rates)
        print(f"  {'baseline':>18}  success_rate={mean_bl:.1%}  (seeds: {', '.join(f'{v:.1%}' for v in baseline_seed_rates.values())})")
    for variant in _ordered_variants(set(breakdown["variant"].dropna().unique())):
        rates = breakdown[breakdown["variant"] == variant]["success_rate"].values
        label = VARIANT_TO_CAPTION.get(variant, variant)
        print(f"  {label:>18}  success_rate={rates.mean():.1%}  (seeds: {', '.join(f'{r:.1%}' for r in rates)})")


def plot_episode_success(ax, df: pd.DataFrame, baseline_breakdown=None, baseline_seed_rates=None) -> None:
    variants = _ordered_variants(set(df["variant"].dropna().unique()))
    x = np.arange(len(variants))
    seen_reasons: set = set()

    def _bar(xi, h, bottom_, color, reason):
        hatch = REASON_HATCH.get(reason, "")
        ax.bar(xi, h, width=BAR_WIDTH, bottom=bottom_, color=color,
               alpha=BAR_ALPHA, hatch=hatch, edgecolor="black", linewidth=0.5)

    has_baseline = False # baseline_breakdown is not None
    x_offset = 0

    ordered, means = mean_breakdowns(df, variants, pos_col="variant")
    bottom = np.zeros(len(variants))
    for reason in ordered:
        for i, v in enumerate(variants):
            _bar(x[i] + x_offset, means[reason][i], bottom[i], VARIANT_TO_COLOR.get(v, BASELINE_COLOR), reason)
        bottom += means[reason]
        seen_reasons.add(reason)

    for i, v in enumerate(variants):
        seed_rates = {row["seed"]: row["success_rate"]
                      for _, row in df[df["variant"] == v].iterrows()}
        seeds = sorted(seed_rates)
        jitter = np.linspace(-0.06, 0.06, len(seeds))
        for jit, seed in zip(jitter, seeds):
            ax.scatter(x[i] + x_offset + jit, seed_rates[seed],
                       color="black", s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.8)

    tick_x = list(x + x_offset)
    tick_labels = (["Baseline\n(C4)"] if has_baseline else []) + [
        VARIANT_TO_CAPTION.get(v, v) for v in variants
    ]
    ax.set_xticks(tick_x)
    # ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Episode outcome fraction")
    ax.set_xlabel("Domain Randomization ID")
    ax.grid(axis="y")
    ax.set_ylim(0.90, 1.01)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    legend_handles = []
    if has_baseline:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=BAR_ALPHA, label="Baseline (C4)"))
    for reason in [r for r in REASON_HATCH if r in seen_reasons]:
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, fc="lightgray", hatch=REASON_HATCH[reason], edgecolor="black",
            label=REASON_LABELS.get(reason, reason)))
    legend_handles.append(plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="black",
        markersize=8, label="Per-seed success rate"))
    ax.legend(handles=legend_handles, frameon=True, edgecolor="k", loc="center left", bbox_to_anchor=(1, 0.5))


def plot_breakdown(breakdown, baseline_breakdown, baseline_seed_rates, runs_root, scenario, output_dir):
    print_success_rates(breakdown, baseline_seed_rates)
    fig, ax = plt.subplots(
        figsize=(TEXTWIDTH_IN, 0.4 * TEXTWIDTH_IN), constrained_layout=True
    )
    plot_episode_success(ax, breakdown, baseline_breakdown=baseline_breakdown, baseline_seed_rates=baseline_seed_rates)
    out_path = output_dir / f"episode_success_{runs_root.name}_{scenario}.pdf"
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
        from bluesky_gym.metrics.evaluation_metrics import build_metric_fn, make_pop_samplers

        from bluesky_gym.metrics.evaluation_metrics import bounds_from_df

        bs.init()
        all_csvs = [
            find_csv(run_dir, args.scenario)
            for run_dir in runs_root.iterdir()
            if PATTERN.search(run_dir.name)
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
            baseline_cache_path = runs_root / f"cached_baseline_metrics_{args.scenario}.csv"
            if args.cache and baseline_cache_path.exists():
                print("Using cached baseline metrics...")
                baseline_metrics = pd.read_csv(baseline_cache_path)
            else:
                baseline_metrics = collect_baseline_metrics(
                    list(args.baseline), args.scenario, calculate_metrics, args.mean_episode_length)
                if args.cache:
                    print(f"Saving baseline metrics to {baseline_cache_path} ...")
                    baseline_metrics.to_csv(baseline_cache_path, index=False)

        for frame in (run_metrics, baseline_metrics):
            if frame is not None and not frame.empty:
                frame["combined"] = frame["normalized_fuel"] + frame["normalized_noise"]
                add_reward(frame)

        plot_metrics(run_metrics, baseline_metrics, runs_root, args.scenario, output_dir)

    if "breakdown" in selected:
        breakdown = collect_breakdown_data(runs_root, PATTERN, args.scenario)
        if not breakdown.empty:
            baseline_breakdown = None
            baseline_seed_rates = {}
            if args.baseline:
                baseline_breakdown = collect_baseline_breakdown(args.baseline, args.scenario)
                baseline_seed_rates = collect_baseline_seed_rates(args.baseline, args.scenario)
                if baseline_breakdown is None:
                    print(f"Baseline — no usable trajectory data in {args.baseline} (plotting without baseline)")
            plot_breakdown(breakdown, baseline_breakdown, baseline_seed_rates, runs_root, args.scenario, output_dir)
        else:
            print("No breakdown data found. Run generate_trajectories.py on the sweep runs first.")
