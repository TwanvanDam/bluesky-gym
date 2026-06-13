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

from scripts.common.colors import (
    BASELINE_COLOR,
    FALLBACK_REASON_COLOR,
    REASON_COLORS,
    qual,
)
from scripts.common.sweep_plotting import (
    REASON_LABELS,
    SUCCESS_REASON,
    draw_boxplot,
    mean_breakdowns,
    seed_color_map, compute_baseline, collect_breakdown_data, add_reward, collect_baseline_metrics, collect_run_metrics,
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
    "scale": "Scale [1, 7.6]",
    "power": "Power [0.52, 0.70]",
    "floor": "Floor [0, 40.2]",
    "zoom": "Zoom [1x - 2x]",
    "flip": "Flip",
    "flip_zoom": "Flip + Zoom",
    "power_flip": "Power + Flip",
    "power_zoom": "Power + Zoom",
    "power_flip_zoom": "Power + Flip + Zoom",
}
VARIANT_ORDER = list(VARIANT_TO_CAPTION)

METRICS = [
    ("fuel", "fuel [kg]"),
    ("noise", "noise [W·s]"),
    ("normalized_fuel", "normalized fuel"),
    ("normalized_noise", "normalized noise"),
    ("combined", "normalized fuel + noise"),
    ("reward", "reward"),
]


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
) -> None:
    variants = _ordered_variants(set(df["variant"].dropna().unique()))

    fig, ax = plt.subplots(figsize=(14, 5))
    legend_handles = []

    # Reference baseline box + quartile lines spanning the plot for comparison.
    has_baseline = baseline_df is not None and not baseline_df.empty
    if has_baseline:
        draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=0.6, label="Baseline (C4)")
        )
        q1, median, q3 = baseline_df[metric].quantile([0.25, 0.5, 0.75])
        for val, ls in [(median, "--"), (q1, ":"), (q3, ":")]:
            ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=0.6)

    # One box per transform variant.
    for i, variant in enumerate(variants):
        data = df[df["variant"] == variant][metric].values
        if len(data) == 0:
            continue
        draw_boxplot(ax, data, position=i + 1, color=qual(i), box_width=BOX_WIDTH)

    tick_x = ([0] if has_baseline else []) + [i + 1 for i in range(len(variants))]
    tick_labels = (["Baseline\n(C4)"] if has_baseline else []) + [
        VARIANT_TO_CAPTION.get(v, v) for v in variants
    ]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
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
    plt.close(fig)


def plot_metrics(run_metrics, baseline_metrics, runs_root, scenario, output_dir):
    for metric, ylabel in METRICS:
        plot_metric_boxplot(
            run_metrics, baseline_metrics, metric, ylabel,
            scenario, runs_root.name, output_dir,
        )


# --------------------------------------------------------------------------- breakdown

def plot_episode_success(ax, df: pd.DataFrame, baseline: float | None = None) -> None:
    variants = _ordered_variants(set(df["variant"].dropna().unique()))
    x = np.arange(len(variants))
    seed_colors = seed_color_map(df)

    ordered, means = mean_breakdowns(df, variants, pos_col="variant")
    bottom = np.zeros(len(variants))
    for reason in ordered:
        if reason == SUCCESS_REASON:
            bar_colors = [qual(i) for i in range(len(variants))]
        else:
            bar_colors = [REASON_COLORS.get(reason, FALLBACK_REASON_COLOR)] * len(variants)
        ax.bar(x, means[reason], width=BAR_WIDTH, bottom=bottom,
               color=bar_colors, alpha=BAR_ALPHA, label=REASON_LABELS.get(reason, reason))
        bottom += means[reason]

    for i, v in enumerate(variants):
        seed_rates = {row["seed"]: row["success_rate"] for _, row in df[df["variant"] == v].iterrows()}
        seeds = sorted(seed_rates)
        jitter = np.linspace(-0.08, 0.08, len(seeds))
        for xi, seed in zip(jitter, seeds):
            ax.scatter(x[i] + xi, seed_rates[seed],
                       color=seed_colors[seed], s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.8)

    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline success ({baseline:.0%})", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_TO_CAPTION.get(v, v) for v in variants], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Episode outcome fraction")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    outcome_handles, outcome_labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(outcome_handles, outcome_labels, frameon=False, fontsize=8,
                     title="Episode outcome", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    seed_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=f"Seed {s}")
        for s, c in seed_colors.items()
    ]
    ax.legend(handles=seed_handles, frameon=False, fontsize=8,
              title="Seed", loc="lower left", bbox_to_anchor=(1.01, 0.0))


def plot_episode_length(ax, df: pd.DataFrame, baseline: float | None = None) -> None:
    variants = _ordered_variants(set(df["variant"].dropna().unique()))
    x = np.arange(len(variants))
    seed_colors = seed_color_map(df)

    for i, v in enumerate(variants):
        v_df = df[df["variant"] == v]
        all_lengths = []
        seeds = sorted(row["seed"] for _, row in v_df.iterrows() if row["length"] is not None)
        slot_width = BAR_WIDTH / max(len(seeds), 1)
        seed_centers = {s: x[i] - BAR_WIDTH / 2 + (j + 0.5) * slot_width for j, s in enumerate(seeds)}
        for _, row in v_df.iterrows():
            if row["length"] is None:
                continue
            lengths = row["length"].values
            all_lengths.extend(lengths)
            jitter = np.random.default_rng(row["seed"]).uniform(-slot_width * 0.35, slot_width * 0.35, len(lengths))
            ax.scatter(seed_centers[row["seed"]] + jitter, lengths,
                       color=seed_colors[row["seed"]], s=DOT_SIZE * 0.5, zorder=5,
                       alpha=DOT_ALPHA, edgecolors="none")
        if all_lengths:
            ax.bar(x[i], np.mean(all_lengths), width=BAR_WIDTH, color=qual(i), alpha=BAR_ALPHA)

    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline ({baseline:.0f} s)", zorder=3)
        ax.legend(frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_TO_CAPTION.get(v, v) for v in variants], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Mean episode length (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    seed_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=f"Seed {s}")
        for s, c in seed_colors.items()
    ]
    ax.legend(handles=seed_handles, frameon=False, fontsize=8, title="Seed", loc="upper right")


def plot_breakdown(breakdown, baseline_rate, baseline_length, runs_root, scenario, output_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_episode_success(ax, breakdown, baseline=baseline_rate)
    fig.tight_layout()
    out_path = output_dir / f"episode_success_{runs_root.name}_{scenario}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_episode_length(ax, breakdown, baseline=baseline_length)
    fig.tight_layout()
    out_path = output_dir / f"episode_length_{runs_root.name}_{scenario}.png"
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

        bs.init()
        # Fixed-map overview: legacy TiffMapSource branch ignores bounds and is shared
        # across all sweep runs (post-resample clip at the given percentile).
        samplers = make_pop_samplers(
            TransformedTiffMapSourceConfig(file_path=args.map_path), bounds=None,
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
                baseline_rate, baseline_length = compute_baseline(args.baseline[0], args.scenario)
                if baseline_rate is None:
                    print(f"Baseline — no usable trajectory data in {args.baseline[0]} (plotting without baseline)")
                else:
                    print(f"Baseline — success rate: {baseline_rate:.1%}, mean length: {baseline_length:.1f} s")
            plot_breakdown(breakdown, baseline_rate, baseline_length, runs_root, args.scenario, output_dir)
        else:
            print("No breakdown data found. Run generate_trajectories.py on the sweep runs first.")
