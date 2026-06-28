"""
Plot the multi-scale observation sweep: noise/fuel metric boxplots and/or
episode-outcome breakdown for runs named multi_scale_{group}{variant}_seed{NN}
(group 1-5, variant a/b).

    python -m scripts.plot_multi_scale_sweep <runs_root> --baseline <C4_run>
    python -m scripts.plot_multi_scale_sweep <runs_root> --plots breakdown

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
    mean_breakdowns,
    seed_color_map, collect_breakdown_data, add_reward, collect_baseline_metrics, collect_run_metrics,
    collect_baseline_breakdown, collect_baseline_seed_rates,
    run_sweep_args_parser,
)

BOX_WIDTH = 0.35
BOX_OFFSET = 0.2  # half-gap between the two variants in a group
BAR_WIDTH = 0.6
DOT_ALPHA = 0.8
DOT_SIZE = 60

# {multi_scale_}{group}{variant}_seed{NN}, group 1-5, variant a/b
PATTERN = re.compile(r"^(?:multi_scale_)?(?P<group_num>\d)(?P<variant>[ab])_seed(?P<seed>\d+)$")

VARIANT_TO_OBSERVATION = {
    "1a": "C2 + C4",
    "1b": "F2 + C4",
    "2a": "C4 + C8",
    "2b": "C4 + F8",
    "3a": "C4 + C16",
    "3b": "C4 + F16",
    "4a": "C2 + C4 + C8",
    "4b": "C2 + C4 + C16",
    "5a": "C2 + C16",
    "5b": "C2 + C8",
}

VARIANT_TO_ALPHA = {
    "a": BOXPLOT_ALPHA,
    "b": BOXPLOT_ALPHA_LIGHT,
}

REASON_HATCH = {
    "success":         "",
    "failed_approach": "////",
    "max_steps":       "....",
    "out_of_bounds":   "xxxx",
}

METRICS = [
    ("fuel", "fuel [kg]"),
    ("noise", "noise [W·s]"),
    ("normalized_fuel", "normalized fuel"),
    ("normalized_noise", "normalized noise"),
    ("combined", "normalized fuel + noise"),
    ("reward", "reward"),
    ("reward_unclipped", "reward (no noise clipping"),
]

def _config_ids(df: pd.DataFrame) -> list[str]:
    return sorted(df["config_id"].dropna().unique(), key=lambda c: (int(c[:-1]), c[-1]))


def _add_config_id(df: pd.DataFrame) -> pd.DataFrame:
    """config_id is the composite of the group_num / variant named groups."""
    df["config_id"] = df["group_num"].astype(str) + df["variant"]
    return df


def _group_tick_labels(groups: list[int]) -> list[str]:
    return [str(g) for g in groups]


# ---------------------------------------------------------------------------- metrics

def _config_x_positions(config_ids: list[str]) -> dict[str, float]:
    """Place variant 'a' at group_num - BOX_OFFSET, variant 'b' at group_num + BOX_OFFSET."""
    positions = {}
    for cid in config_ids:
        group = int(cid[:-1])
        offset = -BOX_OFFSET if cid[-1] == "a" else +BOX_OFFSET
        positions[cid] = group + offset
    return positions


def plot_metric_boxplot(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    metric: str,
    ylabel: str,
    scenario: str,
    runs_name: str,
    output_dir: Path,
) -> list[dict]:
    config_ids = _config_ids(df)
    x_pos = _config_x_positions(config_ids)

    fig, ax = plt.subplots(figsize=(0.49 * TEXTWIDTH_IN, 0.49 * 0.78 * TEXTWIDTH_IN))
    legend_handles = []
    rows: list[dict] = []

    # Baseline box + reference lines
    if baseline_df is not None and not baseline_df.empty:
        draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR, box_width=BOX_WIDTH)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=BOXPLOT_ALPHA, label="Baseline")
        )
        s = boxplot_stats(baseline_df[metric].values)
        print(f"  {'baseline':>8}  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
        rows.append({"config_id": "baseline", "metric": metric, **s})
        for val, ls in [(s["q50"], "--"), (s["q25"], ":"), (s["q75"], ":")]:
            ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=BOXPLOT_ALPHA)

    # Per-config boxes
    for cid in config_ids:
        xp = x_pos[cid]
        data = df[df["config_id"] == cid][metric].values
        if len(data) == 0:
            continue
        s = boxplot_stats(data)
        print(f"  {cid:>8}  {metric:<22}  Q1={s['q25']:8.3f}  median={s['q50']:8.3f}  Q3={s['q75']:8.3f}")
        rows.append({"config_id": cid, "metric": metric, **s})
        draw_boxplot(ax, data, position=xp, color=MULTI_SCALE_COLOR, box_width=BOX_WIDTH, alpha=VARIANT_TO_ALPHA[cid[-1]])

    groups = sorted({int(cid[:-1]) for cid in config_ids})
    ax.grid(axis="y")
    ax.set_xticks([0] + groups)
    ax.set_xticklabels(["C4"] + _group_tick_labels(groups))
    ax.set_xlabel("Observation configuration group")
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_{runs_name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)
    return rows


def save_legend(output_dir: Path, runs_name: str, scenario: str) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=BOXPLOT_ALPHA, label="Baseline (C4)"),
        plt.Rectangle((0, 0), 1, 1, fc=MULTI_SCALE_COLOR, alpha=BOXPLOT_ALPHA, label="Variant a"),
        plt.Rectangle((0, 0), 1, 1, fc=MULTI_SCALE_COLOR, alpha=BOXPLOT_ALPHA_LIGHT, label="Variant b"),
    ]
    fig = plt.figure(figsize=(1.5, 0.49 * TEXTWIDTH_IN * 0.6),constrained_layout=True)
    legend = fig.legend(handles=handles, loc="center left", ncol=1)
    legend.get_frame().set_edgecolor("k")
    out_path = output_dir / f"legend_modes_{runs_name}_{scenario}.pdf"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


def plot_metrics(run_metrics, baseline_metrics, runs_root, scenario, output_dir):
    _add_config_id(run_metrics)
    all_rows: list[dict] = []
    for metric, ylabel in METRICS:
        all_rows.extend(plot_metric_boxplot(
            run_metrics, baseline_metrics, metric, ylabel,
            scenario, runs_root.name, output_dir,
        ))
    csv_path = output_dir / f"boxplot_stats_{runs_root.name}_{scenario}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")
    save_legend(output_dir, runs_root.name, scenario)


# --------------------------------------------------------------------------- breakdown

VARIANT_OFFSET = {"a": -BOX_OFFSET, "b": +BOX_OFFSET}


def plot_breakdown(breakdown, baseline_breakdown, baseline_seed_rates, runs_root, scenario, output_dir):
    _add_config_id(breakdown)

    config_ids = _config_ids(breakdown)
    groups = sorted({int(cid[:-1]) for cid in config_ids})

    fig, ax = plt.subplots(figsize=(0.85 * TEXTWIDTH_IN, 0.4* 0.85 * TEXTWIDTH_IN), constrained_layout=True)
    seen_reasons: set = set()

    def _bar(x_, h, bottom_, color, reason, alpha):
        hatch = REASON_HATCH.get(reason, "")
        ax.bar(x_, h, width=BOX_WIDTH, bottom=bottom_, color=color,
               alpha=alpha, hatch=hatch, edgecolor="black", linewidth=0.5)

    # --- baseline bar at x=0 ---
    if baseline_breakdown is not None:
        bottom = 0.0
        for reason in [SUCCESS_REASON] + [r for r in baseline_breakdown.index if r != SUCCESS_REASON]:
            frac = baseline_breakdown.get(reason, 0.0)
            if frac <= 0:
                continue
            _bar(0, frac, bottom, BASELINE_COLOR, reason, BOXPLOT_ALPHA)
            bottom += frac
            seen_reasons.add(reason)

    if baseline_seed_rates:
        seeds = sorted(baseline_seed_rates)
        jitter = np.linspace(-0.06, 0.06, len(seeds))
        for jit, seed in zip(jitter, seeds):
            ax.scatter(jit, baseline_seed_rates[seed],
                       color="black", s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                       edgecolors="white", linewidths=0.5)

    # --- per-variant stacked bars ---
    for variant in ("a", "b"):
        variant_cids = [c for c in config_ids if c.endswith(variant)]
        if not variant_cids:
            continue
        variant_df = breakdown[breakdown["config_id"].isin(variant_cids)]
        ordered, means = mean_breakdowns(variant_df, variant_cids, pos_col="config_id")
        xi = np.array([int(cid[:-1]) for cid in variant_cids])
        bottom = np.zeros(len(variant_cids))
        alpha = VARIANT_TO_ALPHA[variant]
        for reason in ordered:
            _bar(xi + VARIANT_OFFSET[variant], means[reason], bottom, MULTI_SCALE_COLOR, reason, alpha)
            bottom += means[reason]
            seen_reasons.add(reason)

        for cid in variant_cids:
            xi_base = int(cid[:-1]) + VARIANT_OFFSET[variant]
            seed_rates = {row["seed"]: row["success_rate"]
                          for _, row in breakdown[breakdown["config_id"] == cid].iterrows()}
            seeds = sorted(seed_rates)
            jitter = np.linspace(-0.06, 0.06, len(seeds))
            for jit, seed in zip(jitter, seeds):
                ax.scatter(xi_base + jit, seed_rates[seed],
                           color="black", s=DOT_SIZE, zorder=5, alpha=DOT_ALPHA,
                           edgecolors="white", linewidths=0.8)

    ax.set_xticks([0] + groups)
    ax.set_xticklabels(["C4"] + _group_tick_labels(groups))
    ax.set_xlabel("Observation configuration group")
    ax.set_ylabel("Episode outcome fraction")
    ax.grid(axis="y")
    ax.set_ylim(0.87, 1.01)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    legend_handles = []
    if baseline_breakdown is not None:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=BOXPLOT_ALPHA, label="Baseline (C4)"))
    for variant in ("a", "b"):
        if any(c.endswith(variant) for c in config_ids):
            legend_handles.append(plt.Rectangle(
                (0, 0), 1, 1, fc=MULTI_SCALE_COLOR, alpha=VARIANT_TO_ALPHA[variant], label=f"Variant {variant}"))
    for reason in [r for r in REASON_HATCH if r in seen_reasons]:
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, fc="lightgray", hatch=REASON_HATCH[reason], edgecolor="black",
            label=REASON_LABELS.get(reason, reason)))
    legend_handles.append(plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="black",
        markersize=8, label="Per-seed success rate"))
    ax.legend(handles=legend_handles, frameon=True, edgecolor="k", loc="center left", bbox_to_anchor=(1, 0.5))

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
        from bluesky_gym.maps.map_sources import TiffMapSourceConfig
        from bluesky_gym.metrics.evaluation_metrics import build_metric_fn, make_pop_samplers

        bs.init()
        # Fixed-map overview: legacy TiffMapSource branch ignores bounds and is shared
        # across all sweep runs (post-resample clip at the given percentile).
        samplers = make_pop_samplers(
            TiffMapSourceConfig(file_path=args.map_path), bounds=None,
            clip_percentile=args.noise_clip_percentile, train_resampling="cubic_spline", true_resampling="average")
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
