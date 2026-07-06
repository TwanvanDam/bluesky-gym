"""
Generate a LaTeX means table (siunitx `S`-column tabular) from a per-episode
metrics CSV, for inclusion in the thesis paper.

    python -m scripts.create_means_table <csv_path> -o <tex_path>
    python -m scripts.create_means_table <csv_path> -o <tex_path> \
        --group-by mode,resolution --baseline <baseline_csv> --fuel-weight 0.5

The input CSV is one produced by the sweep-plotting caches
(cached_metrics_<scenario>.csv): the metric columns
fuel/noise/normalized_fuel/normalized_noise/success plus one or more
config-identifying columns (e.g. mode, resolution) and a seed column.

Per config group the table reports the reward mean/median, cost mean, normalized
fuel/noise means and the success rate. Only the `\\begin{tabular} ... \\end{tabular}`
block is written; the surrounding table/caption/label live in the paper .tex.
"""

import argparse
from pathlib import Path

import pandas as pd

# Metric columns written by compute_episode_metrics. Everything else that is not
# the seed column is treated as a config-identifying column (the group-by keys).
METRIC_COLS = {
    "fuel", "noise", "noise_clipped",
    "normalized_fuel", "normalized_noise", "normalized_noise_clipped",
    "success",
}
SEED_COL = "seed"

# Reward shaping constants, matching the training reward (see add_reward).
SUCCESS_BONUS = 5.0
FAILURE_PENALTY = -1.0

BASELINE_LABEL = "No-map"

# (aggregate key, number of decimals, "max"|"min" for best-value bolding). Order
# is the left-to-right column order of the table body.
VALUE_COLUMNS = [
    ("reward_mean", 3, "max"),
    ("reward_median", 3, "max"),
    ("cost_mean", 3, "min"),
    ("fuel_mean", 3, "min"),
    ("noise_mean", 3, "min"),
    ("success_pct", 1, "max"),
]

HEADER = r"""\begin{tabular}{l S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=3.1]}
	\hline
	\textbf{Configuration} & {\textbf{Reward}} & {\textbf{Reward}} & {\textbf{Cost}} & {\textbf{Fuel}} & {\textbf{Noise}} & {\textbf{Success}} \\
	 & {\textbf{(mean)}} & {\textbf{(median)}} & {\textbf{(mean)}} & {\textbf{(mean)}} & {\textbf{(mean)}} & {[\unit{\percent}]} \\ \hline"""


def add_derived(df: pd.DataFrame, fuel_weight: float) -> None:
    """Add per-episode `reward` and `cost` columns in place.

    Reward = (+5 if success else -1) with normalized fuel and noise entering as
    costs; cost is just the weighted normalized fuel/noise sum. Uses the
    unclipped normalized_noise so the table matches the sweep summary CSVs.
    """
    w = fuel_weight
    success_term = df["success"].map({True: SUCCESS_BONUS, False: FAILURE_PENALTY})
    df["reward"] = success_term - w * df["normalized_fuel"] - (1 - w) * df["normalized_noise"]
    df["cost"] = w * df["normalized_fuel"] + (1 - w) * df["normalized_noise"]


def auto_group_by(df: pd.DataFrame) -> list[str]:
    """Config columns to group on: everything that is not a metric or the seed."""
    return [c for c in df.columns if c not in METRIC_COLS and c != SEED_COL]


def aggregate(df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """Per config-group summary statistics, sorted by the group keys.

    Returns a frame with the group-by columns plus reward_mean, reward_median,
    cost_mean, fuel_mean, noise_mean, success_pct (success as a percentage).
    """
    grouped = df.groupby(group_by, sort=True)
    agg = grouped.agg(
        reward_mean=("reward", "mean"),
        reward_median=("reward", "median"),
        cost_mean=("cost", "mean"),
        fuel_mean=("normalized_fuel", "mean"),
        noise_mean=("normalized_noise", "mean"),
        success_pct=("success", "mean"),
    ).reset_index()
    agg["success_pct"] *= 100.0
    return agg


def _fmt_key(value) -> str:
    """Render a config-key value: integral floats as ints, everything else as-is."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def make_label(values: list, group_by: list[str]) -> str:
    """Row label from the group-key values (e.g. "Centered 4", "1 a").

    Joins the values with a space and capitalizes the first letter. For
    mode+resolution this yields the "Centered 4" / "Forward 16" style; for other
    groupings it capitalizes the first character of the joined string.
    """
    joined = " ".join(_fmt_key(v) for v in values)
    return joined[:1].upper() + joined[1:]


def _best_mask(rounded: list[float], direction: str) -> list[bool]:
    """Boolean mask marking every row whose rounded value equals the best value."""
    target = max(rounded) if direction == "max" else min(rounded)
    return [r == target for r in rounded]


def build_rows(agg: pd.DataFrame, group_by: list[str],
               baseline: pd.DataFrame | None) -> list[dict]:
    """Assemble ordered table rows with labels, block ids and raw values.

    The optional baseline is prepended as its own block labeled "No-map". Each
    row dict carries `label`, `block` (first group-by value, or a sentinel for
    the baseline) and one raw value per VALUE_COLUMNS key.
    """
    rows: list[dict] = []
    if baseline is not None:
        b = baseline.iloc[0]
        rows.append({
            "label": BASELINE_LABEL,
            "block": "__baseline__",
            **{key: float(b[key]) for key, _, _ in VALUE_COLUMNS},
        })
    for _, r in agg.iterrows():
        rows.append({
            "label": make_label([r[c] for c in group_by], group_by),
            "block": _fmt_key(r[group_by[0]]),
            **{key: float(r[key]) for key, _, _ in VALUE_COLUMNS},
        })
    return rows


def format_cells(rows: list[dict]) -> list[list[str]]:
    """Turn raw rows into text cells, bolding the best (rounded) value per column.

    Bolding compares on the rounded, displayed value so ties are all bolded and
    the emphasis matches what the reader sees. Returns one list of cell strings
    (label + value cells) per row.
    """
    cells = [[row["label"]] for row in rows]
    for key, nd, direction in VALUE_COLUMNS:
        rounded = [round(row[key], nd) for row in rows]
        best = _best_mask(rounded, direction)
        for i, value in enumerate(rounded):
            disp = f"{value:.{nd}f}"
            cells[i].append(r"\textbf{" + disp + r"}" if best[i] else disp )
    return cells


def render(rows: list[dict], cells: list[list[str]]) -> str:
    """Render the padded, block-separated tabular body under the fixed header."""
    widths = [max(len(row[col]) for row in cells) for col in range(len(cells[0]))]
    lines = [HEADER]
    for i, (row, cell) in enumerate(zip(rows, cells)):
        padded = [c.ljust(widths[j]) for j, c in enumerate(cell)]
        last_in_block = i == len(rows) - 1 or rows[i + 1]["block"] != row["block"]
        suffix = r" \\ \hline" if last_in_block else r" \\"
        lines.append("\t" + " & ".join(padded) + suffix)
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def create_table(csv_path: Path, group_by: list[str] | None,
                 baseline_path: Path | None, fuel_weight: float) -> str:
    """Build the full tabular string for the given metrics CSV."""
    df = pd.read_csv(csv_path)
    if group_by is None:
        group_by = auto_group_by(df)
    add_derived(df, fuel_weight)
    if not group_by:
        raise ValueError(f"No config columns found in {csv_path}; pass --group-by explicitly.")

    agg = aggregate(df, group_by)

    baseline_agg = None
    if baseline_path is not None:
        baseline_df = pd.read_csv(baseline_path)
        add_derived(baseline_df, fuel_weight)
        baseline_df["_all"] = 0  # single pooled group → one "No-map" row
        baseline_agg = aggregate(baseline_df, ["_all"])

    rows = build_rows(agg, group_by, baseline_agg)
    cells = format_cells(rows)
    return render(rows, cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX means table from a per-episode metrics CSV.")
    parser.add_argument("csv_path", type=Path,
                        help="input per-episode metrics CSV")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="path to write the .tex tabular block")
    parser.add_argument("--group-by", type=str, default=None,
                        help="comma-separated config columns (default: auto-detect "
                             "the non-metric, non-seed columns, e.g. mode,resolution)")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="optional baseline CSV, appended as a first 'No-map' row")
    parser.add_argument("--fuel-weight", type=float, default=0.5,
                        help="weight w on normalized fuel in reward/cost (default: 0.5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_by = args.group_by.split(",") if args.group_by else None
    table = create_table(args.csv_path, group_by, args.baseline, args.fuel_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"Wrote → {args.output}")
    print(table)


if __name__ == "__main__":
    main()
