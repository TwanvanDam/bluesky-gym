import argparse
from pathlib import Path

import pandas as pd
from bluesky_gym.maps.map_sources import TiffMapSourceConfig
from bluesky_gym.metrics.evaluation_metrics import build_metric_fn, make_pop_samplers, bounds_from_df
from scripts.common.run_paths import resolve_run, RunPaths, load_trajectory_details
from scripts.config import ExperimentConfig

# Fallback map for the legacy bare-CSV path (no run config to read from).
DEFAULT_MAP = Path("./scripts/population_maps/europe_3035.tif")

def plot_noise_bar_chart(df: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt
    noise_by_angle = df.groupby("start_angle")["calculated_noise"].sum()
    noise_by_angle.plot(kind="bar")
    plt.title("Total Noise by Start Angle")
    plt.xlabel("Start Angle (degrees)")
    plt.ylabel("Total Noise")
    plt.xticks(rotation=45)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fuel_bar_chart(df: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt
    fuel_by_angle = df.groupby("start_angle")["calculated_fuel"].sum()
    fuel_by_angle.plot(kind="bar")
    plt.title("Total Fuel Consumption by Start Angle")
    plt.xlabel("Start Angle (degrees)")
    plt.ylabel("Total Fuel Consumption (kg)")
    plt.xticks(rotation=45)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _build_calculate_metrics(df: pd.DataFrame, traj_dir: Path, population_config):
    """Build the metric fn for this trajectory subdir.

    Reproduces the eval map the agent flew over: training map pipeline with the file
    swapped to the subdir's eval map, clip percentile and resampling read from config.
    Falls back to a default map when no run config is available (bare-CSV path).
    """
    if population_config is not None:
        details = load_trajectory_details(traj_dir) or {}
        map_path = details.get("map_path")
        base_cfg = population_config.map_source_config
        map_config = base_cfg.model_copy(update={"file_path": map_path}) if map_path else base_cfg
        clip_percentile = population_config.normalization_percentile
        train_resampling = population_config.resampling
    else:
        map_config = TiffMapSourceConfig(file_path=DEFAULT_MAP)
        clip_percentile, train_resampling = 99.9, "average"

    samplers = make_pop_samplers(map_config, bounds_from_df(df),
                                 clip_percentile=clip_percentile, train_resampling=train_resampling)
    return build_metric_fn(samplers)


def process_trajectory_csv(csv_path: Path, traj_dir: Path, population_config=None) -> None:
    fuel_path = traj_dir / "fuel_bar_chart.png"
    noise_path = traj_dir / "noise_bar_chart.png"
    if fuel_path.exists() and noise_path.exists():
        print(f"Plots already exist, skipping: {traj_dir}")
        return

    df = pd.read_csv(csv_path)
    df["start_angle"] = ((df["start_angle"] / 10).round() * 10).astype(int)
    calculate_metrics = _build_calculate_metrics(df, traj_dir, population_config)
    df = calculate_metrics(df)

    for start_angle, group in df.groupby("start_angle"):
        print(f"Start Angle: {start_angle} {group['calculated_fuel'].sum()} kg, {group['sim_dt'].sum()} seconds")
        print(f"Start Angle: {start_angle} {group['calculated_noise'].sum()} noise, {group['sim_dt'].sum()} seconds")

    plot_fuel_bar_chart(df, fuel_path)
    plot_noise_bar_chart(df, noise_path)


def process_for_run(run_paths: RunPaths) -> None:
    if not run_paths.trajectories_dir.exists():
        print(f"No trajectories found for {run_paths.run_id}")
        return
    population_config = ExperimentConfig.load(run_paths.config).population_config
    for csv_path in sorted(run_paths.trajectories_dir.rglob("trajectories.csv")):
        traj_dir = csv_path.parent
        print(f"\n--- {run_paths.run_id} / {traj_dir.name} ---")
        process_trajectory_csv(csv_path, traj_dir, population_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process trajectories and compute metrics.")
    parser.add_argument("run_refs", nargs="+", help="Run reference(s) or path to a trajectories.csv")
    args = parser.parse_args()

    # Legacy: if a single arg is a CSV file, process that directly (uses DEFAULT_MAP)
    if len(args.run_refs) == 1 and args.run_refs[0].endswith(".csv"):
        csv_path = Path(args.run_refs[0])
        process_trajectory_csv(csv_path, csv_path.parent)
    else:
        for run_paths in [resolve_run(r) for r in args.run_refs]:
            process_for_run(run_paths)
