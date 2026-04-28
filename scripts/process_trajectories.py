import argparse
from pathlib import Path

import pandas as pd
from bluesky.tools.position import Position

from bluesky_gym.maps.map_sources import TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler, MapObservationConfig
from bluesky_gym.metrics.fuel_model import FuelModel
from bluesky_gym.metrics.noise_model import NoiseConfig
from scripts.common.run_paths import resolve_run, RunPaths


def calculate_fuel(altitude, tas, sim_dt, mass):
    fuel_flow = fuel_model.step_fuel_flow(mass=mass, tas=tas, altitude=altitude) * sim_dt
    return fuel_flow


def calculate_noise(lat, lon, altitude, sim_dt):
    pos = Position(name=f"{lat},{lon}", reflat=0, reflon=0)
    kernel_shape_meters, kernel_shape_pixels = noise_model.get_noise_power_kernel_shape_meters_and_pixels(altitude)
    noise_kernel_map_extract_config = MapObservationConfig(shape=kernel_shape_pixels, range=kernel_shape_meters)
    population_map = raster_sampler.get_observation_clipped(center_position=pos, orientation=0,
                                                            observation_config=noise_kernel_map_extract_config)
    noise = noise_model.step_total_noise(population_map, altitude, sim_dt)
    return noise


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


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    altitude_key = "altitude" if "altitude" in df.columns else "alt"
    df["calculated_fuel"] = df.apply(
        lambda row: calculate_fuel(row[altitude_key], row["tas"], row["sim_dt"], row["mass"]), axis=1)
    df["calculated_noise"] = df.apply(
        lambda row: calculate_noise(row["lat"], row["lon"], row[altitude_key], row["sim_dt"]), axis=1)
    return df


def process_trajectory_csv(csv_path: Path, traj_dir: Path) -> None:
    fuel_path = traj_dir / "fuel_bar_chart.png"
    noise_path = traj_dir / "noise_bar_chart.png"
    if fuel_path.exists() and noise_path.exists():
        print(f"Plots already exist, skipping: {traj_dir}")
        return

    df = pd.read_csv(csv_path)
    df["start_angle"] = ((df["start_angle"] / 10).round() * 10).astype(int)
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
    for csv_path in sorted(run_paths.trajectories_dir.rglob("trajectories.csv")):
        traj_dir = csv_path.parent
        print(f"\n--- {run_paths.run_id} / {traj_dir.name} ---")
        process_trajectory_csv(csv_path, traj_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process trajectories and compute metrics.")
    parser.add_argument("run_refs", nargs="+", help="Run reference(s) or path to a trajectories.csv")
    args = parser.parse_args()

    validation_map_config = TiffMapSourceConfig(
        file_path="/home/twanvandam/Thesis/scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff")
    validation_map = validation_map_config.build()
    raster_sampler = RasterSampler(map_source=validation_map, resampling="sum", destination_crs="epsg:3035")

    noise_model_config = NoiseConfig()
    noise_model = noise_model_config.build()
    fuel_model = FuelModel("a320")

    # Legacy: if a single arg is a CSV file, process that directly
    if len(args.run_refs) == 1 and args.run_refs[0].endswith(".csv"):
        csv_path = Path(args.run_refs[0])
        process_trajectory_csv(csv_path, csv_path.parent)
    else:
        for run_paths in [resolve_run(r) for r in args.run_refs]:
            process_for_run(run_paths)
