import argparse
from pathlib import Path

import pandas as pd
from bluesky.tools.position import Position

from bluesky_gym.maps.map_datasets import TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler
from bluesky_gym.metrics.noise_model import NoiseConfig
from bluesky_gym.metrics.fuel_model import FuelModel
from scripts.common.run_paths import resolve_run, iter_runs, find_runs, RunPaths

def calculate_fuel(altitude, tas, sim_dt, mass):
    fuel_flow = fuel_model.step_fuel_flow(mass=mass, tas=tas, altitude=altitude) * sim_dt
    return fuel_flow

def calculate_noise(lat, lon, altitude, sim_dt):
    pos = Position(name=f"{lat},{lon}", reflat=0, reflon=0)
    kernel_shape_meters, kernel_shape_pixels = noise_model.get_noise_power_kernel_shape_meters_and_pixels(altitude)
    population_map = raster_sampler.get_observation_clipped(center_position=pos, orientation=0, out_meters=kernel_shape_meters, out_shape=kernel_shape_pixels)
    noise = noise_model.step_total_noise(population_map, altitude, sim_dt)
    return noise

def plot_noise_bar_chart(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    noise_by_angle = df.groupby("start_angle")["calculated_noise"].sum()
    noise_by_angle.plot(kind="bar")
    plt.title("Total Noise by Start Angle")
    plt.xlabel("Start Angle (degrees)")
    plt.ylabel("Total Noise")
    plt.xticks(rotation=45)
    plt.show()

def plot_fuel_bar_chart(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    fuel_by_angle = df.groupby("start_angle")["calculated_fuel"].sum()
    fuel_by_angle.plot(kind="bar")
    plt.title("Total Fuel Consumption by Start Angle")
    plt.xlabel("Start Angle (degrees)")
    plt.ylabel("Total Fuel Consumption (kg)")
    plt.xticks(rotation=45)
    plt.show()

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    altitude_key = "altitude" if "altitude" in df.columns else "alt"
    df["calculated_fuel"] = df.apply(lambda row: calculate_fuel(row[altitude_key], row["tas"], row["sim_dt"], row["mass"]), axis=1)
    df["calculated_noise"] = df.apply(lambda row: calculate_noise(row["lat"], row["lon"], row[altitude_key], row["sim_dt"]), axis=1)
    return df

def process_trajectory_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df["start_angle"] = ((df["start_angle"] / 10).round() * 10).astype(int)
    df = calculate_metrics(df)

    for start_angle, group in df.groupby("start_angle"):
        print(f"Start Angle: {start_angle} {group['calculated_fuel'].sum()} kg, {group['sim_dt'].sum()} seconds")
        print(f"Start Angle: {start_angle} {group['calculated_noise'].sum()} noise, {group['sim_dt'].sum()} seconds")

    plot_fuel_bar_chart(df)
    plot_noise_bar_chart(df)

def process_for_run(run_paths: RunPaths) -> None:
    if not run_paths.trajectories_dir.exists():
        print(f"No trajectories found for {run_paths.run_id}")
        return
    for traj_dir in sorted(run_paths.trajectories_dir.iterdir()):
        csv_path = traj_dir / "trajectories.csv"
        if csv_path.exists():
            print(f"\n--- {run_paths.run_id} / {traj_dir.name} ---")
            process_trajectory_csv(csv_path)

def collect_runs(args) -> list[RunPaths]:
    if args.env:
        return list(iter_runs(env_name=args.env))
    if args.pattern:
        return find_runs(pattern=args.pattern, env_name=None)
    return [resolve_run(r) for r in args.run_refs]

if __name__ == '__main__':
    validation_map_config = TiffMapSourceConfig(file_path="/home/twanvandam/Thesis/scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff")
    validation_map = validation_map_config.build()
    raster_sampler = RasterSampler(map_source=validation_map, resampling="cubic_spline", destination_crs="epsg:3035")

    noise_model_config = NoiseConfig()
    noise_model = noise_model_config.build()
    fuel_model = FuelModel("a320")

    parser = argparse.ArgumentParser(description="Process trajectories and compute metrics.")
    parser.add_argument("run_refs", nargs="*", help="Run reference(s) or path to a trajectories.csv")
    parser.add_argument("--env", default=None, help="Process for all runs of this env name.")
    parser.add_argument("--pattern", default=None, help="Glob pattern to match run names.")
    args = parser.parse_args()

    if not args.run_refs and not args.env and not args.pattern:
        parser.error("Provide run reference(s), --env, or --pattern.")

    # Legacy: if a single arg is a CSV file, process that directly
    if len(args.run_refs) == 1 and args.run_refs[0].endswith(".csv"):
        process_trajectory_csv(Path(args.run_refs[0]))
    else:
        runs = collect_runs(args)
        if not runs:
            print("No matching runs found.")
            raise SystemExit(1)
        for run_paths in runs:
            process_for_run(run_paths)
