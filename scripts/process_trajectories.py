import argparse

import pandas as pd
from bluesky.tools.position import Position

from bluesky_gym.maps.map_datasets import TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler
from bluesky_gym.metrics.noise_model import NoiseConfig
from bluesky_gym.metrics.fuel_model import FuelModel

def calculate_fuel(altitude, tas, sim_dt, mass):
    fuel_flow = fuel_model.step_fuel_flow(mass=mass, tas=tas, altitude=altitude) * sim_dt
    return fuel_flow

def calculate_noise(lat, lon, altitude, sim_dt):
    pos = Position(name=f"{lat},{lon}", reflat=0, reflon=0)
    kernel_shape_meters, kernel_shape_pixels = noise_model.get_noise_power_kernel_shape_meters_and_pixels(altitude)
    population_map = raster_sampler.get_observation_clipped(center_position=pos, orientation=0, out_meters=kernel_shape_meters, out_shape=kernel_shape_pixels)
    noise = noise_model.step_total_noise(population_map, altitude, sim_dt)
    return noise

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["calculated_fuel"] = df.apply(lambda row: calculate_fuel(row["altitude"], row["tas"], row["sim_dt"], row["mass"]), axis=1)
    df["calculated_noise"] = df.apply(lambda row: calculate_noise(row["lat"], row["lon"], row["altitude"], row["sim_dt"]), axis=1)
    return df

if __name__ == '__main__':
    validation_map_config = TiffMapSourceConfig(file_path="/home/twanvandam/Thesis/scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff")
    validation_map = validation_map_config.build()
    raster_sampler = RasterSampler(map_source=validation_map, resampling="cubic_spline", destination_crs="epsg:3035")

    noise_model_config = NoiseConfig()
    noise_model = noise_model_config.build()
    fuel_model = FuelModel("a320")

    parser = argparse.ArgumentParser(description="Plot trajectories on map")
    parser.add_argument("trajectories_csv", type=str, help="Path to CSV file containing trajectory data")
    args = parser.parse_args()
    df = pd.read_csv(args.trajectories_csv)
    df = calculate_metrics(df)

    for start_angle, group in df.groupby("start_angle"):
        print(f"Start Angle: {start_angle} {group['calculated_fuel'].sum()} kg, {group['sim_dt'].sum()} seconds")
        print(f"Start Angle: {start_angle} {group['calculated_noise'].sum()} noise, {group['sim_dt'].sum()} seconds")


