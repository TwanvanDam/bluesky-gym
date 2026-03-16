import argparse
import pickle
from pathlib import Path

import bluesky
import numpy as np
import pandas as pd
import pyproj
from bluesky.tools.position import Position
from matplotlib import pyplot as plt
from rasterio.plot import plotting_extent
from stable_baselines3 import SAC
import gymnasium as gym
from bluesky_gym.envs.common import functions
from bluesky_gym.maps.map_datasets import MapSourceConfigType, TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler
from scripts.config import ExperimentConfig
from scripts.run_experiment import load_env_from_config
import bluesky as bs

def plot_trajectories(trajectories: pd.DataFrame, map_config: MapSourceConfigType):
    bs.init()
    map_source = TiffMapSourceConfig(file_path=map_config.file_path).build()
    raster_sampler = RasterSampler(map_source, resampling="cubic_spline", destination_crs="epsg:3035")
    destination = Position(name="EHAM/RW18R", reflat=0, reflon=0)

    coordinate_transformer = pyproj.Transformer.from_crs("WGS84", raster_sampler.destination_crs, always_xy=True)
    destination_xy = coordinate_transformer.transform(destination.lon, destination.lat)

    trajectories["x"], trajectories["y"] = coordinate_transformer.transform(trajectories["lon"].values, trajectories["lat"].values)

    x_min = - 25_000 + trajectories["x"].min()
    x_max = 25_000 + trajectories["x"].max()
    y_min = -25_000 + trajectories["y"].min()
    y_max = 25_000 + trajectories["y"].max()

    background = raster_sampler.get_background(x_min, y_min, x_max, y_max, width=512, height=512)
    background_transform = raster_sampler.get_dst_transform_from_bounds(x_min, y_min, x_max, y_max, width=512, height=512)
    extent = plotting_extent(background, background_transform)

    plt.imshow(background, extent=extent, origin="upper", cmap="Blues", vmin=0, vmax=np.nanpercentile(background, 99))
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.scatter(*destination_xy, marker=".", linewidths=5)

    for start_angle, group in trajectories.groupby("start_angle"):
        plt.plot(group["x"], group["y"], color="black")
    plt.title("Trajectories on Map")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trajectories on map")
    parser.add_argument("trajectories_csv", type=str, help="Path to CSV file containing trajectory data")
    args = parser.parse_args()
    df = pd.read_csv(args.trajectories_csv)
    with open(Path(args.trajectories_csv).parent.joinpath("details.pkl"), "rb") as f:
        trajectory_details = pickle.load(f)

    map_config = TiffMapSourceConfig(file_path=trajectory_details["map_path"])
    plot_trajectories(df, map_config)