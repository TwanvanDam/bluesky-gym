import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from rasterio.plot import plotting_extent
from stable_baselines3 import SAC
import gymnasium as gym
from bluesky_gym.envs.base_navigation_env import Airport, Position
from bluesky_gym.envs.common import functions
from scripts.config import ExperimentConfig, MapSourceConfig
from scripts.run_experiment import load_env_from_config

def load_env_and_model(run_name: str, render_mode: str | None = "human", map_config: MapSourceConfig | None = None) -> tuple[gym.Env, SAC]:
    experiment_config_path = Path("scripts/common/results/configs_backup").joinpath(run_name).with_suffix(".yaml")
    model_path = Path("scripts/common/results/models_backup").joinpath(run_name).with_suffix(".zip")
    experiment_config = ExperimentConfig.load(experiment_config_path)

    # Override map source config if provided
    if map_config and experiment_config.population_config:
        experiment_config.population_config.map_source_config = map_config

    env, _ = load_env_from_config(experiment_config=experiment_config, render_mode=render_mode)

    if experiment_config.training_config.algorithm == "SAC":
        model = SAC.load(model_path, env=env, device='auto')
    else:
        raise NotImplementedError

    return env, model

def render_experiment(run_name: str, map_config: MapSourceConfig | None = None):
    env, model = load_env_and_model(run_name, render_mode="human", map_config=map_config)
    while True:
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

def plot_trajectories_on_map(run_name: str, angle_interval: int = 30, distance: int = 300, map_config: MapSourceConfig | None = None):
    env, model = load_env_and_model(run_name, render_mode=None, map_config=map_config)

    angles = np.arange(0, 360, angle_interval)
    destination = Airport(Position(lat=52.334, lon=4.7092), hdg=180)
    figure = plt.figure()
    env.reset(seed=42)
    background = env.env.background_map.copy()
    print(env.env.map_source.crs)
    print(env.unwrapped.pygame_crs)
    background[background <= 0] = np.nan  # Set zero values to NaN for better visualization
    extent = plotting_extent(background, env.env.background_transform)
    plt.imshow(background, extent=extent, origin="upper", cmap="Blues", vmin=0, vmax=np.nanpercentile(background, 99))
    for angle in list(angles):
        aircraft_lat, aircraft_lon = functions.get_point_at_distance(destination.position.lat, destination.position.lon,
                                   distance, angle)
        done = False
        obs, info = env.reset(options={
            "airport_lat": destination.position.lat,
            "airport_lon": destination.position.lon,
            "airport_hdg": destination.hdg,
            "aircraft_lat": aircraft_lat,
            "aircraft_lon": aircraft_lon,
        }, seed=42)
        while not done:
            for key, value in obs.items():
                if "map" in key:
                    obs[key] = np.zeros_like(value)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        points = [env.unwrapped.coordinate_transformer.transform(position.lon, position.lat) for position in
                  env.unwrapped.aircraft_positions]
        xs, ys = zip(*points)
        plt.plot(xs, ys, color="black")
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.scatter(*env.unwrapped.coordinate_transformer.transform(destination.position.lon, destination.position.lat), marker=".", linewidths=5)
    plt.show()


if __name__ == '__main__':
    run_name = "PopulationWrapper-v0/2026-03-07_10_55_19"
    validation_map = MapSourceConfig(type="tiff", file_path="scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff")

    parser = argparse.ArgumentParser(description="Show trained RL model(s) from experiment config(s).")
    parser.add_argument(
        "name",
        nargs="?",
        default=run_name,
        help=f"Name of a single experiment run. If omitted, {run_name} is used.",
    )
    args = parser.parse_args()
    plot_trajectories_on_map(args.name, map_config=validation_map)
    # render_experiment(args.name) #, map_config=validation_map)