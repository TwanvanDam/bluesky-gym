import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rasterio.plot import plotting_extent
from stable_baselines3 import SAC
import gymnasium as gym
from bluesky_gym.envs.base_navigation_env import Destination, Position
from bluesky_gym.envs.common import functions
from bluesky_gym.maps.map_datasets import MapSourceConfigType, TiffMapSourceConfig
from scripts.config import ExperimentConfig
from scripts.run_experiment import load_env_from_config

def load_env_and_model(run_name: str, render_mode: str | None = "human", map_config: MapSourceConfigType | None = None) -> tuple[gym.Env, SAC]:
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

def render_experiment(run_name: str, map_config: MapSourceConfigType | None = None):
    env, model = load_env_and_model(run_name, render_mode="human", map_config=map_config)
    while True:
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"Fuel: {info['total_episode_fuel_used']:.2f} kg, Reward:{info['total_episode_fuel_reward']:.2f}")
        print(f"Noise: {info['total_episode_noise']:.2f}, Reward:{info['total_episode_noise_reward']:.2f}")
        print(f"Episode Length: {info['episode_length_seconds']/60:.2f} minutes")

def _simulate_trajectories(
        env: gym.Env,
        model: SAC,
        angle_interval: int,
        distance: int,
        seed: int,
        option: str = "map",
) -> tuple[pd.DataFrame, np.ndarray, tuple[float, float, float, float], tuple[float, float], list[list[tuple[float, float]]]]:
    save_keys = ["episode_length_seconds", "total_episode_fuel_used", "total_episode_fuel_reward", "total_episode_noise", "total_episode_noise_reward"]
    results = []
    coordinate_transformer = env.unwrapped.coordinate_transformer
    angles = np.arange(0, 360, angle_interval)
    destination = Destination(Position(lat=52.334, lon=4.7092), hdg=180)
    env.reset(seed=seed)
    background = env.env.background_map.copy()
    background_transform = env.env.get_background_transform()
    background[background <= 0] = np.nan  # Set zero values to NaN for better visualization
    extent = plotting_extent(background, background_transform)
    destination_xy = coordinate_transformer.transform(destination.position.lon, destination.position.lat)
    trajectories = []

    for start_angle in list(angles):
        aircraft_lat, aircraft_lon = functions.get_point_at_distance(destination.position.lat, destination.position.lon,
                                                              distance, start_angle)
        done = False
        options = {
            "airport_lat": destination.position.lat,
            "airport_lon": destination.position.lon,
            "airport_hdg": destination.hdg,
            "aircraft_lat": aircraft_lat,
            "aircraft_lon": aircraft_lon,
        }
        obs, info = env.reset(options=options, seed=seed)
        while not done:
            if option == "no_map":
                for key, value in obs.items():
                    if "map" in key:
                        obs[key] = np.zeros_like(value)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        points = [coordinate_transformer.transform(position.lon, position.lat) for position in
                  env.unwrapped.aircraft_positions]
        trajectories.append(points)
        results.append({"start_angle": start_angle, "map_type": option , **{key: info[key] for key in save_keys}})

    return results, background, extent, destination_xy, trajectories


def _plot_trajectories(background: np.ndarray,
                       extent: tuple[float, float, float, float],
                       destination_xy: tuple[float, float],
                       trajectories: list[list[tuple[float, float]]]):
    plt.imshow(background, extent=extent, origin="upper", cmap="Blues", vmin=0, vmax=np.nanpercentile(background, 99))
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.scatter(*destination_xy, marker=".", linewidths=5)
    for points in trajectories:
        plt.plot(*zip(*points), color="black")
    plt.show()


def _plot_trajectory_metrics(results: pd.DataFrame):
    plt.bar(results["start_angle"], results["total_episode_fuel_used"], color="blue", label="Fuel Used [kg]", width=8)
    plt.bar(results["start_angle"], -results["total_episode_noise"], color="red", label="Noise", width=8)
    plt.xlabel("Starting Angle (degrees)")
    plt.ylabel("Total Episode Metrics")
    plt.title("Total Fuel Used and Noise by Starting Angle")
    plt.xticks(results["start_angle"])
    plt.legend()
    plt.show()

def _plot_trajectory_metrics_comparison(results: pd.DataFrame):
    import seaborn as sns

    plot_df = results.copy()
    plot_df["total_episode_noise"] = -plot_df["total_episode_noise"]
    plot_df = plot_df.melt(
        id_vars=["start_angle", "map_type"],
        value_vars=["total_episode_fuel_used", "total_episode_noise"],
        var_name="metric",
        value_name="value",
    )
    plot_df["metric"] = plot_df["metric"].replace({
        "total_episode_fuel_used": "Fuel Used [kg]",
        "total_episode_noise": "Noise",
    })

    # 1. Create the faceted bar chart
    g = sns.catplot(
        data=plot_df,
        kind="bar",
        x="start_angle",
        y="value",
        hue="metric",
        col="map_type",
        dodge=False
    )

    # 2. Update labels and title
    g.set_axis_labels("Starting Angle (degrees)", "Total Episode Metrics")
    g.figure.suptitle("Total Fuel Used and Noise by Starting Angle", y=1.05)

    # 3. Adjust legend and display
    # sns.move_legend(g, "upper right")
    plt.show()


def plot_trajectories_on_map(run_name: str, angle_interval: int = 45, distance: int = 200, map_config: MapSourceConfigType | None = None, seed: int = 42):
    results = []
    for option in ["map", "no_map"]:
        env, model = load_env_and_model(run_name, render_mode=None, map_config=map_config)
        results_simulation, background, extent, destination_xy, trajectories = _simulate_trajectories(
            env=env,
            model=model,
            angle_interval=angle_interval,
            distance=distance,
            seed=seed,
            option = option
        )
        _plot_trajectories(background, extent, destination_xy, trajectories)
        results.extend(results_simulation)
    pd.DataFrame(results).to_csv("Test.csv")

def compare_trajectories_on_map(run_name: str, angle_interval: int = 30, distance: int = 300, map_config: MapSourceConfigType | None = None):
    env, model = load_env_and_model(run_name, render_mode=None, map_config=map_config)
    angles = np.arange(0, 360, angle_interval)
    coordinate_transformer = env.unwrapped.coordinate_transformer
    destination = Destination(Position(lat=52.334, lon=4.7092), hdg=180)
    fig, axs = plt.subplots(1,2)
    for ax, obs_type in zip(axs, ["with_map", "without_map"]):
        env.reset(seed=42)
        background = env.env.background_map.copy()
        background_transform = env.env.get_background_transform()
        extent = plotting_extent(background, background_transform)
        ax.imshow(background, extent=extent, origin="upper", cmap="Blues", vmin=0, vmax=np.nanpercentile(background, 99))
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
                    if "map" in key and obs_type == "without_map":
                        obs[key] = np.zeros_like(value)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            points = [coordinate_transformer.transform(position.lon, position.lat) for position in
                      env.unwrapped.aircraft_positions]
            xs, ys = zip(*points)
            ax.plot(xs, ys, color="black")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.scatter(*coordinate_transformer.transform(destination.position.lon, destination.position.lat), marker=".", linewidths=5)
        ax.set_title(f"{obs_type.replace('_', ' ').title()}")
    plt.show()

if __name__ == '__main__':
    run_name = "PopulationWrapper-v0/2026-03-07_10_55_19"
    validation_map = TiffMapSourceConfig(file_path="scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff")
    # validation_map = MapSourceConfig(type="population_density")

    parser = argparse.ArgumentParser(description="Show trained RL model(s) from experiment config(s).")
    parser.add_argument(
        "name",
        nargs="?",
        default=run_name,
        help=f"Name of a single experiment run. If omitted, {run_name} is used.",
    )
    args = parser.parse_args()
    # plot_trajectories_on_map(args.name, map_config=validation_map)
    # df = pd.read_csv("Test.csv")
    # _plot_trajectory_metrics_comparison(df)
    render_experiment(args.name, map_config=validation_map)