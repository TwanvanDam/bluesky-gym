import pickle
from pathlib import Path
from typing import Any

import bluesky
import gymnasium as gym
import numpy as np
import pandas as pd
from bluesky.tools.position import Position
from stable_baselines3 import SAC

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from bluesky_gym.envs.common import functions
from bluesky_gym.envs.common.environment_factory import load_env_and_model
from bluesky_gym.envs.common.functions import find_env_layer
from bluesky_gym.maps.map_datasets import TiffMapSourceConfig, RandomMapSourceConfig
from bluesky.tools.aero import nm


def remove_maps_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    for key, value in observation.items():
        if "map" in key:
            observation[key] = np.zeros_like(value)
    return observation


def simulate_trajectories(
        env: gym.Env,
        model: SAC,
        angle_interval: int,
        distance: int,
        seed: int,
        map_in_observation: bool = True,
        runway: str = "EHAM/RW18R"
) -> pd.DataFrame:
    navigation_env = find_env_layer(env, BaseNavigationEnv)
    navigation_env.save_trajectory = True
    destination = Position(name=runway, reflat=0, reflon=0)
    trajectories = []

    for start_angle in np.arange(0, 360, angle_interval):
        aircraft_lat, aircraft_lon = functions.get_point_at_distance(destination.lat, destination.lon,
                                                                     distance, start_angle)
        done = False
        options = {
            "destination_lat": destination.lat,
            "destination_lon": destination.lon,
            "destination_hdg": destination.refhdg,
            "aircraft_lat": aircraft_lat,
            "aircraft_lon": aircraft_lon,
        }
        obs, info = env.reset(options=options, seed=seed)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        episode_trajectories = navigation_env.get_telemetry_history()
        for record in episode_trajectories:
            record["start_angle"] = start_angle
        trajectories.extend(episode_trajectories)
    return pd.DataFrame(trajectories)


if __name__ == '__main__':
    bluesky.init()

    trajectory_details = {
        'run_name' : "PopulationWrapper-v0/2026-03-07_10_55_19",
        'runway' : "EHAM/RW18R",
        'map_path' : "/home/twanvandam/Thesis/scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff",
        'map_in_observation' : True,
        'start_distance' : 150 * nm / 1000,
    }
    trajectory_folder = Path("scripts/common/results/trajectories").joinpath(trajectory_details["run_name"])
    trajectory_folder.mkdir(parents=True, exist_ok=True)
    with open(trajectory_folder.joinpath("details.pkl"), "wb") as f:
        pickle.dump(trajectory_details, f)

    if trajectory_details["map_in_observation"]:
        validation_map = TiffMapSourceConfig(file_path=trajectory_details["map_path"])
    else:
        validation_map = RandomMapSourceConfig(type="zero", resolution_m=1000, source_unit="people_per_pixel")
    env, model = load_env_and_model(trajectory_details["run_name"], render_mode=None, map_config=validation_map)
    trajectories = simulate_trajectories(env, model, angle_interval=10, distance=trajectory_details["start_distance"], seed=42, map_in_observation=trajectory_details["map_in_observation"], runway=trajectory_details["runway"])
    trajectories.to_csv(trajectory_folder.joinpath("trajectories.csv"), index=False)
