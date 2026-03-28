import argparse
import pickle
from pathlib import Path
from typing import Any

import bluesky
import gymnasium as gym
import numpy as np
import pandas as pd
from bluesky.tools.position import Position
from stable_baselines3 import SAC
from tqdm import tqdm

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from bluesky_gym.envs.common import functions
from bluesky_gym.envs.common.environment_factory import load_env_and_model, normalize_run_name
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
        runway: str = "EHAM/RW18R",
        name: str = ""
) -> pd.DataFrame:
    navigation_env = find_env_layer(env, BaseNavigationEnv)
    navigation_env.save_trajectory = True
    destination = Position(name=runway, reflat=0, reflon=0)
    trajectories = []

    angles = np.arange(0, 360, angle_interval)
    desc = f"Angles [{name}]" if name else "Angles"
    for start_angle in tqdm(angles, desc=desc, leave=False):
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
    parser = argparse.ArgumentParser(description="Generate trajectories for a trained run.")
    parser.add_argument("run_name", type=str, help="Name of the run (e.g. 'PopulationWrapper-v0/2026-03-27_...')")
    args = parser.parse_args()
    run_name = normalize_run_name(args.run_name)

    bluesky.init()

    trajectory_configs = [
        {
            'runway': "EHAM/RW27",
            'map_path': "/home/twanvandam/Thesis/scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff",
            'map_in_observation': True,
            'start_distance': 150 * nm / 1000,
        },
        {
            'runway': "EHAM/RW18R",
            'map_path': "/home/twanvandam/Thesis/scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff",
            'map_in_observation': True,
            'start_distance': 150 * nm / 1000,
        },
        {
            'runway': "EHAM/RW27",
            'map_path': "/home/twanvandam/Thesis/scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff",
            'map_in_observation': False,
            'start_distance': 150 * nm / 1000,
        },
    ]

    for trajectory_details in tqdm(trajectory_configs, desc="Trajectory configs"):
        trajectory_details['run_name'] = run_name
        name = f"{trajectory_details['runway']}_{'map' if trajectory_details['map_in_observation'] else 'no_map'}"
        trajectory_folder = Path("scripts/common/results/trajectories") / run_name / name
        trajectory_folder.mkdir(parents=True, exist_ok=True)
        with open(trajectory_folder / "details.pkl", "wb") as f:
            pickle.dump(trajectory_details, f)

        if trajectory_details["map_in_observation"]:
            validation_map = TiffMapSourceConfig(file_path=trajectory_details["map_path"])
        else:
            validation_map = RandomMapSourceConfig(type="zero", resolution_m=1000, source_unit="people_per_pixel")

        env, model = load_env_and_model(run_name, render_mode=None, map_config=validation_map)
        trajectories = simulate_trajectories(
            env, model,
            angle_interval=10,
            distance=trajectory_details["start_distance"],
            seed=42,
            runway=trajectory_details["runway"],
            name=name
        )
        trajectories.to_csv(trajectory_folder / "trajectories.csv", index=False)
