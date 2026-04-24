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
from bluesky_gym.envs.common.environment_factory import load_env_and_model
from bluesky_gym.envs.common.functions import find_env_layer
from bluesky_gym.maps.map_datasets import TiffMapSourceConfig, RandomMapSourceConfig
from bluesky.tools.aero import nm
from scripts.common.run_paths import resolve_run, iter_runs, find_runs, RunPaths


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
        latlon: tuple[float, float] | None = None,
        name: str = ""
) -> pd.DataFrame:
    navigation_env = find_env_layer(env, BaseNavigationEnv)
    navigation_env.save_trajectory = True
    destination = Position(name=runway, reflat=0, reflon=0)

    # override destination lat/lon if provided to ensure fair comparison between new and legacy model
    if latlon:
        destination.lat = latlon[0]
        destination.lon = latlon[1]

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


def generate_for_run(run_paths: RunPaths) -> None:
    trajectory_configs = [
        {
            'runway': "EHAM/RW27",
            'latlon' : (52.3322, 4.75),
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
            'latlon' : (52.3322, 4.75),
            'map_path': "/home/twanvandam/Thesis/scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff",
            'map_in_observation': False,
            'start_distance': 150 * nm / 1000,
        },
    ]

    for trajectory_details in tqdm(trajectory_configs, desc=f"Configs [{run_paths.run_name}]"):
        trajectory_details['run_name'] = run_paths.run_id
        name = f"{trajectory_details['runway']}_{('map' if trajectory_details['map_in_observation'] else 'no_map')}"
        trajectory_folder = run_paths.trajectory_subdir(name)
        try:
            trajectory_folder.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"Trajectory folder already exists, skipping: {trajectory_folder}")
            continue
        with open(trajectory_folder / "details.pkl", "wb") as f:
            pickle.dump(trajectory_details, f)

        if trajectory_details["map_in_observation"]:
            validation_map = TiffMapSourceConfig(file_path=trajectory_details["map_path"])
        else:
            validation_map = RandomMapSourceConfig(type="zero", resolution_m=1000, source_unit="people_per_pixel")

        env, model = load_env_and_model(run_paths, render_mode=None, map_config=validation_map)
        trajectories = simulate_trajectories(
            env, model,
            angle_interval=10,
            distance=trajectory_details["start_distance"],
            seed=42,
            runway=trajectory_details["runway"],
            latlon=trajectory_details.get("latlon", None),
            name=name
        )
        trajectories.to_csv(trajectory_folder / "trajectories.csv", index=False)


def collect_runs(args) -> list[RunPaths]:
    """Resolve CLI arguments to a list of RunPaths."""
    if args.env:
        return list(iter_runs(env_name=args.env))
    if args.pattern:
        return find_runs(pattern=args.pattern, env_name=None)
    return [resolve_run(r) for r in args.run_refs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate trajectories for trained run(s).")
    parser.add_argument("run_refs", nargs="*", help="Run reference(s) (e.g. 'PopulationWrapper-v0/RealMap_base_2026-...')")
    parser.add_argument("--env", default=None, help="Generate for all runs of this env name.")
    parser.add_argument("--pattern", default=None, help="Glob pattern to match run names.")
    args = parser.parse_args()

    if not args.run_refs and not args.env and not args.pattern:
        parser.error("Provide run reference(s), --env, or --pattern.")

    runs = collect_runs(args)
    if not runs:
        print("No matching runs found.")
        raise SystemExit(1)

    bluesky.init()

    for run_paths in tqdm(runs, desc="Runs"):
        print(f"\nGenerating trajectories for: {run_paths.run_id}")
        generate_for_run(run_paths)
