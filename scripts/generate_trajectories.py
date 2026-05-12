import argparse
import dataclasses
import pickle
from dataclasses import dataclass
from pathlib import Path

import bluesky
import gymnasium as gym
import numpy as np
import pandas as pd
from bluesky.tools.aero import nm
from bluesky.tools.position import Position
from stable_baselines3 import SAC
from tqdm import tqdm

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from bluesky_gym.envs.common import functions
from bluesky_gym.envs.common.environment_factory import load_env_and_model
from bluesky_gym.envs.common.functions import find_env_layer
from bluesky_gym.maps.map_sources import TiffMapSourceConfig, RandomMapSourceConfig
from scripts.common.run_paths import resolve_run, RunPaths


@dataclass
class TrajectoryEvalConfig:
    """Defines a single evaluation scenario for trajectory generation."""
    runway: str
    start_distance: float  # [km] radius from the runway at which aircraft start
    map_path: Path | None = None

    # Force a runway starting position that is different from the bluesky database to ensure fair comparison to legacy models.
    destination_latlon: tuple[float, float] | None = None


def simulate_trajectories(
        env: gym.Env,
        model: SAC,
        angle_interval: int,
        distance: float,
        seed: int,
        runway: str = "EHAM/RW18R",
        destination_latlon: tuple[float, float] | None = None,
        progress_label: str = ""
) -> pd.DataFrame:
    navigation_env = find_env_layer(env, BaseNavigationEnv)
    navigation_env.save_trajectory = True
    destination = Position(name=runway, reflat=0, reflon=0)

    if destination_latlon:
        destination.lat = destination_latlon[0]
        destination.lon = destination_latlon[1]

    all_records = []
    angles = np.arange(0, 360, angle_interval)
    for start_angle in tqdm(angles, desc=f"Angles [{progress_label}]", leave=False):
        aircraft_lat, aircraft_lon = functions.get_point_at_distance(
            destination.lat, destination.lon, distance, start_angle
        )
        options = {
            "destination_lat": destination.lat,
            "destination_lon": destination.lon,
            "destination_hdg": destination.refhdg,
            "aircraft_lat": aircraft_lat,
            "aircraft_lon": aircraft_lon,
        }
        obs, info = env.reset(options=options, seed=seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        termination_reason = info.get("termination_reason", None)
        episode_records = navigation_env.get_telemetry_history()
        for record in episode_records:
            record["start_angle"] = start_angle
            record["termination_reason"] = termination_reason
        all_records.extend(episode_records)

    return pd.DataFrame(all_records)


def generate_for_run(run_paths: RunPaths, eval_configs: list[TrajectoryEvalConfig]) -> None:
    for eval_config in tqdm(eval_configs, desc=f"Configs [{run_paths.run_name}]"):
        runway_id = eval_config.runway.replace("/", "_")
        map_suffix = "map" if eval_config.map_path else "no_map"
        subdir_label = f"{runway_id}_{map_suffix}"

        trajectory_folder = run_paths.trajectory_subdir(subdir_label)

        try:
            trajectory_folder.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"Trajectory folder already exists, skipping: {trajectory_folder}")
            continue

        with open(trajectory_folder / "details.pkl", "wb") as f:
            pickle.dump(dataclasses.asdict(eval_config), f)

        validation_map = (
            TiffMapSourceConfig(file_path=eval_config.map_path)
            if eval_config.map_path
            else RandomMapSourceConfig(type="zero", resolution_m=1000, source_unit="people_per_pixel")
        )

        env, model = load_env_and_model(run_paths, render_mode=None, map_config=validation_map)
        trajectories = simulate_trajectories(
            env, model,
            angle_interval=10,
            distance=eval_config.start_distance,
            seed=42,
            runway=eval_config.runway,
            destination_latlon=eval_config.destination_latlon,
            progress_label=subdir_label,
        )
        trajectories.to_csv(trajectory_folder / "trajectories.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate trajectories for trained run(s).")
    parser.add_argument("run_refs", nargs="+",
                        help="Run reference(s) (e.g. 'PopulationWrapper-v0/RealMap_base_2026-...')")
    args = parser.parse_args()

    maps_base_path = Path(__file__).parent / "population_maps"
    real_map_path = maps_base_path / "europe_3035_1km.tif"

    eval_configs = [
        TrajectoryEvalConfig(
            runway="EHAM/RW27",
            destination_latlon=(52.3322, 4.75),
            map_path=real_map_path,
            start_distance=250,
        ),
        TrajectoryEvalConfig(
            runway="EDDF/RW25R",
            map_path=real_map_path,
            start_distance=250,
        ),
    ]

    runs = [resolve_run(r) for r in args.run_refs]
    if not runs:
        print("No matching runs found.")
        raise SystemExit(1)

    bluesky.init()

    for run_paths in tqdm(runs, desc="Runs"):
        print(f"\nGenerating trajectories for: {run_paths.run_id}")
        generate_for_run(run_paths, eval_configs)
