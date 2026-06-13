import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from bluesky_gym.maps.map_sources import TiffMapSourceConfig, RandomMapSourceConfig, \
    TransformedTiffMapSourceConfig, MapSourceConfigType
from bluesky_gym.maps.map_transforms import Clip, ScaleValues
from scripts.common.run_paths import resolve_run, RunPaths, write_trajectory_details
from scripts.config import ExperimentConfig


@dataclass
class TrajectoryEvalConfig:
    """Defines a single evaluation scenario for trajectory generation."""
    runway: str
    start_distance: float  # [km] radius from the runway at which aircraft start
    map_path: Path | None = None
    model: Literal["best", "final"] = "best"

    # Force a runway starting position that is different from the bluesky database to ensure fair comparison to legacy models.
    destination_latlon: tuple[float, float] | None = None

    # Label used in the trajectory subdirectory name; defaults to "map"/"no_map".
    map_label: str | None = None

    # Scale population density to change the ratio of fuel to noise.
    scale_density: float | None = None


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


def eval_map_config(train_map_config: MapSourceConfigType, trajectory_config: TrajectoryEvalConfig) -> MapSourceConfigType:
    """Reuse the training map pipeline, swapping only the file and making it deterministic."""
    if isinstance(train_map_config, TransformedTiffMapSourceConfig):
        value_transforms = [transform for transform in train_map_config.value_transforms
                                     if isinstance(transform, Clip)]
        if trajectory_config.scale_density:
            value_transforms = [ScaleValues(factor=(trajectory_config.scale_density, trajectory_config.scale_density)), *value_transforms]
        return train_map_config.model_copy(update={
            "file_path" : str(trajectory_config.map_path),
            "spatial_transforms": [],
            "value_transforms": value_transforms
        })
    elif isinstance(train_map_config, TiffMapSourceConfig):
        if trajectory_config.scale_density:
            raise NotImplementedError("Scaling population density is not supported yet for legacy tiff files")
        return train_map_config.model_copy(update={"file_path": str(trajectory_config.map_path)})
    else:
        raise ValueError(f"Invalid map config: {train_map_config}")


def generate_for_run(run_paths: RunPaths, eval_configs: list[TrajectoryEvalConfig]) -> None:
    train_config = ExperimentConfig.load(run_paths.config)

    for eval_config in tqdm(eval_configs, desc=f"Configs [{run_paths.run_name}]"):
        runway_id = eval_config.runway.replace("/", "_")
        map_suffix = eval_config.map_label or ("map" if eval_config.map_path else "no_map")
        subdir_label = f"{runway_id}_{map_suffix}_{eval_config.model}"

        trajectory_folder = run_paths.trajectory_subdir(subdir_label)

        try:
            trajectory_folder.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"Trajectory folder already exists, skipping: {trajectory_folder}")
            continue

        write_trajectory_details(trajectory_folder, dataclasses.asdict(eval_config))

        if eval_config.map_path:
            if train_config.population_config is None:
                raise ValueError(
                    f"Run {run_paths.run_id} has no population_config; cannot evaluate on a map."
                )
            validation_map = eval_map_config(train_config.population_config.map_source_config, eval_config)
        else:
            validation_map = RandomMapSourceConfig(type="zero", resolution_m=1000, source_unit="people_per_pixel")

        env, model = load_env_and_model(run_paths, render_mode=None, map_config=validation_map, model_type=eval_config.model)
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
    parser.add_argument("--model", default="best", type=str, help="Trained model: 'best' or 'final', default='best'")
    parser.add_argument("--runway", default="EHAM/RW27", type=str, help="Select the runway to use.")
    parser.add_argument("--lat_lon", default=None, type=float, nargs=2, help="Force different Latitude/Longitude coordinates.")
    parser.add_argument("--start_distance", default=250, type=int, help="Start distance in km.")
    parser.add_argument("--map_path", default=Path("scripts/population_maps/europe_3035_1km.tif"), type=Path, help="Trained map path")
    parser.add_argument("--no_map", action="store_true",
                        help="Fly on a zeroed-out map (RandomMapSource 'zero'); ignored for runs without a population_config.")
    parser.add_argument("--label", default="", type=str, help="Map label to correctly identify trajectories")
    parser.add_argument("--scale_density", type=float, help="Scale the density map.")
    args = parser.parse_args()

    eval_configs = [
        TrajectoryEvalConfig(
            runway=args.runway,
            destination_latlon=tuple(args.lat_lon) if args.lat_lon else None,
            map_path=None if args.no_map else args.map_path,
            model=args.model,
            start_distance=args.start_distance,
            map_label=args.label or None,
            scale_density=args.scale_density
        )]

    runs = [resolve_run(r) for r in args.run_refs]
    if not runs:
        print("No matching runs found.")
        raise SystemExit(1)

    bluesky.init()

    for run_paths in tqdm(runs, desc="Runs"):
        print(f"\nGenerating trajectories for: {run_paths.run_id}")
        generate_for_run(run_paths, eval_configs)
