from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import SAC

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from bluesky_gym.maps.map_datasets import MapSourceConfigType
from bluesky_gym.wrappers.distance_normalizer import DistanceNormalization
from bluesky_gym.wrappers.map_observation_normalizer import MapObservationNormalizer
from bluesky_gym.wrappers.population import Population
from bluesky_gym.wrappers.sin_cos_normalizer import SinCosNormalization
from scripts.common.run_paths import RunPaths, resolve_run
from scripts.config import ExperimentConfig


def load_env_from_config(experiment_config: ExperimentConfig, render_mode: str | None = None) -> tuple[gym.Env, str]:
    env = BaseNavigationEnv(config=experiment_config.navigation_config, render_mode=render_mode)

    if experiment_config.navigation_config.use_sin_cos_obs:
        env = SinCosNormalization(env)

    if experiment_config.navigation_config.normalize_distance_obs:
        env = DistanceNormalization(env, normalization_factor=experiment_config.navigation_config.normalize_distance_obs)

    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    if experiment_config.population_config:
        env = Population(env, config=experiment_config.population_config)
        if experiment_config.population_config.observation_normalization:
            env = MapObservationNormalizer(env, mode=experiment_config.population_config.observation_normalization)
        env_name = "PopulationWrapper-v0"
    else:
        env_name = "BaseNavigationEnv-v0"

    return env, env_name

def load_env_and_model(run: str | RunPaths, render_mode: str | None = "human", map_config: MapSourceConfigType | None = None) -> tuple[gym.Env, SAC]:
    run_paths = resolve_run(run) if isinstance(run, str) else run

    experiment_config = ExperimentConfig.load(run_paths.config)

    # Override map source config if provided
    if map_config and experiment_config.population_config:
        experiment_config.population_config.map_source_config = map_config

    env, _ = load_env_from_config(experiment_config=experiment_config, render_mode=render_mode)

    if experiment_config.agent_config.algorithm == "SAC":
        try:
            model = SAC.load(run_paths.model, env=env, device='auto')
        except FileNotFoundError:
            latest = run_paths.latest_checkpoint()
            if latest:
                print(f"Final model not found. Loading latest checkpoint: {latest}")
                model = SAC.load(latest, env=env, device='auto')
            else:
                raise FileNotFoundError(
                    f"No model or checkpoints found for run {run_paths.run_id}"
                )
    else:
        raise NotImplementedError

    return env, model
