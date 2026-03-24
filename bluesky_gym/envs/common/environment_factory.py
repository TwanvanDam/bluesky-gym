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
from scripts.config import ExperimentConfig


RESULTS_ROOT = Path("scripts/common/results")
CONFIGS_BACKUP_ROOT = RESULTS_ROOT / "configs_backup"
MODELS_BACKUP_ROOT = RESULTS_ROOT / "models_backup"


def normalize_run_name(run_name: str) -> str:
    """Normalize a run reference to the canonical '<env>/<timestamp>' format."""
    input_path = Path(run_name).expanduser()
    stripped_path = input_path.with_suffix("")

    # Accept absolute/relative paths under backup roots.
    for root in (CONFIGS_BACKUP_ROOT, MODELS_BACKUP_ROOT):
        absolute_root = root.resolve()
        for candidate in (stripped_path, (Path.cwd() / stripped_path)):
            try:
                return candidate.resolve().relative_to(absolute_root).as_posix()
            except ValueError:
                continue

    # If no known prefix is present, assume canonical run id was provided.
    return stripped_path.as_posix().lstrip("./")

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

def load_env_and_model(run_name: str, render_mode: str | None = "human", map_config: MapSourceConfigType | None = None) -> tuple[gym.Env, SAC]:
    normalized_run_name = normalize_run_name(run_name)
    experiment_config_path = CONFIGS_BACKUP_ROOT.joinpath(normalized_run_name).with_suffix(".yaml")
    model_path = MODELS_BACKUP_ROOT.joinpath(normalized_run_name).with_suffix(".zip")
    experiment_config = ExperimentConfig.load(experiment_config_path)

    # Override map source config if provided
    if map_config and experiment_config.population_config:
        experiment_config.population_config.map_source_config = map_config

    env, _ = load_env_from_config(experiment_config=experiment_config, render_mode=render_mode)

    if experiment_config.agent_config.algorithm == "SAC":
        model = SAC.load(model_path, env=env, device='auto')
    else:
        raise NotImplementedError

    return env, model