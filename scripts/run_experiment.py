import argparse
import datetime
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import SAC

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from bluesky_gym.wrappers.distance_normalizer import DistanceNormalization
from bluesky_gym.wrappers.map_observation_normalizer import MapObservationNormalizer
from bluesky_gym.wrappers.population import Population
from bluesky_gym.wrappers.sin_cos_normalizer import SinCosNormalization

from scripts.common.logger import TensorboardCallback
from scripts.config import ExperimentConfig
from scripts.feature_extractors import CombinedExtractor

def load_env_from_config(experiment_config: ExperimentConfig, render_mode: str | None = None) -> tuple[gym.Env, str]:
    env = BaseNavigationEnv(config=experiment_config.navigation_config, render_mode=render_mode)

    if experiment_config.navigation_config.use_sin_cos_obs:
        env = SinCosNormalization(env)

    if experiment_config.navigation_config.normalize_distance_obs:
        env = DistanceNormalization(env)

    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    if experiment_config.population_config:
        env = Population(env, config=experiment_config.population_config)
        if experiment_config.population_config.observation_normalization:
            env = MapObservationNormalizer(env, mode=experiment_config.population_config.observation_normalization)
        env_name = "PopulationWrapper-v0"
    else:
        env_name = "BaseNavigationEnv-v0"

    return env, env_name

def train_model(experiment_config_path: Path):
    experiment_config = ExperimentConfig.load(experiment_config_path)
    experiment_config.run_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    env, env_name = load_env_from_config(experiment_config=experiment_config)

    training_config = experiment_config.training_config
    log_dir = logs_dir.joinpath(env_name)
    run_dir = models_dir.joinpath(env_name, experiment_config.run_name).with_suffix(".zip")

    # Save config backup
    configs_backup_dir = base_results_dir / "configs_backup" / env_name
    configs_backup_dir.mkdir(parents=True, exist_ok=True)
    experiment_config.save(configs_backup_dir / f"{experiment_config.run_name}.yaml")

    # Create checkpoints directory
    checkpoints_dir = base_results_dir / "checkpoints" / env_name / experiment_config.run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Model
    policy_kwargs = None
    if hasattr(experiment_config, 'feature_extractor_config') and experiment_config.feature_extractor_config:
        from functools import partial
        policy_kwargs = {
            "features_extractor_class": CombinedExtractor,
            "features_extractor_kwargs": {"config": experiment_config.feature_extractor_config}
        }

    if training_config.algorithm == "SAC":
        model = SAC(
            training_config.policy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "auto",
            policy_kwargs=policy_kwargs
        )
    else:
        raise NotImplementedError
    model.learn(
        total_timesteps=training_config.total_timesteps,
        callback=TensorboardCallback(
            experiment_config=experiment_config,
            validation_env=env,
            plot_frequency=experiment_config.training_config.validation_episodes,
            save_frequency=experiment_config.training_config.save_frequency,
            save_dir=str(checkpoints_dir),
        ),
        tb_log_name=experiment_config.run_name,
    )
    model.save(run_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL model(s) from experiment config(s).")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to a single experiment YAML config. If omitted, all configs in HPC/experiments/ are run.",
    )
    args = parser.parse_args()

    base_results_dir = Path("scripts/common/results")
    logs_dir = base_results_dir / "logs_backup"
    models_dir = base_results_dir / "models_backup"

    if args.config:
        train_model(Path(args.config))
    else:
        experiments_dir = Path("HPC/experiments")
        for experiment_config_path in experiments_dir.iterdir():
            train_model(experiment_config_path)
