import argparse
import datetime
from pathlib import Path

import torch.cuda
from stable_baselines3 import SAC

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from bluesky_gym.wrappers.population import Population
from scripts.common.logger import TensorboardCallback
from scripts.config import ExperimentConfig

def train_model(experiment_config_path: Path):
    experiment_config = ExperimentConfig.load(experiment_config_path)
    experiment_config.run_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    # Initialize Environment
    env = BaseNavigationEnv(config=experiment_config.navigation_config)
    if experiment_config.population_config:
        env_name = "PopulationWrapper-v0"
        env = Population(env, config=experiment_config.population_config)
    else:
        env_name = "BaseNavigationEnv-v0"


    training_config = experiment_config.training_config
    log_dir = logs_dir.joinpath(env_name)
    run_dir = models_dir.joinpath(env_name, experiment_config.run_name).with_suffix(".zip")

    # Save config backup
    configs_backup_dir = base_results_dir / "configs_backup" / env_name
    configs_backup_dir.mkdir(parents=True, exist_ok=True)
    experiment_config.save(configs_backup_dir / f"{experiment_config.run_name}.yaml")

    # Initialize Model
    if training_config.algorithm == "SAC":
        model = SAC(training_config.policy, env, verbose=1, tensorboard_log=log_dir, device="cuda" if torch.cuda.is_available() else "auto")
    else:
        raise NotImplementedError
    model.learn(
        total_timesteps=training_config.total_timesteps,
        callback=TensorboardCallback(experiment_config=experiment_config, validation_env=env),
        tb_log_name=experiment_config.run_name,
    )
    model.save(run_dir)

def show_model(run_name: str):


    model = SAC.load()



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
