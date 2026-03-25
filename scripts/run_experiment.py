import argparse
import datetime
from pathlib import Path

import torch
from stable_baselines3 import SAC

from bluesky_gym.envs.common.environment_factory import load_env_from_config
from scripts.common.logger import TensorboardCallback
from scripts.config import ExperimentConfig
from scripts.feature_extractors import CombinedExtractor


def _run_name_exists(env_name: str, run_name: str) -> bool:
    return any(
        [
            (base_results_dir / "configs_backup" / env_name / f"{run_name}.yaml").exists(),
            (models_dir / env_name / f"{run_name}.zip").exists(),
            (base_results_dir / "checkpoints" / env_name / run_name).exists(),
        ]
    )


def _generate_unique_run_name(experiment_config_path: Path, env_name: str) -> str:
    # Include config stem and microseconds to keep names readable and unique in batch runs.
    base_name = f"{experiment_config_path.stem}_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')}"
    run_name = base_name
    suffix = 1

    while _run_name_exists(env_name, run_name):
        run_name = f"{base_name}_{suffix:02d}"
        suffix += 1

    return run_name

def initialize_agent(experiment_config: ExperimentConfig, env, log_dir: Path | str) -> SAC:
    agent_config = experiment_config.agent_config
    if not agent_config.feature_extractor:
        raise ValueError("Feature extractor config must be provided in the experiment config.")

    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": {"config": agent_config.feature_extractor},
        "net_arch" : agent_config.network_arch
    }

    if agent_config.algorithm == "SAC":
        model = SAC(
            agent_config.policy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "auto",
            policy_kwargs=policy_kwargs
        )
    else:
        raise NotImplementedError(f"Algorithm {experiment_config.agent_config.algorithm.algorithm} is not implemented.")

    print(f"Algorithm: {agent_config.algorithm}")
    print(f"Policy: {agent_config.policy}")
    print(f"Network Architecture: {model.policy}")
    return model

def train_model(experiment_config_path: Path):
    experiment_config = ExperimentConfig.load(experiment_config_path)

    env, env_name = load_env_from_config(experiment_config=experiment_config)
    experiment_config.run_name = _generate_unique_run_name(experiment_config_path, env_name)

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

    model = initialize_agent(experiment_config, env, log_dir)

    print(f"Environment: {env_name}")
    print(f"Training timesteps: {training_config.total_timesteps}")

    model.learn(
        total_timesteps=training_config.total_timesteps,
        callback=TensorboardCallback(
            experiment_config=experiment_config,
            validation_env=env,
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
        help="Path to a single experiment YAML config.",
    )
    args = parser.parse_args()

    base_results_dir = Path("scripts/common/results")
    logs_dir = base_results_dir / "logs_backup"
    models_dir = base_results_dir / "models_backup"

    if args.config:
        train_model(Path(args.config))
    else:
        raise ValueError("No config path provided. Please provide a path to an experiment YAML config.")