import argparse
import datetime
import shutil
from pathlib import Path

import torch
from stable_baselines3 import SAC

from bluesky_gym.envs.common.environment_factory import build_env
from scripts.common.logger import TensorboardCallback
from scripts.common.run_paths import RunPaths, write_metadata, update_metadata
from scripts.config import ExperimentConfig, TrainingConfig
from scripts.feature_extractors import CombinedExtractor


def _generate_unique_run_name(experiment_config_path: Path, env_name: str, seed: int | None = None) -> str:
    base_name = experiment_config_path.stem
    if seed is not None:
        run_name = f"{base_name}_seed{seed:02d}"
    else:
        run_name = base_name
    suffix = 1

    while RunPaths.from_run_id(env_name, run_name).exists():
        run_name = f"{run_name}_{suffix:02d}"
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

    seed = experiment_config.training_config.seed if experiment_config.training_config else None

    if agent_config.algorithm == "SAC":
        model = SAC(
            agent_config.policy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "auto",
            policy_kwargs=policy_kwargs,
            seed=seed,
            learning_starts=10_000
        )
    else:
        raise NotImplementedError(f"Algorithm {experiment_config.agent_config.algorithm.algorithm} is not implemented.")

    print(f"Algorithm: {agent_config.algorithm}")
    print(f"Policy: {agent_config.policy}")
    print(f"Network Architecture: {model.policy}")
    print(f"Seed: {seed}")
    return model

def train_model(experiment_config_path: Path, slurm_job_id: str | None = None,
                slurm_log_out: str | None = None, slurm_log_err: str | None = None,
                seed: int | None = None):
    experiment_config = ExperimentConfig.load(experiment_config_path)

    if seed is not None:
        if experiment_config.training_config is None:
            experiment_config.training_config = TrainingConfig()
        experiment_config.training_config.seed = seed

    env, env_name = build_env(experiment_config)
    run_name = _generate_unique_run_name(experiment_config_path, env_name, seed=seed)
    experiment_config.run_name = run_name

    run_paths = RunPaths.from_run_id(env_name, run_name)
    run_paths.create_dirs()

    # Save config
    experiment_config.save(run_paths.config)

    # Write initial metadata
    metadata = {
        "run_name": run_name,
        "env_name": env_name,
        "config_stem": experiment_config_path.stem,
        "created_at": datetime.datetime.now().isoformat(),
        "status": "running",
    }
    if slurm_job_id:
        metadata["slurm_job_id"] = slurm_job_id
    write_metadata(run_paths, **metadata)

    model = initialize_agent(experiment_config, env, str(run_paths.tensorboard_dir))

    training_config = experiment_config.training_config
    print(f"Environment: {env_name}")
    print(f"Run directory: {run_paths.root}")
    print(f"Training timesteps: {training_config.total_timesteps}")

    model.learn(
        total_timesteps=training_config.total_timesteps,
        callback=TensorboardCallback(
            experiment_config=experiment_config,
            validation_env=env,
            save_frequency=experiment_config.training_config.save_frequency,
            save_dir=str(run_paths.checkpoints_dir),
        ),
        tb_log_name=run_name,
    )
    model.save(run_paths.model)
    update_metadata(run_paths, status="completed")

    # Copy SLURM logs into the run directory
    slurm_logs = []
    if slurm_log_out:
        slurm_logs.append(Path(slurm_log_out))
    if slurm_log_err:
        slurm_logs.append(Path(slurm_log_err))
    if slurm_job_id and not slurm_logs:
        slurm_logs = [
            Path(f"HPC/logs/out/slurm-{slurm_job_id}.out"),
            Path(f"HPC/logs/err/slurm-{slurm_job_id}.err"),
        ]
    for src in slurm_logs:
        if src.exists():
            shutil.copy2(src, run_paths.slurm_dir / src.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL model(s) from experiment config(s).")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to a single experiment YAML config.",
    )
    parser.add_argument("--slurm-job-id", default=None, help="SLURM job ID for log association.")
    parser.add_argument("--slurm-log-out", default=None, help="Path to SLURM stdout log file.")
    parser.add_argument("--slurm-log-err", default=None, help="Path to SLURM stderr log file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config).")
    args = parser.parse_args()

    if args.config:
        train_model(
            Path(args.config),
            slurm_job_id=args.slurm_job_id,
            slurm_log_out=args.slurm_log_out,
            slurm_log_err=args.slurm_log_err,
            seed=args.seed,
        )
    else:
        raise ValueError("No config path provided. Please provide a path to an experiment YAML config.")
