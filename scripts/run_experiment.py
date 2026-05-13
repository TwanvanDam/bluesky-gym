import argparse
import datetime
import shutil
import warnings
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from bluesky_gym.envs.common.environment_factory import build_env
from scripts.common.logger import BestModelCallback, TensorboardCallback
from scripts.common.run_paths import RunPaths, read_metadata, resolve_run, update_metadata, write_metadata
from scripts.config import ExperimentConfig, TrainingConfig
from scripts.feature_extractors import CombinedExtractor

# supress Pygame warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

def _get_env_name(experiment_config: ExperimentConfig) -> str:
    if experiment_config.population_config:
        return "PopulationWrapper-v0"
    return "BaseNavigationEnv-v0"


def _make_env(experiment_config: ExperimentConfig):
    def _init() -> gym.Env:
        env, _ = build_env(experiment_config)
        return Monitor(env)
    return _init


def _build_vec_env(experiment_config: ExperimentConfig, n_envs: int):
    fns = [_make_env(experiment_config) for _ in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns)


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
        "net_arch": agent_config.network_arch
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
            learning_rate=agent_config.learning_rate,
            batch_size=agent_config.batch_size,
            buffer_size=agent_config.buffer_size,
            learning_starts=agent_config.learning_starts,
            gamma=agent_config.gamma,
            tau=agent_config.tau
        )
    else:
        raise NotImplementedError(f"Algorithm {experiment_config.agent_config.algorithm.algorithm} is not implemented.")

    print(f"Algorithm: {agent_config.algorithm}")
    print(f"Policy: {agent_config.policy}")
    print(f"Network Architecture: {model.policy}")
    print(f"Seed: {seed}")
    return model


def _build_callbacks(experiment_config: ExperimentConfig, run_paths: RunPaths) -> CallbackList:
    training_config = experiment_config.training_config
    eval_freq = training_config.save_frequency
    return CallbackList([
        TensorboardCallback(experiment_config=experiment_config),
        CheckpointCallback(
            save_freq=eval_freq,
            save_path=str(run_paths.checkpoints_dir),
            name_prefix="checkpoint",
            save_replay_buffer=training_config.save_replay_buffer,
            verbose=1,
        ),
        EveryNTimesteps(
            n_steps=eval_freq,
            callback=BestModelCallback(
                save_path=run_paths.best_model,
                run_paths=run_paths,
                n_episodes_window=training_config.n_eval_episodes,
            ),
        ),
    ])


def _copy_slurm_logs(
        run_paths: RunPaths,
        slurm_job_id: str | None,
        slurm_log_out: str | None,
        slurm_log_err: str | None,
) -> None:
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


def train_model(
        experiment_config_path: Path,
        slurm_job_id: str | None = None,
        slurm_log_out: str | None = None,
        slurm_log_err: str | None = None,
        seed: int | None = None,
) -> None:
    experiment_config = ExperimentConfig.load(experiment_config_path)

    if seed is not None:
        if experiment_config.training_config is None:
            experiment_config.training_config = TrainingConfig()
        experiment_config.training_config.seed = seed

    n_envs = experiment_config.training_config.n_envs if experiment_config.training_config else 1
    env_name = _get_env_name(experiment_config)
    env = _build_vec_env(experiment_config, n_envs)
    run_name = _generate_unique_run_name(experiment_config_path, env_name, seed=seed)
    experiment_config.run_name = run_name

    run_paths = RunPaths.from_run_id(env_name, run_name)
    run_paths.create_dirs()

    experiment_config.save(run_paths.config)

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
        callback=_build_callbacks(experiment_config, run_paths),
        tb_log_name=run_name,
    )
    model.save(run_paths.final_model)
    update_metadata(run_paths, status="completed")
    _copy_slurm_logs(run_paths, slurm_job_id, slurm_log_out, slurm_log_err)


def resume_training(
        run_ref: str,
        slurm_job_id: str | None = None,
        slurm_log_out: str | None = None,
        slurm_log_err: str | None = None,
) -> None:
    run_paths = resolve_run(run_ref)
    config = ExperimentConfig.load(run_paths.config)
    training_config = config.training_config

    latest_ckpt = run_paths.latest_checkpoint()
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for run {run_paths.run_id}.")

    print(f"Resuming run: {run_paths.run_id}")
    print(f"Loading checkpoint: {latest_ckpt}")

    n_envs = training_config.n_envs if training_config else 1
    env = _build_vec_env(config, n_envs)

    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": {"config": config.agent_config.feature_extractor},
        "net_arch": config.agent_config.network_arch,
    }
    model = SAC.load(
        latest_ckpt,
        env=env,
        device="cuda" if torch.cuda.is_available() else "auto",
        custom_objects={"policy_kwargs": policy_kwargs},
        verbose=1,
    )

    if training_config.save_replay_buffer:
        replay_buffer_path = latest_ckpt.with_name(latest_ckpt.stem + "_replay_buffer.pkl")
        if replay_buffer_path.exists():
            model.load_replay_buffer(str(replay_buffer_path))
            print(f"Replay buffer loaded from: {replay_buffer_path}")
        else:
            warnings.warn(
                f"save_replay_buffer is True but no replay buffer found at {replay_buffer_path}. "
                "Training will continue without it."
            )

    metadata = read_metadata(run_paths)
    resumes = metadata.get("resumes", [])
    resumes.append({
        "resumed_at": datetime.datetime.now().isoformat(),
        "from_checkpoint": latest_ckpt.name,
    })
    update_metadata(run_paths, resumes=resumes, status="running")
    if slurm_job_id:
        update_metadata(run_paths, slurm_job_id=slurm_job_id)

    model.learn(
        total_timesteps=training_config.total_timesteps,
        callback=_build_callbacks(config, run_paths),
        tb_log_name=run_paths.run_name,
        reset_num_timesteps=False,
    )
    model.save(run_paths.final_model)
    update_metadata(run_paths, status="completed")
    _copy_slurm_logs(run_paths, slurm_job_id, slurm_log_out, slurm_log_err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL model(s) from experiment config(s).")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to a single experiment YAML config.",
    )
    parser.add_argument("--resume", default=None, metavar="RUN_REF",
                        help="Resume training from the latest checkpoint of an existing run.")
    parser.add_argument("--slurm-job-id", default=None, help="SLURM job ID for log association.")
    parser.add_argument("--slurm-log-out", default=None, help="Path to SLURM stdout log file.")
    parser.add_argument("--slurm-log-err", default=None, help="Path to SLURM stderr log file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config).")
    args = parser.parse_args()

    if args.resume:
        resume_training(
            args.resume,
            slurm_job_id=args.slurm_job_id,
            slurm_log_out=args.slurm_log_out,
            slurm_log_err=args.slurm_log_err,
        )
    elif args.config:
        train_model(
            Path(args.config),
            slurm_job_id=args.slurm_job_id,
            slurm_log_out=args.slurm_log_out,
            slurm_log_err=args.slurm_log_err,
            seed=args.seed,
        )
    else:
        raise ValueError("Provide either a config path or --resume <run_ref>.")
