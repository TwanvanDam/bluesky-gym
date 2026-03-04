import argparse
from pathlib import Path
from stable_baselines3 import SAC
from scripts.config import ExperimentConfig
from scripts.run_experiment import load_env_from_config


def render_experiment(run_name: str):
    experiment_config_path = Path("scripts/common/results/configs_backup").joinpath(run_name).with_suffix(".yaml")
    model_path = Path("scripts/common/results/models_backup").joinpath(run_name).with_suffix(".zip")
    experiment_config = ExperimentConfig.load(experiment_config_path)

    env, _ = load_env_from_config(experiment_config=experiment_config, render_mode="human")

    if experiment_config.training_config.algorithm == "SAC":
        model = SAC.load(model_path, env=env, device='auto')
    else:
        raise NotImplementedError

    while True:
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated



if __name__ == '__main__':
    run_name = "BaseNavigationEnv-v0/2026-0~3"
    parser = argparse.ArgumentParser(description="Show trained RL model(s) from experiment config(s).")
    parser.add_argument(
        "name",
        nargs="?",
        default=run_name,
        help=f"Name of a single experiment run. If omitted, {run_name} is used.",
    )
    args = parser.parse_args()

    render_experiment(args.name)