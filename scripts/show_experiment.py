from pathlib import Path

from stable_baselines3 import SAC

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from bluesky_gym.wrappers.population import Population
from scripts.config import ExperimentConfig

def load_model_from_name(run_name: str):
    experiment_config_path = Path("scripts/common/results/configs_backup").joinpath(run_name).with_suffix(".yaml")
    model_path = Path("scripts/common/results/models_backup").joinpath(run_name).with_suffix(".zip")
    experiment_config = ExperimentConfig.load(experiment_config_path)

    env = BaseNavigationEnv(config=experiment_config.navigation_config)
    if experiment_config.population_config:
        env = Population(env, config=experiment_config.population_config)

    if experiment_config.training_config.algorithm == "SAC":
        model = SAC.load(model_path, env=env, device='auto')
    else:
        raise NotImplementedError
    return model, env

def render_experiment(run_name: str):
    model, env = load_model_from_name(run_name)
    env.render_mode = 'human'

    while True:
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated



if __name__ == '__main__':
    run_name = "BaseNavigationEnv-v0/2026-0~3"
    render_experiment(run_name)