from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from affine import Affine
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from bluesky_gym.wrappers.map_datsets import TiffMapSource, RandomMapSource
from bluesky_gym.wrappers.population import Population
from bluesky_gym.wrappers.random_map_generators import generate_random_shapes_map
from scripts.config import ExperimentConfig, MapSourceConfig, PopulationConfig





if __name__ == "__main__":
    experiment_config = ExperimentConfig.load(Path("./scripts/common/results/configs_backup/PopulationWrapper-v0/TestMapConfig.yaml"))

    env = BaseNavigationEnv(config = experiment_config.navigation_config, render_mode="human")
    wrapped = Population(env, experiment_config.population_config)

    while True:
        obs, info = wrapped.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped.step(action)
            done = terminated or truncated
            # print(wrapped.population_observation.min(), wrapped.population_observation.max())
            # print(f"background", wrapped.background_map.min(), wrapped.background_map.max())