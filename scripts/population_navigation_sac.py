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


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_config: dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if len(subspace.shape) == 2:
                h, w = subspace.shape
                n_layers = len(cnn_config["in_channels"])
                layers = []
                for i in range(n_layers):
                    layers.append(nn.Conv2d(
                        in_channels=cnn_config["in_channels"][i],
                        out_channels=cnn_config["out_channels"][i],
                        kernel_size=cnn_config["kernel_size"][i],
                        stride=cnn_config["stride"][i],
                        padding=cnn_config["padding"][i],
                    ))
                    layers.append(nn.ReLU())
                layers.append(nn.Flatten())

                cnn = nn.Sequential(*layers)
                # Compute CNN output size
                with torch.no_grad():
                    n_flatten = cnn(torch.zeros(1, 1, h, w)).shape[1]

                extractors[key] = nn.Sequential(
                    cnn,
                    nn.Linear(n_flatten, cnn_config["output_dim"]),
                    nn.ReLU(),
                )
                total_concat_size += cnn_config["output_dim"]
            else:
                total_concat_size += subspace.shape[0]
                extractors[key] = nn.Flatten()

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: dict) -> torch.Tensor:
        encoded = []
        for key, extractor in self.extractors.items():
            obs = observations[key].float()
            if obs.dim() == 3:
                # (batch, H, W) → (batch, 1, H, W) to add channel dim for Conv2d
                obs = obs.unsqueeze(1)
            encoded.append(extractor(obs))
        return torch.cat(encoded, dim=1)


if __name__ == "__main__":
    experiment_config = ExperimentConfig.load(Path("./scripts/common/results/models_backup/BaseNavigationEnv-v0/New_model_longer_trained.yaml"))

    env = BaseNavigationEnv(config = experiment_config.navigation_config, render_mode="human")
    wrapped = Population(env, experiment_config.population_config)

    MODEL_PATH = Path("./scripts/common/results/models_backup/BaseNavigationEnv-v0/New_model_longer_trained")
    model = SAC.load(MODEL_PATH, env=wrapped)
    experiment_config.save(Path("./scripts/common/results/models_backup/BaseNavigationEnv-v0/New_model_longer_trained.yaml"))

    while True:
        obs, info = wrapped.reset()
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            done = terminated or truncated

    # model = SAC("MultiInputPolicy", wrapped, policy_kwargs=policy_kwargs, verbose=1, device="cuda")
    # policy_kwargs = dict(
    #     features_extractor_class=CombinedExtractor,
    #     features_extractor_kwargs=dict(cnn_config={"in_channels" : [1, 16], "out_channels": [16, 32], "kernel_size": [3,3], "stride": [2,2], "padding" : [1,1], "output_dim" : 64}),
    # )
