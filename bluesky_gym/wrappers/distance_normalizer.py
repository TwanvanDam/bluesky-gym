import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DistanceNormalization(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, normalization_factor: float = 150_000):
        super().__init__(env)
        assert isinstance(env.observation_space,
                          spaces.Dict), "DistanceNormalization only works with Dict observation spaces"
        self.normalization_factor = normalization_factor
        self.transform_keys = ["destination_ground_distance"]

    def observation(self, observation: dict) -> dict:
        new_observation = observation.copy()
        for key in self.transform_keys:
            if key in new_observation:
                value = new_observation.pop(key)[0]
                new_observation[f"{key}"] = np.array([value / self.normalization_factor])
        return new_observation