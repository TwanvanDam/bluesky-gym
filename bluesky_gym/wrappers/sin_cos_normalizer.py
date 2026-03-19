import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SinCosNormalization(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Check if underlying observation space is Dict
        assert isinstance(env.observation_space,
                          spaces.Dict), "SinCosNormalization only works with Dict observation spaces"

        self.transform_keys = ['destination_relative_heading', 'destination_relative_orientation']

        # Determine the new observation space
        new_spaces = env.observation_space.spaces.copy()

        for key in self.transform_keys:
            if key in new_spaces:
                new_spaces.pop(key)
                new_spaces[f"{key}_sin"] = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
                new_spaces[f"{key}_cos"] = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, observation: dict) -> dict:
        new_observation = observation.copy()
        for key in self.transform_keys:
            if key in new_observation:
                value = new_observation.pop(key)[0]
                new_observation[f"{key}_sin"] = np.array([np.sin(np.deg2rad(value))], dtype=np.float64)
                new_observation[f"{key}_cos"] = np.array([np.cos(np.deg2rad(value))], dtype=np.float64)
        return new_observation
