import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Population(gym.Wrapper):
    def __init__(self, env, shape: tuple[int, int]):
        super().__init__(env)
        self.shape = shape
        self.static_map = np.zeros(self.shape)

        assert isinstance(self.env.observation_space, spaces.Dict)
        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            "population_map": spaces.Box(low=0, high=np.inf, shape=self.shape, dtype=np.float64)
        })

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = {**observation, "population_map": self.static_map}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        observation = {**observation, "population_map": self.static_map}
        return observation, info
