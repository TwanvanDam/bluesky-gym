import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MapObservationNormalizer(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, mode: str = "log") -> None:
        super().__init__(env)

        if not hasattr(env, "map_source_max"):
            raise ValueError("Underlying environment must have map_source_max attribute for normalization")
        self.mode = mode

        # Check if underlying observation space is Dict
        assert isinstance(env.observation_space,
                          spaces.Dict), "MapObservationNormalizer only works with Dict observation spaces"
        observation_space = env.observation_space.spaces.copy()
        for key in list(observation_space.keys()):
            if "map" in key:
                original_space = observation_space.pop(key)
                observation_space[key] = spaces.Box(low=0, high=1, shape=original_space.shape,
                                                    dtype=original_space.dtype)

        self.observation_space = spaces.Dict(observation_space)

    def observation(self, observation) -> dict:
        observation_copy = observation.copy()
        for key in list(observation_copy.keys()):
            if "map" in key:
                value = observation_copy[key]
                if not np.min(value) == np.max(value):  # Avoid normalizing if all values are the same (e.g., all zeros)
                    match self.mode:
                        case "log":
                            observation_copy[key] = np.clip(np.log1p(value) / np.log1p(self.env.map_source_max), 0, 1)
                        case "min-max" | "min_max":
                            observation_copy[key] = np.clip(value / self.env.map_source_max, 0, 1)
                        case _:
                            msg = f"Normalization mode {self.mode} is not supported."
                            raise NotImplementedError(msg)
        return observation_copy