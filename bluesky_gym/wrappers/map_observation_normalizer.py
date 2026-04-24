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
        divisor = self.env.map_source_max
        if not (np.isfinite(divisor) and divisor > 0):
            raise ValueError(
                f"map_source_max must be a finite positive number for normalization, got {divisor}. "
                "Check normalization_percentile — it must be in the range (0, 100]."
            )
        observation_copy = observation.copy()
        for key in list(observation_copy.keys()):
            if "map" in key:
                value = np.nan_to_num(observation_copy[key], nan=0.0)
                match self.mode:
                    case "log":
                        observation_copy[key] = np.clip(np.log1p(value) / np.log1p(divisor), 0, 1)
                    case "min-max" | "min_max":
                        observation_copy[key] = np.clip(value / divisor, 0, 1)
                    case _:
                        raise NotImplementedError(f"Normalization mode {self.mode} is not supported.")
        return observation_copy