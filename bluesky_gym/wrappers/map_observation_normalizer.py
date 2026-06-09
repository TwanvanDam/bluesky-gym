import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MapObservationNormalizer(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, mode: str = "log", clip: bool = True) -> None:
        """
        mode: 'log' or 'min-max'
        clip: bool
        When clip is True the normalised value is clamped to [0, 1] (legacy behaviour). When clip is False
        the map source is expected to clip internally (e.g. via the Clip value transform in
        TransformedTiffMapSource).
        """
        super().__init__(env)

        if not hasattr(env, "map_source_max"):
            raise ValueError("Underlying environment must have map_source_max attribute for normalization")
        self.mode = mode

        self.clip = clip

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
        if not (np.isfinite(divisor) or divisor < 0):
            raise ValueError(
                f"map_source_max must be a finite positive number for normalization, got {divisor}. "
                "Check normalization_percentile — it must be in the range (0, 100]."
            )
        elif divisor == 0:
            return observation

        observation_copy = observation.copy()
        for key in list(observation_copy.keys()):
            if "map" in key:
                value = np.nan_to_num(observation_copy[key], nan=0.0)
                match self.mode:
                    case "log":
                        normalized = np.log1p(value) / np.log1p(divisor)
                    case "min-max" | "min_max":
                        normalized = value / divisor
                    case _:
                        raise NotImplementedError(f"Normalization mode {self.mode} is not supported.")

                if self.clip:
                    observation_copy[key] = np.clip(normalized, 0, 1)
                else:
                    max_val = float(normalized.max())
                    if max_val > 1.0 + 1e-6:
                        raise ValueError(
                            f"MapObservationNormalizer: map key '{key}' has a normalised value of "
                            f"{max_val:.4f} > 1 (map_source_max / divisor = {divisor:.4f}). "
                            "The map source must clip values before they reach the normalizer. "
                            "Remedy: enable clip_noise_reward (legacy sources) or use a TransformedTiffMapSource pipeline"
                        )
                    observation_copy[key] = normalized
        return observation_copy