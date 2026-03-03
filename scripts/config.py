from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.random
import pyrallis
from dataclasses import dataclass, field

from bluesky_gym.wrappers.map_datsets import MapSource
from bluesky_gym.wrappers.random_map_generators import generate_cities, generate_random_shapes_map


@dataclass
class SamplingConfig:
    distribution: str = "fixed"

    # Uniform distribution
    low: Optional[float] = None
    high: Optional[float] = None

    # Normal Distribution
    mean: Optional[float] = None
    std: Optional[float] = None

    # Fixed
    value: Optional[float] = None

    def sample(self, rng: numpy.random.Generator) -> float:
        if self.distribution == "fixed":
            return self.value
        elif self.distribution == "uniform":
            return float(rng.uniform(self.low, self.high))
        elif self.distribution == "normal":
            return float(rng.normal(self.mean, self.std))
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

@dataclass
class NavigationConfig:
    ac_name: str = "KL001"
    ac_type: str = "a320"
    ac_initial_spd: int = 200  # [ kts ] (input to cre(), stored internally as m/s)
    ac_initial_alt: int = 3_000  # [ ft ] (input to cre(), stored internally as m)

    # Simulation bounds  [ degrees (WGS84) ]
    lon_min: float = 3.0
    lon_max: float = 7.5
    lat_min: float = 50.5
    lat_max: float = 54.0

    # Simulation Parameters
    max_steps: int = 250
    sim_dt: int = 3  # s
    action_time: int = 60  # s

    # Termination conditions
    faf_distance: float = 25  # km
    iaf_angle: float = 60  # degrees
    iaf_distance: float = 30  # km

    # Initial Conditions [ degrees (WGS84) ]
    airport_lat_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig("fixed", value=52.31))
    airport_lon_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig("fixed", value=4.7))
    airport_hdg_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig("uniform", low=0, high=360))
    aircraft_lat_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig("normal", mean=52.31, std=1))
    aircraft_lon_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig("normal", mean=4.7, std=1))

    pygame_crs: str = "EPSG:28992"
    use_sin_cos_obs: Optional[bool] = False

    # Rewards
    constraint_violation_reward: Optional[float] = -1
    successful_approach_reward: Optional[float] = 50
    fuel_coeff: Optional[float] = 0.025


@dataclass
class TrainingConfig:
    algorithm: str = "SAC"
    policy: str = "MultiInputPolicy"
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    total_timesteps: int = 1_000_000
    validation_episodes: Optional[int] = 10_000

@dataclass
class MapSourceConfig:
    type: str = "tiff" # "tiff", "polygon", "cities"
    file_path: Optional[Path] = None
    kwargs: Optional[dict] = None

    def build(self, env) -> "MapSource":
        from bluesky_gym.wrappers.map_datsets import TiffMapSource, RandomMapSource

        if self.type == "tiff":
            return TiffMapSource(self.file_path)
        elif self.type == "cities":
            generator = generate_cities
        elif self.type == "polygon":
            print(self.kwargs)
            generator = partial(generate_random_shapes_map, **self.kwargs)
        else:
            raise NotImplementedError(f"type {self.type} is not implemented")
        return RandomMapSource.from_env_bounds(env=env, random_map_generator=generator)

@dataclass
class PopulationConfig:
    observation_shape: tuple[int, int] = (64, 64)
    observation_range: tuple[int, int] = (100_000, 100_000)
    noise_penalty_coefficient: float = 0.035
    fuel_to_noise_ratio: float = 0.5  # Equal weighting of fuel and noise
    noise_resolution: int = 1_000
    noise_base: float = 85 # dBA
    noise_cutoff: float = 55 # dBA
    resampling: str = "cubic_spline"
    normalization: str = "log" # [none, min_max, log]
    map_source_config: MapSourceConfig = field(default_factory=lambda: MapSourceConfig())



@dataclass
class ExperimentConfig:
    navigation_config: NavigationConfig = field(default_factory=NavigationConfig)
    training_config: Optional[TrainingConfig] = None
    population_config: Optional[PopulationConfig] = None
    run_name: Optional[str] = None

    def save(self, path: str | Path) -> None:
        pyrallis.dump(self, open(path, "w"))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return pyrallis.parse(config_class=cls, config_path=path)