from pathlib import Path
from typing import Literal

import numpy as np
import numpy.random
import pyrallis
from dataclasses import dataclass, field

from bluesky_gym.wrappers.map_datsets import MapSource

@dataclass
class SamplingConfig:
    distribution: str = "fixed"

    # Uniform distribution
    low: float = np.nan
    high: float = np.nan

    # Normal Distribution
    mean: float = np.nan
    std: float = np.nan

    # Fixed
    value: float = np.nan
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
    ac_initial_spd: int = 200  # [ m/s ]
    ac_initial_alt: int = 3_000  # [ m ]

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

    pygame_crs: str = "EPSG:3035"


@dataclass
class TrainingConfig:
    algorithm: str = "SAC"
    policy: str = "MultiInputPolicy"
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    total_timesteps: int = 1_000_000
    device: str = "auto"

@dataclass
class MapSourceConfig:
    type: str = "tiff" # "tiff" or to be implemented types
    file_path: Path = Path()

    def build(self, env) -> "MapSource":
        from bluesky_gym.wrappers.map_datsets import TiffMapSource, RandomMapSource

        if self.type == "tiff":
            return TiffMapSource(self.file_path)
        else:
            raise NotImplementedError

@dataclass
class PopulationConfig:
    observation_shape: tuple[int, int] = (64, 64)
    observation_range: tuple[int, int] = (100_000, 100_000)
    noise_penalty_coefficient: float = 0.0
    noise_radius_shape: str = "box"
    resampling: str = "cubic_spline"
    normalization: str = "log" # [None, fixed, min/max, log]
    map_source_config: MapSourceConfig = field(default_factory=lambda: MapSourceConfig())



@dataclass
class ExperimentConfig:
    navigation_config: NavigationConfig = field(default_factory=NavigationConfig)
    training_config: TrainingConfig = None
    population_config: PopulationConfig = None
    seed: int = 42

    def save(self, path: str | Path) -> None:
        pyrallis.dump(self, open(path, "w"))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return pyrallis.parse(config_class=cls, config_path=path)





if __name__ == "__main__":
    test = ExperimentConfig()
    test.save(Path("./test.yaml"))
    # test = ExperimentConfig.load(Path("./test.yaml"))
    print(test)