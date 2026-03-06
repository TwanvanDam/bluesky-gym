from pathlib import Path
from typing import Optional, List, Tuple, Union, Any, Dict

import numpy as np
import yaml
from pydantic import BaseModel, Field, ConfigDict


class SamplingConfig(BaseModel):
    distribution: str  # "fixed", "normal" or "uniform"
    low: Optional[float] = None
    high: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    value: Optional[float] = None

    def sample(self, rng: np.random.Generator) -> float:
        if self.distribution == "fixed":
            return self.value
        elif self.distribution == "uniform":
            return float(rng.uniform(self.low, self.high))
        elif self.distribution == "normal":
            return float(rng.normal(self.mean, self.std))
        raise ValueError(f"Unknown distribution {self.distribution}")


class NavigationConfig(BaseModel):
    ac_name: str = "KL001"
    ac_type: str = "a320"
    ac_initial_spd: int = 200  # [ m / s ]
    ac_initial_alt: int = 3_000  # [ m ]

    # All coordinates in degrees (WGS84)
    lon_min: float = 3.0
    lon_max: float = 7.5
    lat_min: float = 50.5
    lat_max: float = 54.0

    max_steps: int = 250
    sim_dt: int = 3  # [ s ]
    action_time: int = 60  # [ s ]
    faf_distance: float = 25  # [ km ]
    iaf_angle: float = 60  # [ degrees ]
    iaf_distance: float = 30  # [ km ]

    # Nested sampling configs with default factories
    airport_lat_sampling: SamplingConfig = Field(
        default_factory=lambda: SamplingConfig(distribution="fixed", value=52.31))
    airport_lon_sampling: SamplingConfig = Field(
        default_factory=lambda: SamplingConfig(distribution="fixed", value=4.7))
    airport_hdg_sampling: SamplingConfig = Field(
        default_factory=lambda: SamplingConfig(distribution="uniform", low=0, high=360))
    aircraft_lat_sampling: SamplingConfig = Field(
        default_factory=lambda: SamplingConfig(distribution="normal", mean=52.31, std=1))
    aircraft_lon_sampling: SamplingConfig = Field(
        default_factory=lambda: SamplingConfig(distribution="normal", mean=4.7, std=1))

    pygame_crs: str = "EPSG:3035"
    use_sin_cos_obs: bool = False
    normalize_distance_obs: bool = True
    constraint_violation_reward: float = -1.0
    successful_approach_reward: float = 50.0
    mean_episode_length: float = 20 * 60  # [ s ]
    total_dense_rewards: float = 0.25  # Summed dense reward on average


class TrainingConfig(BaseModel):
    algorithm: str = "SAC"
    policy: str = "MultiInputPolicy"
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    total_timesteps: int = 1_000_000
    validation_episodes: Optional[int] = 10_000
    save_frequency: Optional[int] = 50_000


class ConvolutionLayerConfig(BaseModel):
    in_channels: Optional[int] = None
    out_channels: int
    kernel_size: int
    stride: int
    padding: int


class PoolingLayerConfig(BaseModel):
    type: str  # "max", "avg"
    kernel_size: int
    stride: int
    padding: int


class LayerBlockConfig(BaseModel):
    conv: Optional[ConvolutionLayerConfig] = None
    pooling: Optional[PoolingLayerConfig] = None
    activation: Optional[str] = None  # "ReLU", "Tanh", "Sigmoid"


class FeatureExtractorConfig(BaseModel):
    layers: List[LayerBlockConfig] = Field(default_factory=list)


class MapSourceConfig(BaseModel):
    type: str = "polygon"  # "tiff", "polygon" or "cities"
    file_path: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def build(self, env):
        from bluesky_gym.wrappers.map_datasets import TiffMapSource, RandomMapSource
        from bluesky_gym.wrappers.random_map_generators import generate_cities, generate_random_shapes_map
        import functools

        if self.type == "tiff":
            if not self.file_path:
                raise ValueError("file_path is required for tiff map source")
            if self.kwargs: raise ValueError(f"MapSource {self.type} does not support kwargs")
            return TiffMapSource(str(self.file_path))  # Convert Path to str if needed
        elif self.type == "cities":
            # Use kwargs to configure the generator if any
            generator = generate_cities
            if self.kwargs:
                generator = functools.partial(generate_cities, **self.kwargs)
            return RandomMapSource.from_env_bounds(env, generator)
        elif self.type == "polygon":
            generator = generate_random_shapes_map
            if self.kwargs:
                generator = functools.partial(generate_random_shapes_map, **self.kwargs)
            return RandomMapSource.from_env_bounds(env, generator)
        else:
            raise ValueError(f"Unknown map source type: {self.type}")


class PopulationConfig(BaseModel):
    observation_shape: List[Tuple[int, int]] = Field(default_factory=lambda: [(64, 64)])  # [px, px]
    observation_range: List[Tuple[int, int]] = Field(default_factory=lambda: [(100_000, 100_000)])  # [m, m]
    noise_penalty_coefficient: float = 1 / (20 * 60 * 2)  # Expected episode duration 20 minutes.
    fuel_to_noise_ratio: float = 0.5
    noise_resolution: int = 1_000  # [ m ]
    noise_base: float = 85  # [ dBA ]
    noise_cutoff: float = 55  # [ dBA ]
    resampling: str = "cubic_spline"
    rendering_normalization: str = "log"  # "log" or "min-max"
    observation_normalization: str = "log"
    map_source_config: MapSourceConfig = Field(default_factory=MapSourceConfig)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    navigation_config: NavigationConfig = Field(default_factory=NavigationConfig)
    training_config: Optional[TrainingConfig] = None
    population_config: Optional[PopulationConfig] = None
    feature_extractor: Optional[FeatureExtractorConfig] = None
    run_name: Optional[str] = None

    def save(self, path: Union[str, Path]) -> None:
        def tuple_representer(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data)

        yaml.add_representer(tuple, tuple_representer)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


if __name__ == '__main__':
    print(ExperimentConfig.load(Path("scripts/common/results/configs_backup/PopulationWrapper-v0/TestMapConfig.yaml")))
