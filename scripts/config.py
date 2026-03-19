from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, ConfigDict

from bluesky_gym.envs.base_navigation_env import NavigationConfig
from bluesky_gym.wrappers.population import PopulationConfig
from scripts.feature_extractors import FeatureExtractorConfig

class TrainingConfig(BaseModel):
    algorithm: str = "SAC"
    policy: str = "MultiInputPolicy"
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    total_timesteps: int = 1_000_000
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
    fuel_to_noise_ratio: float = 0.5
    noise_resolution: int = 1_000  # [ m ]
    noise_base: float = 85  # [ dBA ]
    noise_cutoff: float = 55  # [ dBA ]
    resampling: str = "cubic_spline"
    rendering_normalization: str = "log"  # "log" or "min_max"
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
    def load(cls, path: Union[str, Path]) -> ExperimentConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test loading an experiment config")
    parser.add_argument("config_path", type=str, help="Path to the experiment config YAML file to load and validate")
    args = parser.parse_args()

    dummy = ExperimentConfig.load(args.config_path)
    print(f"Successfully loaded config: {args.config_path}")
