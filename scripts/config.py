from pathlib import Path
from typing import Optional, List, Tuple, Union, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import yaml

class SamplingConfig(BaseModel):
    distribution: str # "fixed", "normal" or "uniform"
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
    ac_initial_spd: int = 200 # [ m / s ]
    ac_initial_alt: int = 3_000 # [ m ]

    # All coordinates in degrees (WGS84)
    lon_min: float = 3.0
    lon_max: float = 7.5
    lat_min: float = 50.5
    lat_max: float = 54.0

    max_steps: int = 250
    sim_dt: int = 3 # [ s ]
    action_time: int = 60 # [ s ]
    faf_distance: float = 25 # [ km ]
    iaf_angle: float = 60 # [ degrees ]
    iaf_distance: float = 30 # [ km ]

    # Nested sampling configs with default factories
    airport_lat_sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig(distribution="fixed", value=52.31))
    airport_lon_sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig(distribution="fixed", value=4.7))
    airport_hdg_sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig(distribution="uniform", low=0, high=360))
    aircraft_lat_sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig(distribution="normal", mean=52.31, std=1))
    aircraft_lon_sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig(distribution="normal", mean=4.7, std=1))

    pygame_crs: str = "EPSG:3035"
    use_sin_cos_obs: bool = False
    normalize_distance_obs: bool = True
    constraint_violation_reward: float = -1.0
    successful_approach_reward: float = 50.0
    fuel_coeff: float = 0.025

class TrainingConfig(BaseModel):
    algorithm: str = "SAC"
    policy: str = "MultiInputPolicy"
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    total_timesteps: int = 1_000_000
    validation_episodes: Optional[int] = 10_000

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
    activation: Optional[str] = None # "ReLU", "Tanh", "Sigmoid"

class FeatureExtractorConfig(BaseModel):
    layers: List[LayerBlockConfig] = Field(default_factory=list)
    output_dim: int

class MapSourceConfig(BaseModel):
    type: str # "tiff" or "random"
    file_path: Optional[Path] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class PopulationConfig(BaseModel):
    observation_shape: Tuple[int, int] = (64, 64) # [px, px]
    observation_range: Tuple[int, int] = (100_000, 100_000) # [m, m]
    noise_penalty_coefficient: float = 0.035
    fuel_to_noise_ratio: float = 0.5
    noise_resolution: int = 1_000 # [ m ]
    noise_base: float = 85 # [ dBA ]
    noise_cutoff: float = 55 # [ dBA ]
    resampling: str = "cubic_spline"
    rendering_normalization: str = "log" # "log" or "min-max"
    map_source_config: MapSourceConfig = Field(default_factory=MapSourceConfig)

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    navigation_config: NavigationConfig = Field(default_factory=NavigationConfig)
    training_config: Optional[TrainingConfig] = None
    population_config: Optional[PopulationConfig] = None
    feature_extractor: Optional[FeatureExtractorConfig] = None
    run_name: Optional[str] = None

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            # .model_dump() converts to a dict safely
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

if __name__ == '__main__':
    print(ExperimentConfig.load(Path("scripts/common/results/configs_backup/PopulationWrapper-v0/TestMapConfig.yaml")))