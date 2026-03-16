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
    validation_episodes: Optional[int] = 10_000
    save_frequency: Optional[int] = 50_000

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
    print(ExperimentConfig.load(Path("scripts/common/results/configs_backup/PopulationWrapper-v0/TestMapConfig.yaml")))
