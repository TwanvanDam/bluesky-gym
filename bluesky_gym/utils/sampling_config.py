from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

class ExclusionZone(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    lat: float
    lon: float
    radius_km: float

    def contains(self, lat: float, lon: float, R: float = 6371) -> bool:
        """Is great circle distance less than 'radius_km'?
        https://en.wikipedia.org/wiki/Great-circle_distance
        """
        lat1, lat2 = np.deg2rad(lat), np.deg2rad(self.lat)
        lon1, lon2 = np.deg2rad(lon), np.deg2rad(self.lon)

        cos_central_angle = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(abs(lon1 - lon2))
        delta_sigma = float(np.arccos(np.clip(cos_central_angle, -1.0, 1.0)))
        return delta_sigma * R <= self.radius_km


class SamplingConfigBase(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    def sample(self, rng: np.random.Generator) -> float:
        raise NotImplementedError("sample() must be implemented by subclasses of SamplingConfig")

class FixedSamplingConfig(SamplingConfigBase):
    distribution: Literal["fixed"] = "fixed"
    value: float

    def sample(self, rng: np.random.Generator) -> float:
        return self.value

class UniformSamplingConfig(SamplingConfigBase):
    distribution: Literal["uniform"] = "uniform"
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

class NormalSamplingConfig(SamplingConfigBase):
    distribution: Literal["normal"] = "normal"
    mean: float = 0
    std: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.std))

SamplingConfig = Annotated[NormalSamplingConfig | FixedSamplingConfig | UniformSamplingConfig, Field(discriminator="distribution")]