from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


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
    mean: float
    std: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.std))

SamplingConfig = Annotated[NormalSamplingConfig | FixedSamplingConfig | UniformSamplingConfig, Field(discriminator="distribution")]
