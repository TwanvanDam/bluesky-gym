from typing import Optional

import numpy as np
from pydantic import BaseModel


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