from __future__ import annotations
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from affine import Affine

from bluesky_gym.maps.map_sources.base import MapSource, MapSourceConfig

class TiffMapSourceConfig(MapSourceConfig):
    type: Literal["tiff"] = "tiff"
    file_path: str | Path
    source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"

    def build(self, env=None) -> TiffMapSource:
        return TiffMapSource(self.file_path, source_unit=self.source_unit)

class TiffMapSource(MapSource):

    def __init__(self, filepath: str | Path, source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"):
        super().__init__(source_unit=source_unit)
        self._dataset = rasterio.open(filepath)
        self.refresh_conversion_factor()
        self._norm_cache: dict[float, float] = {}
        self._mean_cache: float | None = None

    @property
    def crs(self):
        return self._dataset.crs

    @property
    def transform(self) -> Affine:
        return self._dataset.transform

    @property
    def dataset(self):
        return self._dataset

    @property
    def mean_value(self) -> float:
        if not self._mean_cache:
            data = self._dataset.read(1).astype(np.float64)
            self._mean_cache = float(np.mean(self._filter_valid_data(data))) * self.conversion_factor
        return self._mean_cache

    def get_normalization_value(self, percentile: float) -> float:
        if percentile not in self._norm_cache:
            data = self._dataset.read(1).astype(np.float64)
            self._norm_cache[percentile] = float(np.percentile(self._filter_valid_data(data), percentile)) * self.conversion_factor
        return self._norm_cache[percentile]

    def regenerate(self, rng: np.random.Generator | None = None):
        pass

    def close(self):
        self._dataset.close()
