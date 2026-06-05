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
    def _reference_dataset(self):
        return self._dataset

    def regenerate(self, rng: np.random.Generator | None = None):
        pass

    def close(self):
        self._dataset.close()
