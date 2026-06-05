from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pyproj
import rasterio
from pydantic import BaseModel, ConfigDict
from affine import Affine


class MapSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def build(self, env=None) -> MapSource:
        """Build a MapSource. Pass env when the source needs environment context (random maps)."""
        raise NotImplementedError("build() must be implemented by subclasses of MapSourceConfig")

class MapSource(ABC):
    def __init__(self, source_unit: Literal["people_per_pixel", "people_per_km2"] | None = None):
        self._source_unit = source_unit
        self._conversion_factor: float | None = None

    @property
    @abstractmethod
    def crs(self): ...

    @property
    @abstractmethod
    def transform(self) -> Affine: ...

    @property
    @abstractmethod
    def dataset(self) -> rasterio.DatasetReader: ...

    @abstractmethod
    def regenerate(self, rng: np.random.Generator | None = None):
        """Generate a new map (no-op for static sources)."""
        ...

    @abstractmethod
    def get_normalization_value(self, percentile: float) -> float:
        """Return the map value at the given percentile (0–100), in people/km².

        Used as the normalization divisor for the policy observation.
        """
        ...

    @abstractmethod
    def close(self): ...

    @property
    def conversion_factor(self) -> float:
        """Factor to convert raw map values to people_per_km2."""
        if self._conversion_factor is None:
            self.refresh_conversion_factor()
        return self._conversion_factor

    def refresh_conversion_factor(self):
        """Recompute conversion factor after dataset is created or recreated."""
        if self._source_unit == "people_per_km2":
            self._conversion_factor = 1.0
            return
        if self.dataset is None:
            raise RuntimeError("Dataset must be initialized before computing conversion factor.")
        pixel_area_km2 = self._pixel_area_m2(self.dataset) / 1_000_000.0
        self._conversion_factor = 1 / pixel_area_km2
        print(f"Conversion factor (people_per_pixel -> people_per_km2): {self._conversion_factor:.2f}" )

    def _filter_valid_data(self, data: np.ndarray) -> np.ndarray:
        nodata = self.dataset.nodata
        return data[data != nodata] if nodata is not None else data

    @staticmethod
    def _pixel_area_m2(dataset: rasterio.DatasetReader) -> float:
        crs = pyproj.CRS.from_user_input(dataset.crs)
        if crs.is_geographic:
            raise ValueError(
                "Cannot convert people_per_pixel to people_per_km2 for geographic CRS. "
                "Reproject the GeoTIFF to a projected CRS with metric units first."
            )
        resolution = dataset.res
        return abs(resolution[0] * resolution[1])
