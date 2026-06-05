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
        # Reference-statistic caches (mean / percentile of the reference dataset).
        # Valid as long as the reference dataset is unchanged; sources whose map
        # changes each episode call _invalidate_reference_cache() in regenerate().
        self._mean_cache: float | None = None
        self._norm_cache: dict[float, float] = {}

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
    def close(self): ...

    # --- reference statistics --------------------------------------------------
    # mean_value and get_normalization_value are computed from the *reference*
    # dataset: the dataset used as the constant normalization reference. For most
    # sources that is the live ``dataset``; TransformedTiffMapSource points it at
    # the untransformed base raster so the reference stays fixed under the
    # per-episode transform.

    @property
    @abstractmethod
    def _reference_dataset(self) -> rasterio.DatasetReader:
        """Dataset that mean_value / get_normalization_value are computed from."""
        ...

    @property
    def _reference_conversion(self) -> float:
        """people/km² conversion factor for the reference dataset."""
        return self.conversion_factor

    @property
    def mean_value(self) -> float:
        """Mean population density (people/km²) of the reference dataset."""
        if self._mean_cache is None:
            self._mean_cache = self._mean(self._reference_dataset, self._reference_conversion)
        return self._mean_cache

    def get_normalization_value(self, percentile: float) -> float:
        """Reference-dataset value at the given percentile (0–100), in people/km².

        Used as the normalization divisor for the policy observation.
        """
        if percentile not in self._norm_cache:
            self._norm_cache[percentile] = self._percentile(
                self._reference_dataset, self._reference_conversion, percentile)
        return self._norm_cache[percentile]

    def _invalidate_reference_cache(self) -> None:
        """Drop cached reference statistics (call when the reference dataset changes)."""
        self._mean_cache = None
        self._norm_cache.clear()

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

    @staticmethod
    def _valid_pixels(dataset: rasterio.DatasetReader) -> np.ndarray:
        """Band-1 pixels with the nodata sentinel removed, as float64."""
        data = dataset.read(1).astype(np.float64)
        nodata = dataset.nodata
        return data[data != nodata] if nodata is not None else data

    @staticmethod
    def _mean(dataset: rasterio.DatasetReader, conversion: float) -> float:
        """Mean of the valid pixels, in people/km²."""
        return float(np.mean(MapSource._valid_pixels(dataset))) * conversion

    @staticmethod
    def _percentile(dataset: rasterio.DatasetReader, conversion: float, percentile: float) -> float:
        """Value at the given percentile (0–100) of the valid pixels, in people/km²."""
        return float(np.percentile(MapSource._valid_pixels(dataset), percentile)) * conversion

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
