from __future__ import annotations
from pathlib import Path
from typing import Literal, List

import numpy as np
import rasterio
from pydantic import Field
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds as window_from_bounds
from affine import Affine

from bluesky_gym.maps.map_sources.base import MapSourceConfig, MapSource
from bluesky_gym.maps.map_transforms import ValueTransformType, SpatialTransformType, ValuePipeline, SpatialPipeline, \
    Clip


class TransformedTiffMapSourceConfig(MapSourceConfig):
    type: Literal["transformed"] = "transformed"
    file_path: str | Path
    source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"
    value_transforms: List[ValueTransformType] = Field(default_factory=list)
    spatial_transforms: List[SpatialTransformType] = Field(default_factory=list)
    # Extra geographic margin (metres) read around the env bounds so edge observations
    # still sample real data. Should exceed the largest observation half-diagonal.
    window_margin_m: float = 100_000.0

    def build(self, env=None) -> TransformedTiffMapSource:
        assert env is not None, "TransformedTiffMapSource needs env context (call build(env))."
        return TransformedTiffMapSource(
            filepath=self.file_path,
            source_unit=self.source_unit,
            value_transforms=self.value_transforms,
            spatial_transforms=self.spatial_transforms,
            window_margin_m=self.window_margin_m,
            env=env,
        )

class TransformedTiffMapSource(MapSource):
    """Real GeoTIFF population map with per-episode domain-randomization transforms.

    On every ``regenerate`` it reads a native-resolution window of the base raster
    around the current env bounds (a crop, no resampling), applies the sampled
    spatial transforms (flip/zoom) to the ``(array, Affine)`` pair and the value
    transforms (tone curves + clip) to the valid pixels, and writes the result to a
    per-episode in-memory dataset. ``RasterSampler`` then reprojects observations from
    that transformed dataset, so transform+clip happen at native resolution *before*
    the per-observation resample (consistent clipping across resolutions).

    The normalization/reference values (``mean_value``, ``get_normalization_value``,
    ``conversion_factor``) are derived from the *untransformed* base raster so the
    reward reference and observation divisor stay constant (the transform is the only
    variable).
    """

    def __init__(self, filepath: str | Path, env, window_margin_m: float,
                 value_transforms: List[ValueTransformType],
                 spatial_transforms: List[SpatialTransformType],
                 source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"):
        # The working raster is stored already in people/km², so its own conversion is 1.0.
        super().__init__(source_unit="people_per_km2")
        self._base = rasterio.open(filepath)
        self._env = env
        self._window_margin_m = window_margin_m
        self._value_pipeline = ValuePipeline(value_transforms)
        self._spatial_pipeline = SpatialPipeline(spatial_transforms)

        # Base people_per_pixel -> people/km² factor (constant; also used to convert the
        # working window into people/km² before value transforms run).
        if source_unit == "people_per_km2":
            self._base_conversion = 1.0
        else:
            self._base_conversion = 1.0 / (self._pixel_area_m2(self._base) / 1_000_000.0)

        self._memfile: MemoryFile | None = None
        self._dataset: rasterio.DatasetReader | None = None
        self._transform: Affine | None = None
        self._norm_cache: dict[float, float] = {}
        self._mean_cache: float | None = None

        # Resolve any percentile-based Clip bounds against the (constant) base map, so a
        # config can clip at e.g. the base p99.9 exactly like the legacy map_source_max.
        for transform in value_transforms:
            if isinstance(transform, Clip) and transform.percentile is not None:
                clip_value = self.get_normalization_value(transform.percentile)
                transform.upper = (clip_value, clip_value)

    # --- reference values from the untransformed base raster -------------------
    def _filter_base_valid(self, data: np.ndarray) -> np.ndarray:
        nodata = self._base.nodata
        return data[data != nodata] if nodata is not None else data

    @property
    def mean_value(self) -> float:
        if not self._mean_cache:
            data = self._base.read(1).astype(np.float64)
            self._mean_cache = float(np.mean(self._filter_base_valid(data))) * self._base_conversion
        return self._mean_cache

    def get_normalization_value(self, percentile: float) -> float:
        if percentile not in self._norm_cache:
            data = self._base.read(1).astype(np.float64)
            self._norm_cache[percentile] = float(
                np.percentile(self._filter_base_valid(data), percentile)) * self._base_conversion
        return self._norm_cache[percentile]

    # --- per-episode working raster --------------------------------------------
    def regenerate(self, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self._spatial_pipeline.sample(rng)
        self._value_pipeline.sample(rng)

        array, transform = self._read_native_window()
        array, transform = self._spatial_pipeline.apply(array, transform)

        nodata = self._base.nodata
        out = array.astype(np.float64, copy=True)
        valid = out != nodata if nodata is not None else np.ones_like(out, dtype=bool)
        # Convert valid pixels to people/km², apply value transforms, keep nodata sentinel.
        out[valid] = self._value_pipeline.apply(out[valid] * self._base_conversion)

        self._write_working_dataset(out, transform, nodata)

    def _read_native_window(self) -> tuple[np.ndarray, Affine]:
        """Read a native-resolution crop of the base raster covering the env bounds
        plus margin (and zoom-in reach). Reads beyond the base extent are nodata-filled."""
        x_min, y_min = self._env.x_min, self._env.y_min
        x_max, y_max = self._env.x_max, self._env.y_max
        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        expansion = self._spatial_pipeline.max_read_expansion
        read_half_w = ((x_max - x_min) / 2.0 + self._window_margin_m) * expansion
        read_half_h = ((y_max - y_min) / 2.0 + self._window_margin_m) * expansion

        window = window_from_bounds(
            cx - read_half_w, cy - read_half_h, cx + read_half_w, cy + read_half_h,
            transform=self._base.transform,
        ).round_offsets().round_lengths()

        array = self._base.read(
            1, window=window, boundless=True,
            fill_value=self._base.nodata if self._base.nodata is not None else 0.0,
        )
        return array, self._base.window_transform(window)

    def _write_working_dataset(self, array: np.ndarray, transform: Affine, nodata) -> None:
        h, w = array.shape
        reuse = (
            self._dataset is not None
            and self._dataset.height == h
            and self._dataset.width == w
            and self._dataset.dtypes[0] == array.dtype.name
            and self._transform == transform
        )
        if reuse:
            self._dataset.write(array, 1)
            return

        if self._dataset is not None:
            self._dataset.close()
        if self._memfile is not None:
            self._memfile.close()

        self._transform = transform
        self._memfile = MemoryFile()
        self._dataset = self._memfile.open(
            driver="GTiff", height=h, width=w, count=1,
            dtype=array.dtype, crs=self._base.crs, transform=transform, nodata=nodata,
        )
        self._dataset.write(array, 1)

    @property
    def crs(self):
        return self._base.crs

    @property
    def transform(self) -> Affine:
        return self._transform

    @property
    def dataset(self):
        return self._dataset

    def refresh_conversion_factor(self):
        # Working raster is already people/km²; its conversion is the identity.
        self._conversion_factor = 1.0

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
        if self._memfile is not None:
            self._memfile.close()
        self._base.close()