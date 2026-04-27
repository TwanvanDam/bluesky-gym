from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Literal, Annotated

import numpy as np
import pyproj
import rasterio
from pydantic import BaseModel, Field, ConfigDict
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from affine import Affine

from bluesky_gym.maps.map_generators import GeneratorBase, ZeroPopulationGenerator, MapPool


def compute_random_map_layout(env, resolution_m: float) -> tuple[Affine, tuple[int, int], tuple[float, float]]:
    """Derive Affine transform, shape, and metric range from the env's current geographic bounds.

    Must be called after env.reset() has set x_min/y_min/x_max/y_max.
    """
    assert hasattr(env, "x_min") and hasattr(env, "y_min") and hasattr(env, "x_max") and hasattr(env, "y_max"), \
        "Environment must have x_min, y_min, x_max, y_max set (call env.reset() first)."
    x_min, y_min = env.x_min, env.y_min
    x_max, y_max = env.x_max, env.y_max
    rows = int((y_max - y_min) / resolution_m)
    cols = int((x_max - x_min) / resolution_m)
    transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)
    return transform, (rows, cols), (x_max - x_min, y_max - y_min)


class MapSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def build(self, env=None) -> MapSource:
        """Build a MapSource. Pass env when the source needs environment context (random maps)."""
        raise NotImplementedError("build() must be implemented by subclasses of MapSourceConfig")


class TiffMapSourceConfig(MapSourceConfig):
    type: Literal["tiff"] = "tiff"
    file_path: str | Path
    source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"

    def build(self, env=None) -> TiffMapSource:
        return TiffMapSource(self.file_path, source_unit=self.source_unit)


class RandomMapSourceConfig(MapSourceConfig):
    type: Literal["cities", "polygon", "population_density", "zero"]
    resolution_m: float = 1000
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)
    source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"
    pool_size: Optional[int] = None

    def build(self, env) -> RandomMapSource:
        from bluesky_gym.maps.map_generators import PolygonGenerator, CitiesGenerator, PopulationDensityGenerator

        assert hasattr(env, "map_projection_crs"), "Environment must have map_projection_crs attribute."
        map_crs = env.map_projection_crs

        match self.type:
            case "cities":
                generator = CitiesGenerator
            case "polygon":
                generator = PolygonGenerator
            case "population_density":
                generator = PopulationDensityGenerator
            case "zero":
                generator = ZeroPopulationGenerator
            case _:
                raise ValueError(f"Unsupported random map source type: {self.type}")

        # Use placeholder shape/range — update_layout() is called on every regenerate()
        placeholder_shape = (1, 1)
        placeholder_range = (1.0, 1.0)
        random_map_generator = generator(map_shape=placeholder_shape, map_range=placeholder_range, **self.kwargs)
        if self.pool_size is not None:
            random_map_generator = MapPool(generator=random_map_generator, pool_size=self.pool_size)
        return RandomMapSource(map_crs=map_crs, random_map_generator=random_map_generator, env=env, resolution_m=self.resolution_m)


MapSourceConfigType = Annotated[TiffMapSourceConfig | RandomMapSourceConfig, Field(discriminator="type")]


# ---------------------------------------------------------------------------
# Runtime classes
# ---------------------------------------------------------------------------

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


class TiffMapSource(MapSource):
    """Loads a real GeoTIFF population map."""

    def __init__(self, filepath: str | Path, source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"):
        super().__init__(source_unit=source_unit)
        self._dataset = rasterio.open(filepath)
        self.refresh_conversion_factor()
        self._norm_cache: dict[float, float] = {}

    @property
    def crs(self):
        return self._dataset.crs

    @property
    def transform(self) -> Affine:
        return self._dataset.transform

    @property
    def dataset(self):
        return self._dataset

    def get_normalization_value(self, percentile: float) -> float:
        if percentile not in self._norm_cache:
            data = self._dataset.read(1).astype(np.float64)
            self._norm_cache[percentile] = float(np.percentile(self._filter_valid_data(data), percentile)) * self.conversion_factor
        return self._norm_cache[percentile]

    def regenerate(self, rng: np.random.Generator | None = None):
        pass

    def close(self):
        self._dataset.close()


class RandomMapSource(MapSource):
    """Generates a random synthetic population map, re-randomized on each reset.

    Bounds-aware: recomputes its Affine transform and shape from the env on every regenerate(),
    so it follows the env when destination sampling produces variable geographic areas.
    The source_unit is determined by the generator and set on first regenerate().
    """

    def __init__(self, map_crs: str, random_map_generator: GeneratorBase, env, resolution_m: float):
        super().__init__(source_unit=None)  # source_unit is set by the generator on first regenerate()
        self._crs = map_crs
        self._env = env
        self._resolution_m = resolution_m
        self._transform: Affine | None = None
        self._memfile: MemoryFile | None = None
        self._random_map_generator = random_map_generator
        self._dataset: rasterio.DatasetReader | None = None

    @property
    def crs(self):
        return self._crs

    @property
    def transform(self) -> Affine:
        return self._transform

    @property
    def dataset(self):
        return self._dataset

    def get_normalization_value(self, percentile: float) -> float:
        data = self._dataset.read(1).astype(np.float64)
        return float(np.percentile(self._filter_valid_data(data), percentile)) * self.conversion_factor

    def regenerate(self, rng: np.random.Generator | None = None):
        new_transform, new_shape, new_range = compute_random_map_layout(self._env, self._resolution_m)
        self._random_map_generator.update_layout(new_shape, new_range)

        raw_map, source_unit = self._random_map_generator.regenerate(rng=rng)
        self._source_unit = source_unit
        h, w = raw_map.shape

        # Reuse existing MemoryFile only when shape, dtype, AND transform are all unchanged.
        if (self._dataset is not None
                and self._dataset.height == h
                and self._dataset.width == w
                and self._dataset.dtypes[0] == raw_map.dtype.name
                and self._transform == new_transform):
            self._dataset.write(raw_map, 1)
        else:
            if self._memfile is not None:
                if self._dataset is not None:
                    self._dataset.close()
                self._memfile.close()

            self._transform = new_transform
            self._memfile = MemoryFile()
            self._dataset = self._memfile.open(
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype=raw_map.dtype,
                crs=self._crs,
                transform=self._transform,
            )
            self._dataset.write(raw_map, 1)
            self._conversion_factor = None  # reset so it's recomputed for new dataset
            self.refresh_conversion_factor()

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
        if self._memfile is not None:
            self._memfile.close()
