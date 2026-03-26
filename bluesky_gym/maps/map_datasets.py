from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Literal, Annotated

import gymnasium
import numpy as np
import pyproj
import rasterio
from pydantic import BaseModel, Field, ConfigDict
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from affine import Affine

from bluesky_gym.maps.random_map_generators import GeneratorBase, ZeroPopulationGenerator, MapPool


class MapSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    def build(self) -> MapSource:
        """Builds a MapSource instance from this config. For configs that require env context, this will raise NotImplementedError."""
        raise NotImplementedError("build() must be implemented by subclasses of MapSourceConfig")

    def build_for_env(self, env) -> MapSource:
        """Builds a MapSource instance from this config, using env context if needed."""
        raise NotImplementedError("build_for_env() must be implemented by subclasses of MapSourceConfig")

class TiffMapSourceConfig(MapSourceConfig):
    type: Literal["tiff"] = "tiff"
    file_path: str
    source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"

    def build(self) -> TiffMapSource:
        return TiffMapSource(self.file_path, source_unit=self.source_unit)

    def build_for_env(self, env) -> TiffMapSource:
        return self.build()

class RandomMapSourceConfig(MapSourceConfig):
    type: Literal["cities", "polygon", "population_density", "zero"]
    resolution_m: float = 1000
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)
    source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"
    pool_size: Optional[int] = None

    def get_map_details_from_env_bounds(self, env: gymnasium.Env) -> tuple[Affine, tuple[int, int], tuple[float, float]]:
        """Derive Affine transform and map shape from the env's geographic bounds.

        Computes the transform so the
        random raster covers exactly env.(lon_min,lat_min)→(lon_max,lat_max).
        """
        assert hasattr(env, "x_min") and hasattr(env, "y_min") and hasattr(env, "x_max") and hasattr(env, "y_max"), \
            "Environment must have x_min, y_min, x_max, y_max attributes to use RandomMapSourceConfig.from_env_bounds"

        x_min, y_min = env.x_min, env.y_min
        x_max, y_max = env.x_max, env.y_max

        rows, cols = int((y_max - y_min) / self.resolution_m),int((x_max - x_min) / self.resolution_m)
        transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)

        return transform, (rows, cols), (x_max - x_min, y_max - y_min)


    def build_for_env(self, env) -> RandomMapSource:
        from bluesky_gym.maps.random_map_generators import PolygonGenerator, CitiesGenerator, PopulationDensityGenerator

        assert hasattr(env, "pygame_crs") , "Environment must have pygame_crs attribute."
        map_crs = env.pygame_crs
        map_transform, map_shape, map_range = self.get_map_details_from_env_bounds(env)

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
        random_map_generator = generator(map_shape=map_shape, map_range=map_range, **self.kwargs)
        if self.pool_size is not None:
            random_map_generator = MapPool(generator=random_map_generator, pool_size=self.pool_size)
        return RandomMapSource(map_crs=map_crs, map_transform=map_transform, random_map_generator=random_map_generator)

    def build(self) -> RandomMapSource:
        raise NotImplementedError("RandomMapSourceConfig requires env context to build")


MapSourceConfigType = Annotated[TiffMapSourceConfig | RandomMapSourceConfig, Field(discriminator="type")]


class MapSource(ABC):
    def __init__(self, source_unit: Literal["people_per_pixel", "people_per_km2"] = "people_per_pixel"):
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

    @staticmethod
    def _pixel_area_m2(dataset: rasterio.DatasetReader) -> float:
        """Compute pixel area in m^2 from affine transform and projected CRS units."""
        crs = pyproj.CRS.from_user_input(dataset.crs)
        if crs.is_geographic:
            raise ValueError(
                "Cannot convert people_per_pixel to people_per_km2 for geographic CRS. "
                "Reproject the GeoTIFF to a projected CRS with metric units first."
            )
        resolution = dataset.res
        return abs(resolution[0] * resolution[1])

    @property
    def conversion_factor(self) -> float:
        """Factor to convert raw map values to people_per_km2."""
        if self._conversion_factor is None:
            self.refresh_conversion_factor()
        return self._conversion_factor

    def refresh_conversion_factor(self):
        """Refresh conversion factor after dataset is created/recreated."""
        if self._source_unit == "people_per_km2":
            self._conversion_factor = 1.0
            return

        if self.dataset is None:
            raise RuntimeError("Dataset must be initialized before computing conversion factor.")

        self._conversion_factor = self._get_conversion_factor()

    def _get_conversion_factor(self) -> float:
        """If the source unit is people_per_pixel, returns the factor to convert to people_per_km2."""
        pixel_area_km2 = self._pixel_area_m2(self.dataset) / 1_000_000.0
        conversion = 1 / pixel_area_km2
        print(f"MapSource conversion factor (people_per_pixel -> people_per_km2): {conversion:.2f}")
        return conversion

    @property
    def max(self) -> float:
        """Returns the maximum population density value in the map."""
        return self.dataset.read(1).max()

    @abstractmethod
    def close(self):
        ...

class TiffMapSource(MapSource):
    """Loads a real GeoTIFF population map (static — no regeneration)."""

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

    def regenerate(self, rng: np.random.Generator | None = None):
        pass  # Static map, nothing to regenerate

    def close(self):
        self._dataset.close()

class RandomMapSource(MapSource):
    """Generates a random synthetic population map, re-randomized on each reset."""

    def __init__(self, map_crs: str, map_transform: Affine, random_map_generator: GeneratorBase):
        self._crs = map_crs
        self._transform = map_transform
        self._memfile: MemoryFile | None = None
        self._random_map_generator = random_map_generator
        self._dataset: rasterio.DatasetReader | None = None
        self.regenerate()
        super().__init__(source_unit=self._source_unit)

    @property
    def crs(self):
        return self._crs

    @property
    def transform(self) -> Affine:
        return self._transform

    @property
    def dataset(self):
        return self._dataset

    def regenerate(self, rng: np.random.Generator | None = None):
        raw_map, source_unit = self._random_map_generator.regenerate(rng=rng)
        self._source_unit = source_unit
        h, w = raw_map.shape

        # Reuse existing MemoryFile when shape and dtype haven't changed
        if (self._dataset is not None
                and self._dataset.height == h
                and self._dataset.width == w
                and self._dataset.dtypes[0] == raw_map.dtype.name):
            self._dataset.write(raw_map, 1)
        else:
            if self._memfile is not None:
                if self._dataset is not None:
                    self._dataset.close()
                self._memfile.close()

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
            self.refresh_conversion_factor()

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
        if self._memfile is not None:
            self._memfile.close()