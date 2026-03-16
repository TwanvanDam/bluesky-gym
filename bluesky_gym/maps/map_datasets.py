from __future__ import annotations
import functools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Literal, Annotated

import gymnasium
import numpy as np
import rasterio
from pydantic import BaseModel, Field
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from affine import Affine

class MapSourceConfig(BaseModel):
    def build(self) -> MapSource:
        """Builds a MapSource instance from this config. For configs that require env context, this will raise NotImplementedError."""
        raise NotImplementedError("build() must be implemented by subclasses of MapSourceConfig")

    def build_for_env(self, env) -> MapSource:
        """Builds a MapSource instance from this config, using env context if needed."""
        raise NotImplementedError("build_for_env() must be implemented by subclasses of MapSourceConfig")

class TiffMapSourceConfig(MapSourceConfig):
    type: Literal["tiff"] = "tiff"
    file_path: str

    def build(self) -> TiffMapSource:
        return TiffMapSource(self.file_path)

    def build_for_env(self, env) -> TiffMapSource:
        return self.build()

class RandomMapSourceConfig(MapSourceConfig):
    type: Literal["cities", "polygon", "population_density"]
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @staticmethod
    def from_env_bounds(env: gymnasium.Env, random_map_generator: Callable):
        """Derive Affine transform + CRS from the env's geographic bounds.

        Uses env.pygame_crs as the target CRS (same space that rendering
        and observations live in), and computes the transform so the
        random raster covers exactly env.(lon_min,lat_min)→(lon_max,lat_max).
        If no array size is provided, the env.window_size is used.
        """
        assert hasattr(env, "x_min") and hasattr(env, "y_min") and hasattr(env, "x_max") and hasattr(env, "y_max"), \
            "Environment must have x_min, y_min, x_max, y_max attributes to use RandomMapSourceConfig.from_env_bounds"

        assert hasattr(env, "pygame_crs") , "Environment must have pygame_crs attribute."
        map_crs = env.pygame_crs

        x_min, y_min = env.x_min, env.y_min
        x_max, y_max = env.x_max, env.y_max

        rows, cols = random_map_generator().shape
        transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)

        return RandomMapSource(
            map_crs=map_crs,  # synthetic data lives in pygame_crs
            map_transform=transform,
            random_map_generator=random_map_generator,
        )

    def build_for_env(self, env) -> RandomMapSource:
        from bluesky_gym.maps.random_map_generators import generate_cities, generate_random_shapes_map, generate_population_density
        if self.type == "cities":
            generator = generate_cities
            if self.kwargs:
                generator = functools.partial(generate_cities, **self.kwargs)
            return self.from_env_bounds(env, generator)
        elif self.type == "polygon":
            generator = generate_random_shapes_map
            if self.kwargs:
                generator = functools.partial(generate_random_shapes_map, **self.kwargs)
            return self.from_env_bounds(env, generator)
        elif self.type == "population_density":
            generator = generate_population_density
            if self.kwargs:
                generator = functools.partial(generate_population_density, **self.kwargs)
            return self.from_env_bounds(env, generator)
        raise ValueError(f"Unsupported random map source type: {self.type}")

    def build(self) -> RandomMapSource:
        raise NotImplementedError("RandomMapSourceConfig requires env context to build")


MapSourceConfigType = Annotated[TiffMapSourceConfig | RandomMapSourceConfig, Field(discriminator="type")]


class MapSource(ABC):

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

    @property
    def max(self) -> float:
        """Returns the maximum population density value in the map."""
        return self.dataset.read(1).max()

    @abstractmethod
    def close(self):
        ...

class TiffMapSource(MapSource):
    """Loads a real GeoTIFF population map (static — no regeneration)."""

    def __init__(self, filepath: str | Path):
        self._dataset = rasterio.open(filepath)

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

    def __init__(self, map_crs: str, map_transform: Affine, random_map_generator: Callable):
        self._crs = map_crs
        self._transform = map_transform
        self._memfile: MemoryFile | None = None
        self._random_map_generator = random_map_generator
        self._dataset: rasterio.DatasetReader | None = None
        self.regenerate()

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
        if self._memfile is not None:
            self._dataset.close()
            self._memfile.close()

        raw_map = self._random_map_generator(rng=rng)
        h, w = raw_map.shape

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

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
        if self._memfile is not None:
            self._memfile.close()