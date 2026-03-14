import functools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Dict, Any

import numpy as np
import rasterio
from pydantic import BaseModel, Field
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from affine import Affine

class MapSourceConfig(BaseModel):
    type: str = "polygon"  # "tiff", "polygon" or "cities"
    file_path: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def build(self, env):
        from bluesky_gym.maps.random_map_generators import generate_cities, generate_random_shapes_map, generate_population_density

        if self.type == "tiff":
            if not self.file_path:
                raise ValueError("file_path is required for tiff map source")
            if self.kwargs: raise ValueError(f"MapSource {self.type} does not support kwargs")
            return TiffMapSource(str(self.file_path))  # Convert Path to str if needed
        elif self.type == "cities":
            # Use kwargs to configure the generator if any
            generator = generate_cities
            if self.kwargs:
                generator = functools.partial(generate_cities, **self.kwargs)
            return RandomMapSource.from_env_bounds(env, generator)
        elif self.type == "polygon":
            generator = generate_random_shapes_map
            if self.kwargs:
                generator = functools.partial(generate_random_shapes_map, **self.kwargs)
            return RandomMapSource.from_env_bounds(env, generator)
        elif self.type == "population_density":
            generator = generate_population_density
            if self.kwargs:
                generator = functools.partial(generate_population_density, **self.kwargs)
            return RandomMapSource.from_env_bounds(env, generator)
        else:
            raise ValueError(f"Unknown map source type: {self.type}")


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

    def close(self):
        pass

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

    @classmethod
    def from_env_bounds(cls, env, random_map_generator: Callable):
        """Derive Affine transform + CRS from the env's geographic bounds.

        Uses env.pygame_crs as the target CRS (same space that rendering
        and observations live in), and computes the transform so the
        random raster covers exactly env.(lon_min,lat_min)→(lon_max,lat_max).
        If no array size is provided, the env.window_size is used.
        """
        x_min, y_min = env.x_min, env.y_min
        x_max, y_max = env.x_max, env.y_max

        rows, cols = random_map_generator().shape
        transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)

        return cls(
            map_crs=env.pygame_crs,  # synthetic data lives in pygame_crs
            map_transform=transform,
            random_map_generator=random_map_generator,
        )


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