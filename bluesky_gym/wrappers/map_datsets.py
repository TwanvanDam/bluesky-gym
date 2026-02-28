from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import rasterio
from rasterio.io import MemoryFile
from affine import Affine

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
    def regenerate(self):
        """Generate a new map (no-op for static sources)."""
        ...

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

    def regenerate(self):
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
    def from_env_bounds(cls, env, random_map_generator: Callable, array_size: tuple[int, int] | None):
        """Derive Affine transform + CRS from the env's geographic bounds.

        Uses env.pygame_crs as the target CRS (same space that rendering
        and observations live in), and computes the transform so the
        random raster covers exactly env.(lon_min,lat_min)→(lon_max,lat_max).
        If no array size is provided, the env.window_size is used.
        """
        transformer = env.coordinate_transformer
        x_min, y_min = transformer.transform(env.lon_min, env.lat_min)
        x_max, y_max = transformer.transform(env.lon_max, env.lat_max)

        if not array_size:
            array_size = env.window_size
        res_x = (x_max - x_min) / array_size[0]
        res_y = (y_max - y_min) / array_size[1]

        # Raster convention: origin at top-left, y points downward
        map_transform = Affine(res_x, 0.0, x_min,
                               0.0, -res_y, y_max)

        return cls(
            map_crs=env.pygame_crs,  # synthetic data lives in pygame_crs
            map_transform=map_transform,
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

    def regenerate(self):
        if self._memfile is not None:
            self._dataset.close()
            self._memfile.close()

        raw_map = self._random_map_generator()
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