from abc import ABC, abstractmethod
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

    def __init__(self, filepath: str):
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