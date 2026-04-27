from dataclasses import dataclass

import numpy as np
import pyproj
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

from bluesky.tools.position import Position
from bluesky_gym.maps.map_sources import MapSource

@dataclass
class RasterSampler:
    map_source: MapSource
    resampling: str

    destination_crs: str

    def __post_init__(self) -> None:
        self.wgs84_to_dest = pyproj.Transformer.from_crs("wgs84", self.destination_crs, always_xy=True)

    def _get_dst_transform_from_center(self, center_position: Position, orientation: float, out_meters: tuple[float, float],
                           out_shape: tuple[int, int]) -> Affine:
        center_xy = self.wgs84_to_dest.transform(center_position.lon, center_position.lat)

        # Calculate the resolution (meters per pixel) for the output slice
        cols, rows = out_shape
        res_x = out_meters[0] / cols
        res_y = out_meters[1] / rows

        dst_transform = (
                Affine.translation(*center_xy) *
                Affine.rotation(- orientation) *
                Affine.scale(res_x, -res_y) *
                Affine.translation(- cols / 2, -rows / 2)
        )
        return dst_transform

    @staticmethod
    def get_dst_transform_from_bounds(x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int) -> Affine:
        return from_bounds(x_min, y_min, x_max, y_max, width, height)

    def _extract_view_from_map(self, dst_transform: Affine, out_shape: tuple[int, int]) -> np.ndarray:
        destination = np.zeros(out_shape[::-1])

        # Perform the Reprojection
        reproject(
            source=rasterio.band(self.map_source.dataset, 1),
            destination=destination,
            src_transform=self.map_source.transform,
            src_crs=self.map_source.crs,
            dst_transform=dst_transform,
            dst_crs=self.destination_crs,
            resampling=getattr(Resampling, self.resampling)
        )
        return destination * self.map_source.conversion_factor

    def get_view_corners(self, center_position: Position, orientation: float,
                                 out_shape: tuple[int, int], out_meters: tuple[float, float]) -> list[
        tuple[float, float]]:
        """Calculates the corners of the extracted view in screen coordinates (meters in destination CRS)."""
        dst_transform = self._get_dst_transform_from_center(center_position, orientation, out_meters, out_shape)

        cols, rows = out_shape
        pixel_corners = [(0, 0), (cols, 0), (cols, rows), (0, rows)]

        view_corners = []
        for col, row in pixel_corners:
            x_meters, y_meters = dst_transform * (col, row)
            view_corners.append((x_meters, y_meters))

        return view_corners

    def get_observation_clipped(self, center_position: Position, orientation: float, out_shape: tuple[int, int],
                        out_meters: tuple[float, float]) -> np.ndarray:
        """Extracts a rotated slice from the map centered at the given position and orientation.
        Values are clipped to a minimum of 0 to avoid negative population densities."""
        dst_transform = self._get_dst_transform_from_center(center_position, orientation, out_meters, out_shape)
        map_extract = self._extract_view_from_map(dst_transform, out_shape)
        return np.clip(map_extract, 0, None)

    def get_background(self, x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int) -> np.ndarray:
        """Extracts a map from the map source from the given bounds to be used as background.
        Negative values are set to NaN to avoid visualizing them in the background."""
        dst_transform = self.get_dst_transform_from_bounds(x_min, y_min, x_max, y_max, width, height)
        map_extract = self._extract_view_from_map(dst_transform, (width, height))
        map_extract = np.where(map_extract < -100, np.nan, map_extract)
        map_extract = np.clip(map_extract, 0, None)
        return map_extract

    def get_value_at_coordinate(self, coordinate: Position) -> float:
        xy = self.wgs84_to_dest.transform(coordinate.lon, coordinate.lat)
        return next(self.map_source.dataset.sample([xy]))

    def coordinate_on_land(self, coordinate: Position) -> bool:
        return self.get_value_at_coordinate(coordinate) >= 0
