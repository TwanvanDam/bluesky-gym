from typing import Annotated

from pydantic import Field

from bluesky_gym.maps.map_sources.base import MapSourceConfig, MapSource
from bluesky_gym.maps.map_sources.random import RandomMapSourceConfig, RandomMapSource
from bluesky_gym.maps.map_sources.tiff import TiffMapSourceConfig, TiffMapSource
from bluesky_gym.maps.map_sources.transformed import (
    TransformedTiffMapSourceConfig,
    TransformedTiffMapSource,
)

MapSourceConfigType = Annotated[
    TiffMapSourceConfig | TransformedTiffMapSourceConfig | RandomMapSourceConfig,
    Field(discriminator="type"),
]

__all__ = [
    "MapSource",
    "MapSourceConfig",
    "MapSourceConfigType",
    "TiffMapSource",
    "TiffMapSourceConfig",
    "TransformedTiffMapSource",
    "TransformedTiffMapSourceConfig",
    "RandomMapSource",
    "RandomMapSourceConfig",
]
