from typing import Annotated

from pydantic import Field

from bluesky_gym.maps.map_sources.base import MapSourceConfig, MapSource
from bluesky_gym.maps.map_sources.random import RandomMapSourceConfig
from bluesky_gym.maps.map_sources.tiff import TiffMapSourceConfig
from bluesky_gym.maps.map_sources.transformed import TransformedTiffMapSourceConfig

MapSourceConfigType = Annotated[
    TiffMapSourceConfig | TransformedTiffMapSourceConfig | RandomMapSourceConfig,
    Field(discriminator="type"),
]
