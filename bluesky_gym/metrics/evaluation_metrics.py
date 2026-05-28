from pathlib import Path
from typing import Callable

import pandas as pd
from bluesky.tools.position import Position

from bluesky_gym.maps.map_sources import TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler, MapObservationConfig
from bluesky_gym.metrics.fuel_model import FuelModel
from bluesky_gym.metrics.noise_model import NoiseConfig


def build_metric_fn(map_path: Path) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Initialise models once and return a calculate_metrics(df) function."""
    map_source = TiffMapSourceConfig(file_path=map_path).build()
    raster_sampler = RasterSampler(
        map_source=map_source,
        resampling="cubic_spline",
        destination_crs="epsg:3035",
    )
    noise_model = NoiseConfig().build()
    fuel_model = FuelModel("a320")

    def _fuel(altitude, tas, sim_dt, mass):
        return fuel_model.step_fuel_flow(mass=mass, tas=tas, altitude=altitude) * sim_dt

    def _noise(lat, lon, altitude, sim_dt):
        pos = Position(name=f"{lat},{lon}", reflat=0, reflon=0)
        k_m, k_px = noise_model.get_noise_power_kernel_shape_meters_and_pixels(altitude)

        noise_kernel_map_extract_config = MapObservationConfig(shape=k_px, range=k_m)
        pop = raster_sampler.get_observation_clipped(center_position=pos, orientation=0,
                                                     observation_config=noise_kernel_map_extract_config)
        return noise_model.step_total_noise(pop, altitude, sim_dt)

    def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
        alt_key = "altitude" if "altitude" in df.columns else "alt"
        df["calculated_fuel"] = df.apply(
            lambda r: _fuel(r[alt_key], r["tas"], r["sim_dt"], r["mass"]), axis=1
        )
        df["calculated_noise"] = df.apply(
            lambda r: _noise(r["lat"], r["lon"], r[alt_key], r["sim_dt"]), axis=1
        )

        mean_pop_density = map_source.mean_value
        first_rows = df.groupby("start_angle").first()

        # Mean fuel flow [kg/s]
        mean_fuel_flow = first_rows.apply(
            lambda r: fuel_model.step_fuel_flow(mass=r["mass"], tas=r["tas"], altitude=r[alt_key]), axis=1
        )

        # mean noise [W s]
        mean_ref_noise = first_rows[alt_key].apply(
            lambda alt: noise_model.calculate_mean_reference_noise(
                mean_population_density=mean_pop_density, altitude=alt
            )
        )

        df["mean_fuel_flow"] = df["start_angle"].map(mean_fuel_flow)
        df["mean_reference_noise"] = df["start_angle"].map(mean_ref_noise)
        return df

    return calculate_metrics
