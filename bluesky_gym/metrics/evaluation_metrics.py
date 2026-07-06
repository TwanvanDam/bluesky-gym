from types import SimpleNamespace
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import pyproj
from bluesky.tools.position import Position

from bluesky_gym.maps.map_sources import TiffMapSourceConfig, MapSourceConfig, \
    TransformedTiffMapSourceConfig
from bluesky_gym.maps.map_transforms import Clip
from bluesky_gym.maps.raster_sampler import RasterSampler, MapObservationConfig
from bluesky_gym.metrics.fuel_model import FuelModel
from bluesky_gym.metrics.noise_model import NoiseConfig

DEST_CRS = "epsg:3035"

# (lat, lon) -> population extract [people/km²]
PopSampleFn = Callable[[Position, MapObservationConfig], np.ndarray]


class PopSamplers(NamedTuple):
    true: PopSampleFn          # most-correct estimate (average resampling, no clip)
    training: PopSampleFn      # reproduces the reward term (training resampling + clip)
    mean_density: float        # base-raster mean people/km² for the noise reference


def make_pop_samplers(map_config: MapSourceConfig, bounds, *, clip_percentile, train_resampling,
                      true_resampling="average", dest_crs=DEST_CRS) -> PopSamplers:
    """Build the (true, training) population samplers for a run's eval map.

    ``true`` uses area-weighted ``average`` resampling and no clip -> the physically
    correct exposure. ``training`` reproduces the noise reward term: the training
    resampling plus the population clip the env applied. Legacy ``TiffMapSource``
    clipped *after* resample (np.clip on the extract); ``TransformedTiffMapSource``
    bakes the clip in at native resolution *before* resample, so we materialise a
    clip-only working dataset for it instead.
    """
    if isinstance(map_config, TiffMapSourceConfig):
        map_source = map_config.build()
        training_sampler = RasterSampler(map_source, train_resampling, dest_crs)
        true_sampler = RasterSampler(map_source, true_resampling, dest_crs)
        cap = map_source.get_normalization_value(clip_percentile)
        sample_true = lambda position, observation_config: true_sampler.get_observation_clipped(
            center_position=position, orientation=0, observation_config=observation_config)
        sample_training = lambda position, observation_config: np.clip(
            training_sampler.get_observation_clipped(
                center_position=position, orientation=0, observation_config=observation_config), 0, cap)
        mean_density = map_source.mean_value

    elif isinstance(map_config, TransformedTiffMapSourceConfig):
        clip_only = map_config.model_copy(update={"spatial_transforms": [],
                                                  "value_transforms": [Clip(percentile=clip_percentile)]})
        true = map_config.model_copy(update={"spatial_transforms": [], "value_transforms": []})
        source_clip, source_true = clip_only.build(bounds), true.build(bounds)
        # Materialise the per-episode working datasets (deterministic: no spatial transforms).
        source_clip.regenerate()
        source_true.regenerate()
        training_sampler = RasterSampler(source_clip, train_resampling, dest_crs)
        true_sampler = RasterSampler(source_true, true_resampling, dest_crs)
        sample_true = lambda position, observation_config: true_sampler.get_observation_clipped(
            center_position=position, orientation=0, observation_config=observation_config)
        sample_training = lambda position, observation_config: training_sampler.get_observation_clipped(
            center_position=position, orientation=0, observation_config=observation_config)
        mean_density = source_true.mean_value
    else:
        raise ValueError(f"{type(map_config).__name__} is not a supported MapSourceConfig.")

    return PopSamplers(true=sample_true, training=sample_training, mean_density=mean_density)


def bounds_from_df(df: pd.DataFrame, dest_crs: str = DEST_CRS, map_size_margin: float = 100_000) -> SimpleNamespace:
    """Trajectory extent in the destination CRS, for the transformed-map native window.

    The clip value and mean density come from the whole base raster, so only the
    readable geography depends on these bounds; the source adds its own margin.
    """
    transformer = pyproj.Transformer.from_crs("wgs84", dest_crs, always_xy=True)
    xs, ys = transformer.transform(np.asarray(df["lon"]), np.asarray(df["lat"]))
    return SimpleNamespace(x_min=float(xs.min() - map_size_margin), x_max=float(xs.max() + map_size_margin),
                           y_min=float(ys.min() - map_size_margin), y_max=float(ys.max()) + map_size_margin)


def build_metric_fn(samplers: PopSamplers) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Initialise models once and return a calculate_metrics(df) function."""
    noise_model = NoiseConfig().build()
    fuel_model = FuelModel("a320")

    def _fuel(altitude, tas, sim_dt, mass):
        return fuel_model.step_fuel_flow(mass=mass, tas=tas, altitude=altitude) * sim_dt

    def _noise(lat, lon, altitude, sim_dt):
        """Return (true noise, training noise) in W·s.

        The training variant reproduces the reward's noise term (training resampling
        + population clip); the true variant reflects the actual environmental impact.
        """
        pos = Position(name=f"{lat},{lon}", reflat=0, reflon=0)
        k_m, k_px = noise_model.get_noise_power_kernel_shape_meters_and_pixels(altitude)
        cfg = MapObservationConfig(shape=k_px, range=k_m)

        noise = noise_model.step_total_noise(samplers.true(pos, cfg), altitude, sim_dt)
        noise_clipped = noise_model.step_total_noise(samplers.training(pos, cfg), altitude, sim_dt)
        return noise, noise_clipped

    def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
        alt_key = "altitude" if "altitude" in df.columns else "alt"
        df["calculated_fuel"] = df.apply(
            lambda r: _fuel(r[alt_key], r["tas"], r["sim_dt"], r["mass"]), axis=1
        )
        noise_results = df.apply(
            lambda r: _noise(r["lat"], r["lon"], r[alt_key], r["sim_dt"]),
            axis=1, result_type="expand",
        )
        df["calculated_noise"] = noise_results[0]
        df["calculated_noise_clipped"] = noise_results[1]

        first_rows = df.groupby("start_angle").first()

        # Mean fuel flow [kg/s]
        mean_fuel_flow = first_rows.apply(
            lambda r: fuel_model.step_fuel_flow(mass=r["mass"], tas=r["tas"], altitude=r[alt_key]), axis=1
        )

        # mean noise [W s]
        mean_ref_noise = first_rows[alt_key].apply(
            lambda alt: noise_model.calculate_mean_reference_noise(
                mean_population_density=samplers.mean_density, altitude=alt
            )
        )

        df["mean_fuel_flow"] = df["start_angle"].map(mean_fuel_flow)
        df["mean_reference_noise"] = df["start_angle"].map(mean_ref_noise)
        return df

    return calculate_metrics
