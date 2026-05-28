import warnings
from functools import partial
from typing import Callable, List, Literal

import gymnasium as gym
import matplotlib
import numpy as np
import pygame
from affine import Affine
from gymnasium import spaces
from matplotlib.colors import Normalize, FuncNorm
from pydantic import BaseModel, Field, ConfigDict

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv, TerminationReason
from bluesky_gym.maps.map_sources import MapSourceConfigType
from bluesky_gym.maps.raster_sampler import RasterSampler, MapObservationConfig
from bluesky_gym.metrics.noise_model import NoiseModel, NoiseConfig


class PopulationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    noise_model_config: NoiseConfig = Field(default_factory=NoiseConfig)
    map_source_config: MapSourceConfigType

    map_observation_configs: List[MapObservationConfig]

    # Fuel weight determines how much the noise penalty should factor into the overall reward, with 1.0 meaning only fuel consumption matters and 0.0 meaning only noise matters.
    # This allows for easy tuning of the reward function to find the right balance between fuel efficiency and noise reduction.
    fuel_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    resampling: Literal["cubic_spline", "average", "sum", "min", "max", "bilinear", "cubic"] = "cubic_spline"
    normalization_percentile: float = Field(default=99.9, ge=0.0, le=100.0)
    clip_noise_reward: bool = False
    observation_normalization: Literal["log", "min_max", "min-max"] = "log"


class Population(gym.Wrapper):
    def __init__(self, env: gym.Env, config: PopulationConfig, color_map: str = "Blues"):
        super().__init__(env)
        self.total_episode_noise_reward = None
        self.total_episode_noise = None
        self.env: gym.Env = env
        self.base_env: BaseNavigationEnv = self.unwrapped
        self.base_env._render_owned_by_wrapper = True

        self.config = config
        self.noise_model = NoiseModel(config.noise_model_config)

        self.map_source = config.map_source_config.build(self.base_env)
        self.raster_sampler = RasterSampler(self.map_source, resampling=self.config.resampling,
                                            destination_crs=self.base_env.map_projection_crs)
        self.map_source_max: float = np.nan
        self.mean_reference_noise: float = np.nan

        self.window = None
        self.observation_configs = config.map_observation_configs
        self.population_observation = None

        # cache the map used as background since it does not change often.
        self.background_map: None | np.ndarray = None
        self.color_map: str = color_map

        self.clip_indicator_color: tuple[int, int, int, int] = (255, 0, 0, 255)
        self.render_normalizer: Normalize | None = None

        assert isinstance(self.env.observation_space, spaces.Dict)
        maps = {f"population_map_{i}": spaces.Box(low=0, high=np.inf, shape=observation_config.shape, dtype=np.float64)
                for i, observation_config in
                enumerate(self.observation_configs)}

        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            **maps
        })

        self.base_env.fuel_weight = config.fuel_weight
        self.base_env.add_reward_component(self._get_noise_reward)

    @property
    def composite_window_size(self) -> tuple[int, int]:
        return self.base_env.window_size[0] + sum(x_size for x_size, _ in self._get_panel_sizes()), \
            self.base_env.window_size[1]

    def reset(self, seed=None, options=None):
        current_seed = seed
        for _ in range(100):
            observation, info = super().reset(seed=current_seed, options=options)
            current_seed = None  # advance RNG on retries rather than re-seeding to the same value
            self.map_source.regenerate(rng=self.np_random)
            if self.raster_sampler.coordinate_on_land(self.base_env.destination):
                break
        else:
            warnings.warn(
                "Population.reset: destination still not on land after 100 attempts; "
                "proceeding with last sampled positions/map. Check map source coverage "
                "and destination_lat/lon SamplingConfig.",
                stacklevel=2,
            )

        self.map_source_max = self.map_source.get_normalization_value(self.config.normalization_percentile)

        self._update_population_observation()
        observation = self._inject_population_observation(observation)

        self.background_map = self.get_background()

        # Reset noise tracking variables
        self.total_episode_noise = 0.0
        self.total_episode_noise_reward = 0.0

        ac_alt = self.base_env.get_aircraft_altitude()
        noise_kernel_shape_meters, noise_kernel_shape_pixels = self.noise_model.get_noise_power_kernel_shape_meters_and_pixels(ac_alt)
        self.noise_kernel_observation = MapObservationConfig(shape=noise_kernel_shape_pixels, range=noise_kernel_shape_meters)

        self.mean_reference_noise = self.noise_model.calculate_mean_reference_noise(
            mean_population_density=self.map_source.mean_value,
            altitude=ac_alt,
        )

        if self.render_mode is not None:
            self.render_normalizer = self._get_normalization()

        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done:
            info["total_episode_noise"] = self.total_episode_noise
            info["total_episode_noise_reward"] = self.total_episode_noise_reward
        else:
            self._update_population_observation()
        observation = self._inject_population_observation(observation)

        if not done and self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def close(self):
        """Close the rasterio dataset when done"""
        self.map_source.close()
        self.env.close()

    def _inject_population_observation(self, observation: dict) -> dict:
        return {**observation, **self.population_observation}

    def _update_population_observation(self) -> None:
        ac_pos = self.base_env.get_aircraft_position()
        ac_hdg = self.base_env.get_aircraft_heading()
        observations = {f"population_map_{i}":
                            self.raster_sampler.get_observation_clipped(center_position=ac_pos, orientation=ac_hdg,
                                                                        observation_config=observation_config) for
                        i, observation_config in enumerate(self.observation_configs)}
        self.population_observation = observations

    def get_background(self):
        """Returns the background population map for the entire environment bounds, used for rendering the full map as the background."""
        width, height = self.base_env.window_size
        return self.raster_sampler.get_background(x_min=self.base_env.x_min,
                                                  y_min=self.base_env.y_min,
                                                  x_max=self.base_env.x_max,
                                                  y_max=self.base_env.y_max,
                                                  width=width,
                                                  height=height)

    def get_background_transform(self) -> Affine:
        width, height = self.base_env.window_size
        return self.raster_sampler.get_dst_transform_from_bounds(x_min=self.base_env.x_min,
                                                                 y_min=self.base_env.y_min,
                                                                 x_max=self.base_env.x_max,
                                                                 y_max=self.base_env.y_max,
                                                                 width=width,
                                                                 height=height)
    def _get_noise_reward(self) -> tuple[float, bool, TerminationReason]:
        ac_alt = self.base_env.get_aircraft_altitude()
        ac_pos = self.base_env.get_aircraft_position()
        sim_dt = self.base_env.sim_dt

        population_map_extract = self.raster_sampler.get_observation_clipped(center_position=ac_pos,
                                                                             orientation=0,
                                                                             observation_config=self.noise_kernel_observation)
        if self.config.clip_noise_reward:
            # Works by clipping the population map extract that is used to generate the noise reward by the self.map_source_max
            population_map_extract = np.clip(population_map_extract, 0, self.map_source_max)

        step_normalized_noise = self.noise_model.step_normalized_noise(population_map_extract=population_map_extract,
                                                                       altitude=ac_alt,
                                                                       mean_reference_noise=self.mean_reference_noise,
                                                                       sim_dt=sim_dt)
        noise_penalty = - (
                1 - self.base_env.fuel_weight) * step_normalized_noise * self.base_env.dense_reward_scaling

        self.total_episode_noise += step_normalized_noise
        self.total_episode_noise_reward += noise_penalty
        return noise_penalty, False, TerminationReason.NONE

    def render(self):
        # Use a canvas with composite_window_size
        self.base_env.initialize_pygame(self.composite_window_size)
        self.base_env.handle_pygame_events()
        base_surface = pygame.Surface(self.base_env.window_size)
        for draw_function in self.get_base_render_layers():
            draw_function(base_surface)

        canvas = pygame.Surface(self.composite_window_size)
        canvas.blit(base_surface, (0, 0))

        render_dest = (self.base_env.window_size[0], 0)
        for draw_function, panel_size in zip(self.get_panel_render_layers(), self._get_panel_sizes()):
            panel_surface = pygame.Surface(panel_size)
            draw_function(panel_surface)
            canvas.blit(panel_surface, render_dest)
            render_dest = (render_dest[0] + panel_size[0], 0)

        return self.base_env.present_canvas(canvas)

    def _get_panel_sizes(self) -> list[tuple[int, int]]:
        y_size = self.base_env.window_size[1]
        return [
            (int((observation_config.range[0] / observation_config.range[1]) * self.base_env.window_size[0]), y_size)
            for observation_config in
            self.observation_configs]

    def get_base_render_layers(self) -> list[Callable]:
        """Override to insert custom layers into rendering pipeline."""
        return [
            lambda canvas: canvas.fill(pygame.Color("grey")),
            partial(self._render_array, render_size=self.base_env.window_size, array=self.background_map),
            self.base_env.draw_airport,
            self.base_env.draw_aircraft,
            self._draw_box_around_aircraft,
        ]

    def get_panel_render_layers(self) -> list[Callable]:
        return [partial(self._render_array, render_size=size, array=observation) for size, observation in
                zip(self._get_panel_sizes(), self.population_observation.values())]

    def _get_normalization(self) -> Normalize:
        """Get the appropriate matplotlib Normalize instance based on config."""
        v_min = 0
        v_max = self.map_source_max

        if v_min == v_max:
            return Normalize(vmin=0, vmax=1)
        normalization_mode = self.config.observation_normalization
        if normalization_mode == "log":
            return FuncNorm(functions=(np.log1p, np.expm1), vmin=v_min, vmax=v_max, clip=True)
        elif normalization_mode in ["min_max", "min-max"]:
            return Normalize(vmin=v_min, vmax=v_max, clip=True)

    def _convert_heatmap_to_rgba_array(self, population_map: np.ndarray) -> np.ndarray:
        # Mask the area that has no data available ( negative population density )
        no_data_mask = np.isnan(population_map)

        normalized_map = self.render_normalizer(population_map)

        color_data = matplotlib.colormaps[self.color_map](normalized_map)
        rgba_array = (color_data * 255).astype(np.uint8)

        # Mark areas with population density that would be clipped
        clipped_mask = ~no_data_mask & (population_map > self.map_source_max)
        rgba_array[clipped_mask] = self.clip_indicator_color

        # Make areas without data transparent
        rgba_array[no_data_mask, 3] = 0
        return rgba_array

    def _render_array(self, canvas: pygame.Surface, render_size: tuple[int, int], array: np.ndarray) -> None:
        rgba_array = self._convert_heatmap_to_rgba_array(array)
        shape = array.shape[::-1]
        heatmap_surf = pygame.image.frombuffer(rgba_array.tobytes(), shape, "RGBA")
        heatmap_surf = pygame.transform.scale(heatmap_surf, render_size)
        canvas.blit(heatmap_surf, (0, 0))

    def _draw_box_around_aircraft(self, canvas):
        ac_pos = self.base_env.get_aircraft_position()
        ac_hdg = self.base_env.get_aircraft_heading()
        for observation_config in self.observation_configs:
            corners = self.raster_sampler.get_view_corners(center_position=ac_pos,
                                                           orientation=ac_hdg,
                                                           observation_config=observation_config)
            corners = [self.base_env.meters_to_pix(corner) for corner in corners]
            pygame.draw.polygon(canvas, pygame.color.Color("red"), corners, width=2)
