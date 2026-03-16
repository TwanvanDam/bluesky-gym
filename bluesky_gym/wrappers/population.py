from functools import partial
from typing import Callable, List, Tuple

import gymnasium as gym
import matplotlib
import numpy as np
import pygame
from affine import Affine
from gymnasium import spaces
from matplotlib.colors import Normalize, FuncNorm
from pydantic import BaseModel, Field

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv, TerminationReason
from bluesky_gym.maps.map_datasets import MapSourceConfigType, RandomMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler
from bluesky_gym.metrics.noise_model import NoiseModel, NoiseConfig


class PopulationConfig(BaseModel):
    noise_model_config: NoiseConfig = Field(default_factory=NoiseConfig)
    map_source_config: MapSourceConfigType = Field(default_factory=lambda: RandomMapSourceConfig(type="cities"))
    observation_shape: List[Tuple[int, int]] = Field(default_factory=lambda: [(64, 64)])  # [px, px]
    observation_range: List[Tuple[int, int]] = Field(default_factory=lambda: [(100_000, 100_000)])  # [m, m]
    fuel_to_noise_ratio: float = 0.5
    resampling: str = "cubic_spline"
    rendering_normalization: str = "log"  # "log" or "min-max"
    observation_normalization: str = "log"


class Population(gym.Wrapper):
    def __init__(self, env: gym.Env, config: PopulationConfig = PopulationConfig(), color_map: str = "Blues"):
        super().__init__(env)
        self.total_episode_noise_reward = None
        self.total_episode_noise = None
        self.env: gym.Env = env
        self.base_env: BaseNavigationEnv = self.unwrapped
        self.base_env._render_owned_by_wrapper = True

        self.config = config
        self.noise_model = NoiseModel(config.noise_model_config)

        # class to handle all reading and creating of population maps
        self.map_source = config.map_source_config.build_for_env(self.base_env)
        self.raster_sampler = RasterSampler(self.map_source, resampling=self.config.resampling,
                                            destination_crs=self.base_env.pygame_crs)
        self.map_source_max: float = np.nan
        self.mean_step_noise: float = np.nan

        self.window = None
        self.observation_shape = config.observation_shape
        self.observation_range = config.observation_range
        self.population_observation = None

        # cache the map used as background since it does not change often.
        self.background_map: None | np.ndarray = None
        self.color_map: str = color_map
        self.background_max = None
        self.render_normalizer: Normalize | None = None

        assert isinstance(self.env.observation_space, spaces.Dict)
        if len(self.observation_shape) > 1:
            maps = {f"population_map_{i}": spaces.Box(low=0, high=np.inf, shape=shape, dtype=np.float64) for i, shape in
                    enumerate(self.observation_shape)}
        else:
            maps = {"population_map": spaces.Box(low=0, high=np.inf, shape=self.observation_shape[0], dtype=np.float64)}

        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            **maps
        })

        self.base_env.fuel_to_noise_ratio = config.fuel_to_noise_ratio
        self.base_env.add_reward_component(self._get_noise_reward)

    @property
    def composite_window_size(self) -> tuple[int, int]:
        return self.base_env.window_size[0] + sum(x_size for x_size, _ in self._get_panel_sizes()), \
            self.base_env.window_size[1]

    def reset(self, seed=None, options=None):
        # Reset the base env first so that np_random is seeded
        observation, info = self.env.reset(seed=seed, options=options)

        # Now regenerate the map using the seeded random generator
        self.map_source.regenerate(rng=self.base_env.np_random)
        self.map_source_max = self.map_source.max # Cache the max value for the map normalization wrapper
        self._update_population_observation()
        observation = self._inject_population_observation(observation)

        self.background_map = self.get_background()

        # Reset noise tracking variables
        self.total_episode_noise = 0.0
        self.total_episode_noise_reward = 0.0
        self.mean_step_noise = self.noise_model.calculate_mean_step_noise(self.background_map,
                                                                          self.base_env.get_aircraft_altitude(),
                                                                          self.base_env.sim_dt)

        if self.render_mode is not None:
            self.background_max = np.nanmax(self.background_map)
            self.render_normalizer = self._get_normalization(self.background_map)

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
        observations = {f"population_map":
            self.raster_sampler.get_observation_clipped(center_position=ac_pos, orientation=ac_hdg, out_shape=obs_shape,
                                                        out_meters=obs_range) for
            i, (obs_shape, obs_range) in enumerate(zip(self.observation_shape, self.observation_range))}
        self.population_observation = observations
        return

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

        noise_kernel_shape_meters, noise_kernel_shape_pixels = self.noise_model.get_noise_power_kernel_shape_meters_and_pixels(
            ac_alt)

        population_map_extract = self.raster_sampler.get_observation_clipped(center_position=ac_pos,
                                                                             orientation=0,
                                                                             out_shape=noise_kernel_shape_pixels,
                                                                             out_meters=noise_kernel_shape_meters)

        step_normalized_noise = self.noise_model.step_normalized_noise(population_map_extract, ac_alt,
                                                                       self.mean_step_noise, sim_dt)
        noise_penalty = - (
                    1 - self.base_env.fuel_to_noise_ratio) * step_normalized_noise * self.base_env.dense_reward_scaling

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
        return [(int((obs_range[0] / obs_range[1]) * self.base_env.window_size[0]), y_size) for obs_range in
                self.observation_range]

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
        return [partial(self._render_array, render_size=size,array=observation) for size, observation in
                zip(self._get_panel_sizes(), self.population_observation.values())]

    def _get_normalization(self, heatmap: np.ndarray) -> Normalize:
        """Get the appropriate matplotlib Normalize instance based on config."""
        v_min = np.nanmin(heatmap)
        v_max = np.nanpercentile(heatmap, 99)  # Use 99th percentile to avoid outliers dominating the color scale

        if v_min == v_max:
            return Normalize(vmin=0, vmax=v_max)

        if self.config.rendering_normalization == "log":
            return FuncNorm(functions=(np.log1p, np.expm1), vmin=v_min, vmax=v_max)
        elif self.config.rendering_normalization == "min_max":
            return Normalize(vmin=v_min, vmax=v_max)
        else:  # "none" or default
            return Normalize(vmin=0, vmax=1)

    def _convert_heatmap_to_rgba_array(self, population_map: np.ndarray) -> np.ndarray:
        # Mask the area that has no data available ( negative population density )
        no_data_mask = np.isnan(population_map)

        normalized_map = self.render_normalizer(population_map)

        color_data = matplotlib.colormaps[self.color_map](normalized_map)
        rgba_array = (color_data * 255).astype(np.uint8)

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
        for obs_shape, obs_range in zip(self.observation_shape, self.observation_range):
            corners = self.raster_sampler.get_view_corners(center_position=ac_pos,
                                                           orientation=ac_hdg,
                                                           out_shape=obs_shape,
                                                           out_meters=obs_range)
            pygame.draw.polygon(canvas, pygame.color.Color("red"), corners, width=2)


