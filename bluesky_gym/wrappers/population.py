from functools import partial
from typing import Callable

import gymnasium as gym
from bluesky.tools.aero import ft

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv, TerminationReason, Position
import pygame
from gymnasium import spaces
import matplotlib
import rasterio
import rasterio.features
from rasterio.warp import reproject, Resampling
from affine import Affine
import numpy as np
from pyproj import Transformer
from bluesky_gym.wrappers.map_datsets import MapSource
from scripts.config import PopulationConfig


class Population(gym.Wrapper):
    def __init__(self, env: gym.Env, config: PopulationConfig = PopulationConfig(), color_map: str = "Blues"):
        assert isinstance(env, BaseNavigationEnv)
        super().__init__(env)
        self.env: gym.Env = env
        self.base_env: BaseNavigationEnv = self.unwrapped

        self.base_env._render_owned_by_wrapper = True
        self.config = config

        self.window = None
        self.observation_shape = config.observation_shape
        self.observation_range = config.observation_range
        self.population_observation = None

        # class to handle all reading and creating of population maps
        self.map_source = config.map_source_config.build(self.base_env)

        # cache the map used as background since it does not change often.
        self.background_map = None
        self.color_map: str = color_map
        self.metadata = env.metadata.copy()

        assert isinstance(self.env.observation_space, spaces.Dict)
        if len(self.observation_shape) > 1:
            maps = {f"population_map_{i}": spaces.Box(low=0, high=np.inf, shape=shape, dtype=np.float64) for i, shape in enumerate(self.observation_shape)}
        else:
            maps = {"population_map" : spaces.Box(low=0, high=np.inf, shape=self.observation_shape[0], dtype=np.float64)}

        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            **maps
        })

        self.base_env.fuel_to_noise_ratio = config.fuel_to_noise_ratio
        self.base_env.add_reward_component(self._get_noise_reward)


    @property
    def composite_window_size(self) -> tuple[int,int]:
        return self.base_env.window_size[0] + sum(x_size for x_size,_ in self._get_panel_sizes()), self.base_env.window_size[1]

    def reset(self, seed=None, options=None):
        self.map_source.regenerate()
        self.background_map = self._load_background()

        observation, info = self.env.reset(seed=seed, options=options)
        self.population_observation = self._get_population_observation()
        observation = {**observation, "population_map": self.population_observation}

        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # TODO: Verify that population observation updates should be skipped when episode ends
        if not done:
            self.population_observation = self._get_population_observation()
        observation = {**observation, "population_map": self.population_observation}

        if not done and self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def close(self):
        """Close the rasterio dataset when done"""
        self.map_source.close()
        self.env.close()

    def _extract_view_from_map(self, center_position: Position, orientation: float, out_shape: tuple[int, int], out_meters: tuple[float, float]):
        dst_transform = self._get_dst_transform(center_position, orientation, out_meters, out_shape)

        destination = np.zeros(out_shape[::-1])

        # Perform the Reprojection
        reproject(
            source=rasterio.band(self.map_source.dataset, 1),
            destination=destination,
            src_transform=self.map_source.transform,
            src_crs=self.map_source.crs,
            dst_transform=dst_transform,
            dst_crs=self.env.pygame_crs,
            resampling=getattr(Resampling, self.config.resampling)
        )
        return destination

    def _get_dst_transform(self, center_position: Position, orientation: float, out_meters: tuple[float, float],
                           out_shape: tuple[int, int]) -> tuple[Affine, ...]:
        center_xy = self.base_env.coordinate_transformer.transform(center_position.lon, center_position.lat)

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

    def _get_population_observation(self):
        position, heading = self.base_env.get_aircraft_details()
        observations = [np.clip(self._extract_view_from_map(position, heading, obs_shape, obs_range),0, np.inf) for obs_shape, obs_range in zip(self.observation_shape, self.observation_range)]
        return observations

    def _load_background(self):
        center_position = Position(lon=self.base_env.lon_center, lat=self.base_env.lat_center)
        out_meters = self.base_env.x_max - self.base_env.x_min, self.base_env.y_max - self.base_env.y_min
        background = self._extract_view_from_map(center_position, 0, self.base_env.window_size, out_meters)
        return background

    def _get_noise_reward(self) -> tuple[float, bool, TerminationReason]:
        base_noise = self.config.noise_base # [ dBA ]
        base_distance = 1000 * ft # [ft] -> [m]
        noise_cutoff = self.config.noise_cutoff # [ dBA ]
        W_0 = 1e-12

        base_noise_power = 10 ** (base_noise / 10) * W_0 # [ W ]
        noise_cutoff_power = 10 ** (noise_cutoff / 10) * W_0 # [ W ]
        base_noise_power_1m = base_noise_power / ( 1 / (base_distance ** 2)) # [ W ]
        noise_radius = np.sqrt(base_noise_power_1m/noise_cutoff_power) # [ m ] Distance where noise power is lower than cutoff
        noise_radius_rounded = self.config.noise_resolution * np.ceil(noise_radius / self.config.noise_resolution)

        altitude = self.base_env.get_aircraft_altitude()
        x = np.arange(-noise_radius_rounded, noise_radius_rounded + 1, self.config.noise_resolution)
        y = np.arange(-noise_radius_rounded, noise_radius_rounded + 1, self.config.noise_resolution)
        xv, yv = np.meshgrid(x, y)
        distance_squared = xv * xv + yv * yv + altitude * altitude

        ac_position, ac_heading = self.base_env.get_aircraft_details()
        area_around_ac = self._extract_view_from_map(ac_position, 0, distance_squared.shape, (2 * noise_radius_rounded, 2 * noise_radius_rounded))

        sound =  base_noise_power_1m / distance_squared
        sound[sound <= noise_cutoff_power] = 0
        total_noise = np.sum(area_around_ac*sound)
        return (total_noise / self.config.noise_penalty_coefficient) * (1 - self.base_env.fuel_to_noise_ratio), False, TerminationReason.NONE

    def render(self):
        # Use a canvas with composit_window_size
        self.base_env.initialize_pygame(self.composite_window_size)
        self.base_env.handle_pygame_events()
        base_surface = pygame.Surface(self.base_env.window_size)
        for draw_function in self.get_base_render_layers():
            draw_function(base_surface)

        canvas = pygame.Surface(self.composite_window_size)
        canvas.blit(base_surface, (0,0))

        render_dest = (self.base_env.window_size[0], 0)
        for draw_function, panel_size in zip(self.get_panel_render_layers(),self._get_panel_sizes()):
            panel_surface = pygame.Surface(panel_size)
            draw_function(panel_surface)
            canvas.blit(panel_surface, render_dest)
            render_dest = (render_dest[0]+panel_size[0], 0)

        return self.base_env.present_canvas(canvas)

    def _get_panel_sizes(self) -> list[tuple[int, int]]:
        y_size = self.base_env.window_size[1]
        return [(int((obs_range[0] / obs_range[1]) * self.base_env.window_size[0]), y_size) for obs_range in self.observation_range]

    def get_base_render_layers(self) -> list[Callable]:
        """Override to insert custom layers into rendering pipeline."""
        return [
            lambda canvas: canvas.fill(pygame.Color("grey")),
            partial(self._render_array, render_size=self.base_env.window_size, array=self.background_map,
                    transparent=True),
            self.base_env.draw_airport,
            self.base_env.draw_aircraft,
            self._draw_box_around_aircraft,
        ]

    def get_panel_render_layers(self) -> list[Callable]:
        return [partial(self._render_array, render_size=size,
                array=observation, transparent=False) for size, observation in zip(self._get_panel_sizes(), self.population_observation)]

    def normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        heatmap = np.clip(heatmap, 0, np.inf)

        if self.config.rendering_normalization == "log":
            epsilon = 1e-10
            heatmap = np.log1p(heatmap + epsilon)
        elif self.config.rendering_normalization == "min_max":
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        elif self.config.rendering_normalization == "none":
            pass

        return heatmap

    def _convert_heatmap_to_rgba_array(self, population_map: np.ndarray) -> np.ndarray:
        # Mask the area that has no data available ( negative population density )
        no_data_mask = population_map < 0

        # If the map has all the same values, return zeros
        if population_map.min() == population_map.max():
            normalized_map = np.zeros_like(population_map)

        else:
            normalized_map = self.normalize_heatmap(population_map)
            # Ensure values are on the interval [0, 1] for rendering
            if not (population_map.min() >= 0 and population_map.max() <= 1):
                normalized_map = (normalized_map - normalized_map.min()) / (normalized_map.max() - normalized_map.min())

        color_data = matplotlib.colormaps[self.color_map](normalized_map)
        rgba_array = (color_data * 255).astype(np.uint8)

        # Make areas without data transparent
        rgba_array[no_data_mask, 3] = 0
        return rgba_array

    def _render_array(self, canvas: pygame.Surface, render_size: tuple[int, int],
                      array: np.ndarray, transparent: bool = True) -> None:
        rgba_array = self._convert_heatmap_to_rgba_array(array)
        shape = array.shape[::-1]
        if transparent:
            heatmap_surf = pygame.image.frombuffer(rgba_array.tobytes(), shape , "RGBA")
        else:
            heatmap_surf = pygame.image.frombuffer(rgba_array[:, :, :3].tobytes(), shape, "RGB")
        heatmap_surf = pygame.transform.scale(heatmap_surf, render_size)

        canvas.blit(heatmap_surf, (0,0))

    def _get_view_corners_screen(self, center_position: Position, orientation: float,
                                 out_shape: tuple[int, int], out_meters: tuple[float, float]) -> list[
        tuple[float, float]]:
        dst_transform = self._get_dst_transform(center_position, orientation, out_meters, out_shape)

        cols, rows = out_shape
        pixel_corners = [(0, 0), (cols, 0), (cols, rows), (0, rows)]

        screen_corners = []
        for col, row in pixel_corners:
            x, y = dst_transform * (col, row)
            screen_x, screen_y = self.base_env.meters_to_pix((x, y))
            screen_corners.append((screen_x, screen_y))

        return screen_corners

    def _draw_box_around_aircraft(self, canvas):
        ac_position, ac_heading = self.base_env.get_aircraft_details()
        for obs_shape, obs_range in zip(self.observation_shape, self.observation_range):
            corners = self._get_view_corners_screen(ac_position, ac_heading,
                                                    obs_shape, obs_range)
            pygame.draw.polygon(canvas, pygame.color.Color("red"), corners, width=2)






