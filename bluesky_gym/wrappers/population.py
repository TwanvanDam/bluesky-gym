from functools import partial
from typing import Callable

import gymnasium as gym
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


class Population(gym.Wrapper):
    def __init__(self, env: BaseNavigationEnv, map_source: MapSource, observation_shape: tuple[int, int],
                 observation_range: tuple[int, int], render_mode: str | None = None, color_map: str = "Blues"):
        assert isinstance(env, BaseNavigationEnv)
        super().__init__(env)
        self.env: BaseNavigationEnv = env
        self._render_mode = render_mode
        self.window = None
        self.observation_shape = observation_shape
        self.observation_range = observation_range
        self.population_observation = None

        # class to handle all reading and creating of population maps
        self.map_source = map_source
        self.transformer = Transformer.from_crs(self.env.bluesky_crs, self.env.pygame_crs, always_xy=True)

        # cache the map used as background since it does not change often.
        self.background_map = None
        self.color_map: str = color_map
        self.metadata = env.metadata.copy()

        assert isinstance(self.env.observation_space, spaces.Dict)
        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            "population_map": spaces.Box(low=0, high=np.inf, shape=self.observation_shape, dtype=np.float64)
        })
        self.env.add_reward_component(self._get_noise_reward)


    @property
    def window_size(self) -> tuple[int,int]:
        return 2 * self.env.window_size[0], self.env.window_size[1]

    def reset(self, seed=None, options=None):
        self.map_source.regenerate()
        self.background_map = self._load_background()

        observation, info = self.env.reset(seed=seed, options=options)
        self.population_observation = self._get_population_observation()
        observation = {**observation, "population_map": self.population_observation}

        self.render()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            self.population_observation = self._get_population_observation()
        observation = {**observation, "population_map": self.population_observation}
        if not (terminated or truncated):
            self.render()
        return observation, reward, terminated, truncated, info

    def close(self):
        """Close the rasterio dataset when done"""
        self.map_source.close()
        self.env.close()

    def _extract_view_from_map(self, center_position: Position, orientation: float, out_shape: tuple[int, int], out_meters: tuple[float, float]):
        center_xy = self.transformer.transform(center_position.lon, center_position.lat)

        # Calculate the resolution (meters per pixel) for the output slice
        res_x = out_meters[0] / out_shape[0]
        res_y = out_meters[1] / out_shape[1]

        dst_transform = (
                Affine.translation(*center_xy) *
                Affine.rotation(- orientation) *
                Affine.scale(res_x, -res_y) *
                Affine.translation(-out_shape[0] / 2, -out_shape[1] / 2)
        )

        destination = np.zeros(out_shape)

        # Perform the Warp (Reprojection)
        reproject(
            source=rasterio.band(self.map_source.dataset, 1),  # always a dataset now
            destination=destination,
            src_transform=self.map_source.transform,
            src_crs=self.map_source.crs,
            dst_transform=dst_transform,
            dst_crs=self.env.pygame_crs,
            resampling=Resampling.bilinear  # Use 'nearest' for categorical data (masks)
        )
        return destination

    def _get_population_observation(self):
        position, heading = self.env.get_aircraft_details()
        destination = self._extract_view_from_map(position, heading, self.observation_shape, self.observation_range)
        destination = np.clip(destination, 0, np.inf)
        return destination

    def _load_background(self):
        center_position = Position(lon=self.env.lon_center, lat=self.env.lat_center)
        out_meters = self.env.x_max - self.env.x_min, self.env.y_max - self.env.y_min
        return self._extract_view_from_map(center_position, 0, self.env.window_size, out_meters)

    def _get_noise_reward(self) -> tuple[float, bool, TerminationReason]:
        return 0.0, False, TerminationReason.NONE

    def render(self):
        if self._render_mode is None:
            return None

        # Use extended window size
        canvas = self.env.initialize_pygame(self.window_size)
        self.env._handle_pygame_events()

        canvas.fill(pygame.Color("grey"))

        for draw_function in self.get_render_layers():
            draw_function(canvas)

        return self.env._present_canvas(canvas, render_mode=self._render_mode)

    def get_render_layers(self) -> list[Callable]:
        """Override to insert custom layers into rendering pipeline."""
        return [
            partial(self._render_array, position=(0,0), array=self.background_map, transparent=True),
            partial(self._render_array, position=(512,0), array=self.population_observation, transparent=False),
            self.env._draw_airport,
            self.env._draw_aircraft,
            self._draw_box_around_aircraft,
            # self.env._draw_observation_text,
        ]

    def _convert_heatmap_to_rgba_array(self, population_map: np.ndarray) -> np.ndarray:
        epsilon = 1e-10
        normalized_map = population_map.copy()

        sea_mask = normalized_map < 0
        normalized_map = np.clip(normalized_map, epsilon, np.inf)
        normalized_map = np.log1p(normalized_map + epsilon)

        if normalized_map.max() > normalized_map.min():
            normalized_map = (normalized_map - normalized_map.min()) / (normalized_map.max() - normalized_map.min())
        else:
            normalized_map = np.zeros_like(normalized_map)

        color_data = matplotlib.colormaps[self.color_map](normalized_map)
        rgba_array = (color_data * 255).astype(np.uint8)
        rgba_array[sea_mask, 3] = 0
        # return np.transpose(rgba_array, (1, 0, 2))
        return rgba_array

    def _render_array(self, canvas: pygame.Surface, position: tuple[int,int], array: np.ndarray, transparent:bool=True) -> None:
        rgba_array = self._convert_heatmap_to_rgba_array(array)
        if transparent:
            heatmap_surf = pygame.image.frombuffer(rgba_array.tobytes(), rgba_array.shape[:2], "RGBA")
        else:
            heatmap_surf = pygame.image.frombuffer(rgba_array[:, :, :3].tobytes(), rgba_array.shape[:2], "RGB")
        heatmap_size = self.env.window_size
        heatmap_surf = pygame.transform.scale(heatmap_surf, heatmap_size)

        canvas.blit(heatmap_surf, position)

    def _draw_box_around_aircraft(self, canvas):
        ac_position, ac_heading = self.env.get_aircraft_details()

        x_meters, y_meters = self.env.coordinate_transformer.transform(ac_position.lon, ac_position.lat)
        half_w = self.observation_range[0] / 2
        half_h = self.observation_range[1] / 2

        corners = [(-half_w, -half_h), (-half_w, half_h), (half_w, half_h), (half_w, -half_h)]
        angle = np.deg2rad(-ac_heading)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        rotated = []
        for dx, dy in corners:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            x = x_meters + rx
            y = y_meters + ry

            norm_x = (x - self.env.x_min) / (self.env.x_max - self.env.x_min)
            norm_y = (y - self.env.y_min) / (self.env.y_max - self.env.y_min)
            screen_x = norm_x * self.env.window_size[0]
            screen_y = (1 - norm_y) * self.env.window_size[1]
            rotated.append((screen_x, screen_y))

        pygame.draw.polygon(canvas, pygame.color.Color("red"), rotated, width=2)






