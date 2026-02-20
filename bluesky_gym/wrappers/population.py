import gymnasium as gym
from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
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
        self._render_mode = render_mode
        self.window = None
        self.observation_shape = observation_shape
        self.observation_range = observation_range
        self.population_observation = None

        # class to handle all reading and creating of population maps
        self.map_source = map_source

        # cache the map used as background since it does not change often.
        self.background_map = None
        self.color_map: str = color_map
        self.metadata = env.metadata.copy()

        assert isinstance(self.env.observation_space, spaces.Dict)
        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            "population_map": spaces.Box(low=0, high=np.inf, shape=self.observation_shape, dtype=np.float64)
        })

    def _load_background(self):
        transformer = Transformer.from_crs("WGS84", self.map_source.crs, always_xy=True)

        # Example: Convert Netherlands (Lat: 52, Lon: 4.5)
        lon, lat = self.env.lon_center, self.env.lat_center
        center_xy = transformer.transform(lon, lat)

        out_h, out_w = (self.env.window_size, self.env.window_size)
        width_m, height_m = self.env.x_max - self.env.x_min, self.env.y_max - self.env.y_min

        # Calculate the resolution (meters per pixel) for the output slice
        res_x = width_m / out_w
        res_y = height_m / out_h

        dst_transform = (
                Affine.translation(*center_xy) *
                Affine.scale(res_x, -res_y) *
                Affine.translation(-out_w / 2, -out_h / 2)
        )
        # Create the destination array
        # Shape is (count, out_h, out_w) to keep all bands
        destination = np.zeros((out_h, out_w))

        # Perform the Warp (Reprojection)
        reproject(
            source=rasterio.band(self.map_source.dataset, 1),  # always a dataset now
            destination=destination,
            src_transform=self.map_source.transform,
            src_crs=self.map_source.crs,
            dst_transform=dst_transform,
            dst_crs=self.map_source.crs,
            resampling=Resampling.bilinear  # Use 'nearest' for categorical data (masks)
        )
        return destination

    def _get_population_observation(self):
        transformer = Transformer.from_crs("WGS84", self.map_source.crs, always_xy=True)

        # Example: Convert Netherlands (Lat: 52, Lon: 4.5)
        position, heading = self.env.get_aircraft_details()
        center_xy = transformer.transform(position.lon, position.lat)

        out_h, out_w = self.observation_shape
        width_m, height_m = self.observation_range

        # Calculate the resolution (meters per pixel) for the output slice
        res_x = width_m / out_w
        res_y = height_m / out_h

        dst_transform = (
                Affine.translation(*center_xy) *
                Affine.rotation(- heading) *
                Affine.scale(res_x, -res_y) *
                Affine.translation(-out_w / 2, -out_h / 2)
        )

        destination = np.zeros((out_h, out_w))

        # Perform the Warp (Reprojection)
        reproject(
            source=rasterio.band(self.map_source.dataset, 1),  # always a dataset now
            destination=destination,
            src_transform=self.map_source.transform,
            src_crs=self.map_source.crs,
            dst_transform=dst_transform,
            dst_crs=self.map_source.crs,
            resampling=Resampling.bilinear  # Use 'nearest' for categorical data (masks)
        )
        destination = np.clip(destination, 0, np.inf)
        return destination

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            self.population_observation = self._get_population_observation()
        observation = {**observation, "population_map": self.population_observation}
        if not (terminated or truncated):
            self.render()
        return observation, reward, terminated, truncated, info

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

    def reset(self, seed=None, options=None):
        super().reset(seed=None, options=None)
        self.map_source.regenerate()
        self.background_map = self._load_background()
        self.population_observation = self._get_population_observation()

        observation, info = self.env.reset(seed=seed, options=options)

        observation = {**observation, "population_map": self.population_observation}

        self.render()
        return observation, info

    def _array_background(self, canvas):
        background = self._convert_heatmap_to_rgba_array(self.background_map)
        heatmap_surf = pygame.image.frombuffer(background.tobytes(), background.shape[:2], "RGBA")
        heatmap_size = (self.env.window_size, self.env.window_size)
        heatmap_surf = pygame.transform.scale(heatmap_surf, heatmap_size)

        center = list(heatmap_dim / 2 for heatmap_dim in heatmap_size)
        canvas.blit(heatmap_surf, (center[0] - heatmap_size[0] / 2, center[1] - heatmap_size[1] / 2))

    def _render_population_map_observation(self, canvas):
        observation = self._convert_heatmap_to_rgba_array(self.population_observation)

        heatmap_surf = pygame.image.frombuffer(observation[:, :, :3].tobytes(), observation.shape[:2], "RGB")

        heatmap_size = (self.env.window_size, self.env.window_size)
        heatmap_surf = pygame.transform.scale(heatmap_surf, heatmap_size)

        center = (1.5 * heatmap_size[0], 0.5 * heatmap_size[1])
        canvas.blit(heatmap_surf, (center[0] - heatmap_size[0] / 2, center[1] - heatmap_size[1] / 2))

    def _initialize_pygame(self):
        if self.window is None and self._render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((2 * self.env.window_size, self.env.window_size))
            self.clock = pygame.time.Clock()

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
            screen_x = norm_x * self.env.window_size
            screen_y = self.env.window_size - (norm_y * self.env.window_size)
            rotated.append((screen_x, screen_y))

        pygame.draw.polygon(canvas, pygame.color.Color("red"), rotated, width=2)

    def render(self):
        if self._render_mode is None:
            return None

        self._initialize_pygame()  # Only initializes if needed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        canvas = pygame.Surface((2 * self.env.window_size, self.env.window_size))
        canvas.fill(pygame.color.Color("grey"))

        self._array_background(canvas)

        self.env._draw_aircraft(canvas)
        self._draw_box_around_aircraft(canvas)

        self.env._draw_airport(canvas)
        self.env._draw_observation_text(canvas)

        self._render_population_map_observation(canvas)

        if self._render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            # pygame.event.pump()  # Process event queue to prevent flickering/freezing
            pygame.display.update()
            self.clock.tick(self.env.metadata["render_fps"])
        elif self._render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
        return None

    def close(self):
        """Close the rasterio dataset when done"""
        self.map_source.close()
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
        super().close()


