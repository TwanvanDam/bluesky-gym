import itertools
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import bluesky as bs
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import pyproj
from matplotlib.path import Path

import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.common.screen_dummy import ScreenDummy


class TerminationReason(Enum):
    SUCCESS = "success"
    OUT_OF_BOUNDS = "out_of_bounds"
    FAILED_APPROACH = "failed_approach"
    MAX_STEPS = "max_steps"
    NONE = "none"


class EpisodeTerminationTracker:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.episode_reasons = deque(maxlen=window_size)

    def log_episode(self, reason: TerminationReason):
        if len(self.episode_reasons) == self.window_size:
            self.episode_reasons.popleft()
        self.episode_reasons.append(reason)

    def get_statistics(self) -> dict:
        if not self.episode_reasons:
            return {}

        stats = {reason.value: 100 * sum(1 / 50 for episode_reason in self.episode_reasons if episode_reason == reason)
                 for reason in TerminationReason}
        return stats


@dataclass
class Position:
    lat: float
    lon: float


@dataclass
class Airport:
    position: Position
    hdg: float


@dataclass
class NavigationConfig:
    ac_name: str = "KL001"
    ac_type: str = "a320"
    ac_initial_spd: int = 200  # [ m/s ]
    ac_initial_alt: int = 3_000  # [ m ]

    # WGS84 [ degrees ]
    lon_min: float = 3.0
    lon_max: float = 7.5
    lat_min: float = 50.5
    lat_max: float = 54.0
    airport_lat: float = 52.31
    airport_lon: float = 4.7

    pygame_crs: str = "EPSG:3035"

    # Simulation Parameters
    max_steps: int = 250
    sim_dt: int = 3  # s
    action_time: int = 60  # s

    # Termination conditions
    faf_distance: float = 25  # km
    iaf_angle: float = 60  # degrees
    iaf_distance: float = 30  # km


class BaseNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None, window_size: tuple[int, int] = (512, 512),
                 config: NavigationConfig = NavigationConfig()):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.ac_name = config.ac_name
        self.ac_type = config.ac_type
        self.ac_initial_spd = config.ac_initial_spd  # [m/s]
        self.ac_initial_alt = config.ac_initial_alt  # [m]

        self.bluesky_crs = "WGS84"
        self.pygame_crs = config.pygame_crs
        self.coordinate_transformer = pyproj.Transformer.from_crs(
            self.bluesky_crs,
            self.pygame_crs,
            always_xy=True
        )

        self.lon_min, self.lon_max = config.lon_min, config.lon_max
        self.lat_min, self.lat_max = config.lat_min, config.lat_max
        self.airport_lon, self.airport_lat = config.airport_lon, config.airport_lat

        self.lon_center = (self.lon_max + self.lon_min) / 2
        self.lat_center = (self.lat_max + self.lat_min) / 2

        self.x_min, self.y_min = self.coordinate_transformer.transform(self.lon_min, self.lat_min)
        self.x_max, self.y_max = self.coordinate_transformer.transform(self.lon_max, self.lat_max)

        self.observation_space = spaces.Dict(
            {
                "IAF_slant_range": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "Heading_to_Airport": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "Airport_Azimuth": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
            }
        )

        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        self._reward_components: list[Callable] = [
            self._fuel_reward,
            self._termination_reward,
            self._boundary_reward,
            self._truncation_reward,
        ]

        self.episode_tracker = EpisodeTerminationTracker()

        self.max_steps = config.max_steps
        self.sim_dt = config.sim_dt  # s
        self.action_time = config.action_time
        self.action_frequency = int(self.action_time / self.sim_dt)
        self.current_step: int | None = None

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack(f'DT {self.sim_dt};FF')

        self.fuel_used: float | None = None
        self.airport_details: Airport | None = None
        self.aircraft_positions: list[Position] = []

        self.faf_distance = config.faf_distance  # [ km ]
        self.iaf_angle = config.iaf_angle  # [ degrees ]
        self.iaf_distance = config.iaf_distance  # [ km ]

        self.aircraft_length = 20  # [ pixels ]
        self.aircraft_width = 10  # [ pixels ]
        self.aircraft_heading_length = 50  # [ pixels ]

        self.airport_length = 30  # [ pixels ]
        self.airport_width = 10  # [ pixels ]
        self.faf_radius = 30  # [ pixels ]

        # pygame variables
        self.window_size = window_size
        self.window: pygame.Surface | None = None
        self.clock = None
        self.blue_background = pygame.Color(135, 206, 235)

    def reset(self, seed=None, options=None):
        bs.traf.reset()
        super().reset(seed=seed)

        self.current_step = 0
        self.fuel_used = 0.0

        self.airport_details = self._generate_airport(self.np_random)
        self._set_terminal_condition()

        aircraft_initial_position = self._generate_initial_position(self.np_random)
        self.aircraft_positions = [aircraft_initial_position]
        heading_to_airport = fn.get_hdg((aircraft_initial_position.lat, aircraft_initial_position.lon),
                                        (self.airport_details.position.lat, self.airport_details.position.lon))
        bs.traf.cre(self.ac_name, actype=self.ac_type, aclat=aircraft_initial_position.lat,
                    aclon=aircraft_initial_position.lon,
                    achdg=heading_to_airport, acspd=self.ac_initial_spd)

        if self.render_mode is not None:
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        _, ac_hdg = self.get_aircraft_details()
        new_heading = fn.bound_angle_0_360(ac_hdg + action[0] * 180)
        bs.stack.stack(f"HDG {self.ac_name} {new_heading}")

        for i in range(self.action_frequency):
            bs.sim.step()
            ac_pos, _ = self.get_aircraft_details()
            self.aircraft_positions.append(ac_pos)

            if self._get_terminal_condition()[1]:
                break

        reward, terminated, truncated = self._get_reward()
        self.current_step += 1

        observation = self._get_obs()
        info = {}

        if terminated or truncated:
            info["termination_stats"] = self.episode_tracker.get_statistics()
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        elif self.render_mode is not None:
            self.render()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _get_obs(self):
        ac_position, ac_hdg = self.get_aircraft_details()

        correct_heading = (fn.get_hdg((ac_position.lat, ac_position.lon),
                                      (self.faf_lat, self.faf_lon)))

        heading_to_airport = fn.bound_angle_positive_negative_180(correct_heading - ac_hdg) / 180
        airport_azimuth = fn.bound_angle_positive_negative_180(self.airport_details.hdg - ac_hdg) / 180

        observation = {
            "IAF_slant_range": np.array(
                [np.sqrt((self.faf_lat - ac_position.lat) ** 2 + (self.faf_lon - ac_position.lon) ** 2)],
                dtype=np.float64),
            "Heading_to_Airport": np.array([np.clip(heading_to_airport, -1, 1)], dtype=np.float64),
            "Airport_Azimuth": np.array([np.clip(airport_azimuth, -1, 1)], dtype=np.float64),
        }
        return observation

    def get_aircraft_details(self) -> tuple[Position, float]:
        ac_idx = bs.traf.id2idx(self.ac_name)

        ac_hdg = bs.traf.hdg[ac_idx]
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]
        return Position(lat=ac_lat, lon=ac_lon), ac_hdg

    def _get_reward(self):
        total_reward = 0.0
        terminated = False
        termination_reason = TerminationReason.NONE
        truncated = self.current_step >= self.max_steps

        for component in self._reward_components:
            component_reward, component_terminated, reason = component()
            total_reward += component_reward
            if component_terminated and termination_reason == TerminationReason.NONE:
                self.episode_tracker.log_episode(reason=reason)
            terminated = terminated or component_terminated

        return total_reward, terminated, truncated

    def add_reward_component(self, function: Callable) -> None:
        self._reward_components.append(function)

    def _fuel_reward(self, coeff: float = 0.025) -> tuple[float, bool, TerminationReason]:
        ac_idx = bs.traf.id2idx(self.ac_name)
        fuel_flow = bs.traf.perf.fuelflow[ac_idx]
        terminated = False
        return - coeff * fuel_flow, terminated, TerminationReason.NONE

    def _boundary_reward(self) -> tuple[float, bool, TerminationReason]:
        if self._check_out_of_bounds():
            return -1.0, True, TerminationReason.OUT_OF_BOUNDS
        else:
            return 0.0, False, TerminationReason.NONE

    def _truncation_reward(self) -> tuple[float, bool, TerminationReason]:
        if self.current_step >= self.max_steps:
            return -1.0, False, TerminationReason.MAX_STEPS
        return 0.0, False, TerminationReason.NONE

    def _termination_reward(self) -> tuple[float, bool, TerminationReason]:
        return self._get_terminal_condition()

    def _set_terminal_condition(self):
        """Adapted from PathPlanningEnv by Groot et al."""
        num_points = 36

        airport_lat = self.airport_details.position.lat
        airport_lon = self.airport_details.position.lon
        airport_hdg = self.airport_details.hdg

        self.faf_lat, self.faf_lon = fn.get_point_at_distance(airport_lat, airport_lon, self.faf_distance,
                                                              fn.bound_angle_0_360(airport_hdg + 180))
        cw_bound = fn.bound_angle_0_360(airport_hdg + 180) + (self.iaf_angle / 2)
        ccw_bound = fn.bound_angle_0_360(airport_hdg + 180) - (self.iaf_angle / 2)

        angles = np.linspace(cw_bound, ccw_bound, num_points)
        iaf_lat, iaf_lon = fn.get_point_at_distance(self.faf_lat, self.faf_lon, self.iaf_distance, angles)
        self.iaf_center_lat, self.iaf_center_lon = iaf_lat[num_points // 2], iaf_lon[num_points // 2]

        command = f"POLYLINE SINK"
        for lat, lon in zip(iaf_lat, iaf_lon):
            command += f" {lat} {lon}"
        bs.stack.stack(command)

        bs.stack.stack(
            f"POLYLINE RESTRICT {iaf_lat[0]} {iaf_lon[0]} {self.faf_lat} {self.faf_lon} {iaf_lat[-1]} {iaf_lon[-1]}")
        bs.sim.step()

    def _get_terminal_condition(self) -> tuple[float, bool, TerminationReason]:
        terminated = False
        reason = TerminationReason.NONE
        reward = 0

        shapes = bs.tools.areafilter.basic_shapes
        current_pos, _ = self.get_aircraft_details()
        if self.aircraft_positions:
            last_pos = self.aircraft_positions[-1]
            line_ac = Path(np.array([[last_pos.lat, last_pos.lon], [current_pos.lat, current_pos.lon]]))
            line_sink = Path(np.reshape(shapes["SINK"].coordinates, (len(shapes["SINK"].coordinates) // 2, 2)))
            line_restrict = Path(
                np.reshape(shapes["RESTRICT"].coordinates, (len(shapes["RESTRICT"].coordinates) // 2, 2)))

            if line_sink.intersects_path(line_ac):
                reward = 50
                reason = TerminationReason.SUCCESS
                terminated = True

            elif line_restrict.intersects_path(line_ac):
                reward = -1
                reason = TerminationReason.FAILED_APPROACH
                terminated = True

        return reward, terminated, reason

    def _check_out_of_bounds(self) -> bool:
        aircraft_position, aircraft_heading = self.get_aircraft_details()
        aircraft_inside_bounds = (self.lat_min <= aircraft_position.lat <= self.lat_max) and (
                    self.lon_min <= aircraft_position.lon <= self.lon_max)
        return not aircraft_inside_bounds

    def _generate_airport(self, np_random: np.random.Generator) -> Airport:
        return Airport(
            Position(lat=self.airport_lat,
                     lon=self.airport_lon),
            hdg=float(np_random.integers(low=1, high=36) * 10)
        )

    def _generate_initial_position(self, np_random: np.random.Generator) -> Position:
        return Position(
            lat=self.airport_details.position.lat + np_random.normal(loc=0, scale=1),
            lon=self.airport_details.position.lon + np_random.normal(loc=0, scale=1)
        )

    def lat_lon_to_pix(self, position: Position) -> tuple[int, int]:
        x_meters, y_meters = self.coordinate_transformer.transform(position.lon, position.lat)

        return self.meters_to_pix((x_meters, y_meters))

    def meters_to_pix(self, position_meters: tuple[float, float]) -> tuple[int, int]:
        norm_x = (position_meters[0] - self.x_min) / (self.x_max - self.x_min)
        norm_y = (position_meters[1] - self.y_min) / (self.y_max - self.y_min)
        screen_x = int(norm_x * self.window_size[0])
        screen_y = int((1 - norm_y) * self.window_size[1])
        return screen_x , screen_y

    def render(self):
        if self.render_mode is None:
            return None

        canvas = self.initialize_pygame(self.window_size)
        self._handle_pygame_events()

        for draw_function in self.get_render_layers():
            draw_function(canvas)

        return self._present_canvas(canvas, self.render_mode)

    def get_render_layers(self) -> list[Callable]:
        """Return a list of functions that can be run to render the environment."""
        return [self._draw_background,
                self._draw_airport,
                self._draw_aircraft,
                self._draw_observation_text]

    def initialize_pygame(self, canvas_size: tuple[int, int]):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(canvas_size)
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface(canvas_size)
        return canvas

    def _present_canvas(self, canvas: pygame.Surface, render_mode: str | None) -> None | np.ndarray:
        if render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
        return None

    def _handle_pygame_events(self) -> None:
        if self.window is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def _draw_background(self, canvas: pygame.Surface) -> None:
        canvas.fill(self.blue_background)

    def _draw_airport(self, canvas):
        airport_color = pygame.Color("black")
        red_dot_color = pygame.Color("red")

        airport_x_position, airport_y_position = self.lat_lon_to_pix(self.airport_details.position)
        shapes = bs.tools.areafilter.basic_shapes
        line_sink = np.reshape(shapes["SINK"].coordinates, (len(shapes["SINK"].coordinates) // 2, 2))
        line_restrict = np.reshape(shapes["RESTRICT"].coordinates, (len(shapes["RESTRICT"].coordinates) // 2, 2))

        airport_surface = pygame.Surface((self.airport_width, self.airport_length), pygame.SRCALPHA)
        airport_surface.fill(airport_color)
        rotated_airport_surface = pygame.transform.rotate(airport_surface, -self.airport_details.hdg)
        airport_rect = rotated_airport_surface.get_rect(center=(airport_x_position, airport_y_position))
        canvas.blit(rotated_airport_surface, airport_rect)
        pygame.draw.circle(canvas, red_dot_color, (int(airport_x_position), int(airport_y_position)), 5)

        self._draw_line_from_points(canvas, airport_color, list(line_sink))

        self._draw_line_from_points(canvas, airport_color, list(line_restrict))

    def _draw_aircraft(self, canvas):
        aircraft_color = pygame.Color("black")
        ac_position, ac_heading = self.get_aircraft_details()

        red_line_color = pygame.Color("red")
        for point_1, point_2 in itertools.pairwise(self.aircraft_positions):
            x1, y1 = self.lat_lon_to_pix(point_1)
            x2, y2 = self.lat_lon_to_pix(point_2)
            pygame.draw.line(canvas, red_line_color, (x1, y1), (x2, y2), 2)

        ac_x_position, ac_y_position = self.lat_lon_to_pix(ac_position)

        heading_end_x = ac_x_position + np.sin(np.deg2rad(ac_heading)) * self.aircraft_heading_length
        heading_end_y = ac_y_position - np.cos(np.deg2rad(ac_heading)) * self.aircraft_heading_length

        ac_surface = pygame.Surface((self.aircraft_width, self.aircraft_length), pygame.SRCALPHA)
        ac_surface.fill(aircraft_color)
        rotated_ac_surface = pygame.transform.rotate(ac_surface, -ac_heading)
        ac_rect = rotated_ac_surface.get_rect(center=(ac_x_position, ac_y_position))
        canvas.blit(rotated_ac_surface, ac_rect)

        pygame.draw.line(canvas,
                         aircraft_color,
                         (ac_x_position, ac_y_position),
                         (heading_end_x, heading_end_y),
                         width=2
                         )

    def _draw_line_from_points(self, canvas: pygame.Surface, color: pygame.Color, points: list[Position]) -> None:
        for point_1, point_2 in itertools.pairwise(points):
            x1, y1 = self.lat_lon_to_pix(Position(lat=point_1[0], lon=point_1[1]))
            x2, y2 = self.lat_lon_to_pix(Position(lat=point_2[0], lon=point_2[1]))
            pygame.draw.line(canvas, color, (x1, y1), (x2, y2), 2)

    def _draw_observation_text(self, canvas):
        """Draw observation values as text in the upper-left corner."""
        font = pygame.font.Font(None, 24)
        text_color = pygame.Color("black")

        obs = self._get_obs()
        y_offset = 10
        obs = {**obs, "airport_bearing": np.array([self.airport_details.hdg])}

        for key, value in obs.items():
            if "Heading" in key or "Azimuth" in key:
                text = f"{key}: {value[0] * 180:.4f}"
            else:
                text = f"{key}: {value[0]:.4f}"
            text_surface = font.render(text, True, text_color)
            canvas.blit(text_surface, (10, y_offset))
            y_offset += 30
