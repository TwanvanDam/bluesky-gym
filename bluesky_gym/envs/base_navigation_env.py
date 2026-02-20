import itertools
from dataclasses import dataclass

import bluesky as bs
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import pyproj
from matplotlib.path import Path

import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

@dataclass
class Position:
    lat: float
    lon: float


@dataclass
class Airport:
    position: Position
    hdg: float

def deg_to_360(angle: float) -> float:
    """Normalizes angle to the [0, 360] interval"""
    return (angle + 360) % 360

def deg_to_180(angle: float) -> float:
    """Normalizes angle to the [-180, -180] interval"""
    return ((angle + 180) % 360) - 180


class BaseNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode: str | None = None, ac_name:str="KL001", ac_type:str="a320", window_size:int=512, ac_initial_spd:int=200):
        self.ac_name = ac_name
        self.ac_type = ac_type
        self.ac_initial_spd = ac_initial_spd # [m/s]
        self.ac_initial_alt = 3_000 # [m]

        self.bluesky_crs = "EPSG:4326"
        self.map_crs = "EPSG:3035"

        self.coordinate_transformer = pyproj.Transformer.from_crs(
            self.bluesky_crs,
            self.map_crs ,
            always_xy=True
        )

        self.lon_min, self.lon_max = 3.0, 7.5
        self.lat_min, self.lat_max = 50.5, 54.0
        self.lon_center = (self.lon_max + self.lon_min) / 2
        self.lat_center = (self.lat_max + self.lat_min) / 2

        self.x_min, self.y_min = self.coordinate_transformer.transform(self.lon_min, self.lat_min)
        self.x_max, self.y_max = self.coordinate_transformer.transform(self.lon_max, self.lat_max)

        # pygame variables
        self.window_size = window_size
        self.window: pygame.Surface | None = None
        self.clock = None
        self.xy_to_px: float | None = None

        self.observation_space = spaces.Dict(
            {
                "FAF_slant_range": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "Heading_to_Airport": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "Airport_Azimuth": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
            }
        )

        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        self.max_steps = 500
        self.sim_dt = 5  # s
        self.action_time = 120
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
        self.aircraft_initial_position: Position | None = None
        self.aircraft_positions = []

        self.aircraft_length = 20 # pixels
        self.aircraft_width = 10 # pixels
        self.aircraft_heading_length = 50 # pixels

        self.faf_distance = 25 #km
        self.iaf_angle = 60 #degrees
        self.iaf_distance = 30 #km

        self.airport_length = 30
        self.airport_width = 10
        self.faf_radius = 30

        self.blue_background = pygame.Color(135, 206, 235)


    def reset(self, seed=None, options=None):
        bs.traf.reset()
        super().reset(seed=seed)

        self.fuel_used = 0.0
        self.airport_details = Airport(
            Position(lat=52.31,lon=4.76),
            hdg=float(np.random.randint(low=1, high=36) * 10)
        )

        self._set_terminal_condition()
        self.aircraft_initial_position = Position(lat=self.airport_details.position.lat + np.random.normal(loc=0, scale=1),
                                                    lon=self.airport_details.position.lon + np.random.normal(loc=0, scale=1))

        self.aircraft_positions = []

        heading_to_airport = fn.get_hdg((self.aircraft_initial_position.lat, self.aircraft_initial_position.lon),
                                       (self.airport_details.position.lat, self.airport_details.position.lon))

        bs.traf.cre(self.ac_name, actype=self.ac_type, aclat=self.aircraft_initial_position.lat,
                    aclon=self.aircraft_initial_position.lon,
                    achdg=heading_to_airport, acspd=self.ac_initial_spd)

        if self.render_mode is not None:
            self.render()
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        _, ac_hdg = self.get_aircraft_details()
        new_heading = deg_to_360(ac_hdg + action[0] * 90)
        bs.stack.stack(f"HDG {self.ac_name} {new_heading}")

        for i in range(self.action_frequency):
            bs.sim.step()
            ac_pos, _ = self.get_aircraft_details()
            self.aircraft_positions.append(ac_pos)
        self.current_step += 1

        observation = self._get_obs()
        reward, terminated, truncated = self._get_reward()
        info = {"step_reward" : reward}

        # bluesky reset?? bs.sim.reset()
        if terminated or truncated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)
        elif self.render_mode is not None:
            self.render()
        return observation, reward, terminated, truncated, info

    def _check_out_of_bounds(self) -> bool:
        aircraft_position, aircraft_heading = self.get_aircraft_details()
        aircraft_inside_bounds = (self.lat_min <= aircraft_position.lat <= self.lat_max) and (self.lon_min <= aircraft_position.lon <= self.lon_max)
        return not aircraft_inside_bounds

    def _bluesky_to_pygame(self, position: Position) -> tuple[int, int]:
        x_meters, y_meters = self.coordinate_transformer.transform(position.lon, position.lat)

        norm_x = (x_meters - self.x_min) / (self.x_max - self.x_min)
        norm_y = (y_meters - self.y_min) / (self.y_max - self.y_min)

        screen_x = int(norm_x * self.window_size)
        screen_y = int(self.window_size - (norm_y * self.window_size))
        return screen_x, screen_y

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _get_reward(self, fuel_coeff: float = 0.025):
        terminated, reward = self._get_terminal_condition()
        truncated = self.current_step >= self.max_steps

        ac_idx = bs.traf.id2idx(self.ac_name)
        fuel_flow = bs.traf.perf.fuelflow[ac_idx]

        if self._check_out_of_bounds():
            print("Out of Bounds")
            terminated = True
            reward += -1

        if truncated:
            reward += -1
            print("Truncated")

        #ac_position, ac_hdg = self.get_aircraft_details()
        #distance = np.sqrt((self.iaf_center_lat - ac_position.lat)**2 + (self.iaf_center_lon - ac_position.lon)**2)

        reward += - fuel_coeff * fuel_flow # - distance / 500
        return reward, terminated, truncated

    def get_aircraft_details(self) -> tuple[Position, float]:
        ac_idx = bs.traf.id2idx(self.ac_name)

        ac_hdg = bs.traf.hdg[ac_idx]
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]
        return Position(lat=ac_lat, lon=ac_lon), ac_hdg

    def _get_obs(self):
        ac_position, ac_hdg = self.get_aircraft_details()

        correct_heading = (fn.get_hdg((ac_position.lat, ac_position.lon),
                               (self.faf_lat, self.faf_lon)))

        heading_to_airport = deg_to_180(correct_heading - ac_hdg) / 180
        airport_azimuth = deg_to_180(self.airport_details.hdg - ac_hdg) / 180

        observation = {
            "FAF_slant_range": np.array([np.sqrt((self.faf_lat - ac_position.lat)**2 + (self.faf_lon - ac_position.lon)**2)], dtype=np.float64),
            "Heading_to_Airport": np.array([np.clip(heading_to_airport, -1, 1)], dtype=np.float64),
            "Airport_Azimuth": np.array([np.clip(airport_azimuth, -1, 1)], dtype=np.float64),
        }

        return observation

    def render(self):
        if self.render_mode is None:
            return None

        canvas = self._initialize_pygame()
        self._handle_pygame_events()

        self._draw_aircraft(canvas)

        self._draw_airport(canvas)

        self._draw_observation_text(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(canvas), (1,0,2))
        return None

    def lat_lon_to_screen_pos(self, position: Position) -> tuple[float, float]:
        """Converts lat, lon coordinates in the form of a Position object to pygame coordinates."""
        screen_center = self.airport_details.position

        screen_x = self.window_size /2 + self.xy_to_px * (position.lon - screen_center.lon)
        screen_y = self.window_size /2 - self.xy_to_px * (position.lat - screen_center.lat)

        return screen_x, screen_y


    def _draw_airport(self, canvas):
        airport_color = pygame.Color("black")
        red_dot_color = pygame.Color("red")

        airport_x_position, airport_y_position = self._bluesky_to_pygame(self.airport_details.position)
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
            x1, y1 = self._bluesky_to_pygame(point_1)
            x2, y2 = self._bluesky_to_pygame(point_2)
            pygame.draw.line(canvas, red_line_color, (x1, y1), (x2, y2), 2)

        ac_x_position, ac_y_position = self._bluesky_to_pygame(ac_position)

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
            x1, y1 = self._bluesky_to_pygame(Position(lat=point_1[0], lon=point_1[1]))
            x2, y2 = self._bluesky_to_pygame(Position(lat=point_2[0], lon=point_2[1]))
            pygame.draw.line(canvas, color, (x1, y1), (x2, y2), 2)

    def _initialize_pygame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.blue_background)
        return canvas

    def _draw_observation_text(self, canvas):
        """Draw observation values as text in the upper-left corner."""
        font = pygame.font.Font(None, 24)
        text_color = pygame.Color("black")

        obs = self._get_obs()
        y_offset = 10
        obs = {**obs, "airport_bearing": np.array([self.airport_details.hdg])}

        for key, value in obs.items():
            if "Heading" in key or "Azimuth" in key:
                text = f"{key}: {value[0]*180:.4f}"
            else:
                text = f"{key}: {value[0]:.4f}"
            text_surface = font.render(text, True, text_color)
            canvas.blit(text_surface, (10, y_offset))
            y_offset += 30

    def _handle_pygame_events(self) -> None:
        if self.render_mode != "human" or self.window is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def _set_terminal_condition(self):
        """Adapted from PathPlanningEnv by Groot et al."""
        num_points = 36

        airport_lat = self.airport_details.position.lat
        airport_lon = self.airport_details.position.lon
        airport_hdg = self.airport_details.hdg

        self.faf_lat, self.faf_lon = fn.get_point_at_distance(airport_lat, airport_lon, self.faf_distance,
                                                    deg_to_360(airport_hdg + 180))
        cw_bound = deg_to_360(airport_hdg + 180) + (self.iaf_angle / 2)
        ccw_bound = deg_to_360(airport_hdg + 180) - (self.iaf_angle / 2)

        angles = np.linspace(cw_bound, ccw_bound, num_points)
        iaf_lat, iaf_lon = fn.get_point_at_distance(self.faf_lat, self.faf_lon, self.iaf_distance, angles)
        self.iaf_center_lat, self.iaf_center_lon = iaf_lat[num_points // 2], iaf_lon[num_points // 2]

        command = f"POLYLINE SINK"
        for lat, lon in zip(iaf_lat, iaf_lon):
            command += f" {lat} {lon}"
        bs.stack.stack(command)

        bs.stack.stack(f"POLYLINE RESTRICT {iaf_lat[0]} {iaf_lon[0]} {self.faf_lat} {self.faf_lon} {iaf_lat[-1]} {iaf_lon[-1]}")
        bs.sim.step()

    def _get_terminal_condition(self) -> tuple[bool, float]:
        terminated = False
        reward = 0

        shapes = bs.tools.areafilter.basic_shapes
        current_pos, _ = self.get_aircraft_details()
        if self.aircraft_positions:
            last_pos = self.aircraft_positions[-1]
            line_ac = Path(np.array([[last_pos.lat, last_pos.lon], [current_pos.lat, current_pos.lon]]))
            line_sink = Path(np.reshape(shapes["SINK"].coordinates, (len(shapes["SINK"].coordinates) //2, 2)))
            line_restrict = Path(np.reshape(shapes["RESTRICT"].coordinates, (len(shapes["RESTRICT"].coordinates) //2, 2)))

            if line_sink.intersects_path(line_ac):
                print("Reached airport Successfully")
                reward = 50
                terminated = True

            elif line_restrict.intersects_path(line_ac):
                print("Failed to reach airport")
                reward = -1
                terminated = True

        return terminated, reward


