import itertools
from enum import Enum
from typing import Callable, Optional, Literal

import bluesky as bs
import gymnasium as gym
import numpy as np
import pygame
import pyproj
from bluesky.tools.position import Position
from gymnasium import spaces
from matplotlib.path import Path
from pydantic import BaseModel, Field, ConfigDict

import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
from bluesky_gym.metrics.fuel_model import FuelModel
from bluesky_gym.utils.sampling_config import SamplingConfig, UniformSamplingConfig, NormalSamplingConfig


class InitialConditionsSamplingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    simulation_bounds_mode: Literal["degrees", "meters"] = "degrees"
    simulation_bounds_size: float = 5.0

    destination_lat_sampling: SamplingConfig
    destination_lon_sampling: SamplingConfig
    destination_hdg_sampling: SamplingConfig

    aircraft_position_mode: Literal["absolute", "relative"] = "relative"
    aircraft_lat_sampling: SamplingConfig
    aircraft_lon_sampling: SamplingConfig


class NavigationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    ac_name: str = "KL001"
    ac_type: str = "a320"
    ac_initial_spd: float = 200  # [ m / s ]
    ac_initial_alt: float = 3_000  # [ m ]

    map_sampling_config: InitialConditionsSamplingConfig

    max_steps: int = 250
    sim_dt: int = 3  # [ s ]
    action_time: int = 60  # [ s ]
    faf_distance: float = 25  # [ km ]
    iaf_angle: float = 60  # [ degrees ]
    iaf_distance: float = 30  # [ km ]

    map_projection_crs: str = "EPSG:3035"
    use_sin_cos_obs: bool = False
    normalize_distance_obs: Optional[float] = None
    constraint_violation_reward: float = -1.0
    successful_approach_reward: float = 50.0
    mean_episode_length: float = 20 * 60  # [ s ]


class TerminationReason(Enum):
    SUCCESS = "success"
    OUT_OF_BOUNDS = "out_of_bounds"
    FAILED_APPROACH = "failed_approach"
    MAX_STEPS = "max_steps"
    NONE = "none"


def bs_position(lat: float, lon: float, hdg: float | None = None) -> Position:
    pos = Position(name=f"{lat},{lon}", reflat=0, reflon=0)
    if hdg is not None:
        pos.refhdg = hdg
    return pos


class BaseNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: NavigationConfig, render_mode: str | None = None,
                 window_size: tuple[int, int] = (512, 512),
                 save_trajectory: bool = False,
                 ) -> None:
        self.total_episode_fuel_reward = None
        self.episode_length_seconds = None
        self.mean_fuel_flow = None
        self.total_episode_fuel_used = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._render_owned_by_wrapper = False

        self.config = config
        self.ac_name = config.ac_name
        self.save_trajectory = save_trajectory
        self._telemetry_history = []

        self.bluesky_crs = "WGS84"
        self.map_projection_crs = config.map_projection_crs
        self.coordinate_transformer = pyproj.Transformer.from_crs(
            self.bluesky_crs,
            self.map_projection_crs,
            always_xy=True
        )



        self.observation_space = spaces.Dict(
            {
                # Ground distance to the destination in meters [0, inf]
                "destination_ground_distance": spaces.Box(0, np.inf, shape=(1,), dtype=np.float64),

                # The required heading change to reach the destination normalized to [0 ,1], which corresponds to [-180, 180]
                "destination_relative_heading": spaces.Box(-180, 180, shape=(1,), dtype=np.float64),

                # The orientation of the destination relative to the aircraft's heading, normalized to [0 ,1], which corresponds to [-180, 180]
                "destination_relative_orientation": spaces.Box(-180, 180, shape=(1,), dtype=np.float64),
            }
        )

        self.action_space = spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float64)

        self.fuel_weight = 1

        self._reward_components: list[Callable] = [
            self._fuel_reward,
            self._termination_reward,
            self._boundary_reward,
            self._truncation_reward,
        ]

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

        self.fuel_flow_model = FuelModel(self.config.ac_type)
        self.fuel_used_during_step: float | None = None
        self.destination: Position | None = None
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

    def set_simulation_bounds_meters(self) -> None:
        corners_xy = [
            self.coordinate_transformer.transform(self.lon_min, self.lat_min),
            self.coordinate_transformer.transform(self.lon_min, self.lat_max),
            self.coordinate_transformer.transform(self.lon_max, self.lat_min),
            self.coordinate_transformer.transform(self.lon_max, self.lat_max),
        ]
        xs, ys = zip(*corners_xy)
        self.x_min = min(xs)
        self.y_min = min(ys)
        self.x_max = max(xs)
        self.y_max = max(ys)

    def _update_simulation_bounds(self, destination: Position) -> None:
        map_size = self.config.map_sampling_config.simulation_bounds_size
        match self.config.map_sampling_config.simulation_bounds_mode:
            case "degrees":
                self.lon_min, self.lon_max = destination.lon - map_size, destination.lon + map_size
                self.lat_min, self.lat_max = destination.lat - map_size, destination.lat + map_size
            case "meters":
                self.lon_min = fn.get_point_at_distance(destination.lat, destination.lon, map_size, 180)[0]
                self.lon_max = fn.get_point_at_distance(destination.lat, destination.lon, map_size, 0)[0]
                self.lat_min = fn.get_point_at_distance(destination.lat, destination.lon, map_size, 270)[1]
                self.lat_max = fn.get_point_at_distance(destination.lat, destination.lon, map_size, 90)[1]

    def reset(self, seed=None, options: None | dict[str, float] = None):
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Optional dict to force specific positions and headings:
                - destination_lat, destination_lon, destination_hdg: Force destination position and heading.
                - aircraft_lat, aircraft_lon: Force aircraft position.
                - aircraft_hdg: Force aircraft heading (defaults to pointing towards airport).
        """
        bs.traf.reset()
        super().reset(seed=seed)

        options = options or {}

        self.current_step = 0
        self.episode_length_seconds = 0
        self.total_episode_fuel_reward = 0.0
        self.total_episode_fuel_used = 0.0
        self._telemetry_history = []

        self.destination = self._generate_airport(self.np_random, options)
        self._update_simulation_bounds(self.destination)
        self.set_simulation_bounds_meters()
        self._set_terminal_condition()

        for _ in range(100):
            aircraft_initial_position = self._generate_initial_position(self.np_random, options)
            if self.check_inside_bounds(aircraft_initial_position):
                break
        else:
            raise RuntimeError("Could not sample an aircraft position within bounds after 100 attempts. Check SamplingConfig.")
        self.aircraft_positions = [aircraft_initial_position]

        aircraft_initial_hdg = options.get("aircraft_hdg", None)
        if not aircraft_initial_hdg:
            aircraft_initial_hdg = fn.get_hdg((aircraft_initial_position.lat, aircraft_initial_position.lon),
                                              (self.destination.lat, self.destination.lon))
        bs.traf.cre(self.ac_name, actype=self.config.ac_type, aclat=aircraft_initial_position.lat,
                    aclon=aircraft_initial_position.lon,
                    achdg=aircraft_initial_hdg, acspd=self.config.ac_initial_spd, acalt=self.config.ac_initial_alt)
        bs.sim.step()
        self.mean_fuel_flow = self._get_fuel_flow()

        if self.render_mode == "human" and not self._render_owned_by_wrapper:
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        ac_hdg = self.get_aircraft_heading()
        new_heading = fn.bound_angle_0_360(ac_hdg + action[0])
        bs.stack.stack(f"HDG {self.ac_name} {new_heading}")
        reward = 0
        done = False

        for i in range(self.action_frequency):
            bs.sim.step()
            ac_pos = self.get_aircraft_position()
            self.episode_length_seconds += self.sim_dt
            self.aircraft_positions.append(ac_pos)
            if self.save_trajectory:
                self._save_telemetry()

            intermediate_reward, terminated, truncated, reason = self._get_reward()
            reward += intermediate_reward
            done = terminated or truncated
            if done:
                break

        self.current_step += 1
        observation = self._get_obs()
        info = {}

        if done:
            info["termination_reason"] = reason.value
            info["episode_length_seconds"] = self.episode_length_seconds
            info["total_episode_fuel_reward"] = self.total_episode_fuel_reward
            info["total_episode_fuel_used"] = self.total_episode_fuel_used

        elif self.render_mode == "human" and not self._render_owned_by_wrapper:
            self.render()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _get_obs(self):
        ac_pos = self.get_aircraft_position()
        ac_hdg = self.get_aircraft_heading()

        correct_heading = (fn.get_hdg((ac_pos.lat, ac_pos.lon),
                                      (self.destination.lat, self.destination.lon)))

        destination_relative_heading = np.array([fn.bound_angle_positive_negative_180(correct_heading - ac_hdg)])
        destination_relative_orientation = np.array(
            [fn.bound_angle_positive_negative_180(self.destination.refhdg - ac_hdg)])

        destination_x, destination_y = self.coordinate_transformer.transform(self.destination.lon,
                                                                             self.destination.lat)
        aircraft_x, aircraft_y = self.coordinate_transformer.transform(ac_pos.lon, ac_pos.lat)

        destination_ground_distance = np.array(
            [np.sqrt((destination_x - aircraft_x) ** 2 + (destination_y - aircraft_y) ** 2)],
            dtype=np.float64)

        observation = {
            # Ground distance to the destination in meters [0, inf]
            "destination_ground_distance": destination_ground_distance,

            # The required heading change to reach the destination in degrees [-180, 180]
            "destination_relative_heading": destination_relative_heading,

            # The orientation of the destination relative to the aircraft's heading in degrees [-180, 180]
            "destination_relative_orientation": destination_relative_orientation,
        }
        return observation

    def get_aircraft_position(self) -> Position:
        ac_idx = bs.traf.id2idx(self.ac_name)
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]
        return bs_position(lat=ac_lat, lon=ac_lon)

    def get_aircraft_heading(self) -> float:
        ac_idx = bs.traf.id2idx(self.ac_name)
        ac_hdg = bs.traf.hdg[ac_idx]
        return ac_hdg

    def get_aircraft_altitude(self) -> float:
        ac_idx = bs.traf.id2idx(self.ac_name)
        return bs.traf.alt[ac_idx]

    def get_aircraft_mass(self) -> float:
        ac_idx = bs.traf.id2idx(self.ac_name)
        return bs.traf.perf.mass[ac_idx]

    def get_aircraft_tas(self) -> float:
        ac_idx = bs.traf.id2idx(self.ac_name)
        return bs.traf.tas[ac_idx]

    def _get_reward(self) -> tuple[float, bool, bool, TerminationReason]:
        total_reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps
        if truncated:
            reason = TerminationReason.MAX_STEPS
        else:
            reason = TerminationReason.NONE

        for component in self._reward_components:
            component_reward, component_terminated, component_reason = component()
            if component_terminated and reason == TerminationReason.NONE:
                reason = component_reason
            total_reward += component_reward
            terminated = terminated or component_terminated

        return total_reward, terminated, truncated, reason

    def add_reward_component(self, function: Callable) -> None:
        self._reward_components.append(function)

    def _get_fuel_flow(self) -> float:
        ac_tas = self.get_aircraft_tas()  # m/s
        ac_alt = self.get_aircraft_altitude()  # m
        ac_mass = self.get_aircraft_mass()  # kg
        fuel_flow = self.fuel_flow_model.step_fuel_flow(mass=ac_mass, tas=ac_tas, altitude=ac_alt)
        return fuel_flow

    @property
    def dense_reward_scaling(self) -> float:
        return 1 / self.config.mean_episode_length

    def _fuel_reward(self) -> tuple[float, bool, TerminationReason]:
        fuel_flow = self._get_fuel_flow()
        step_fuel_used = fuel_flow * self.sim_dt
        normalized_fuel_usage = step_fuel_used / self.mean_fuel_flow

        self.total_episode_fuel_used += step_fuel_used
        step_fuel_reward = - self.fuel_weight * (normalized_fuel_usage * self.dense_reward_scaling)
        self.total_episode_fuel_reward += step_fuel_reward
        return step_fuel_reward, False, TerminationReason.NONE

    def _boundary_reward(self) -> tuple[float, bool, TerminationReason]:
        if self._check_out_of_bounds():
            return self.config.constraint_violation_reward, True, TerminationReason.OUT_OF_BOUNDS
        else:
            return 0.0, False, TerminationReason.NONE

    def _truncation_reward(self) -> tuple[float, bool, TerminationReason]:
        if self.current_step >= self.max_steps:
            return self.config.constraint_violation_reward, False, TerminationReason.MAX_STEPS
        return 0.0, False, TerminationReason.NONE

    def _termination_reward(self) -> tuple[float, bool, TerminationReason]:
        return self._get_terminal_condition()

    def _set_terminal_condition(self):
        """Adapted from PathPlanningEnv by Groot et al."""
        num_points = 36

        destination_lat = self.destination.lat
        destination_lon = self.destination.lon
        destination_hdg = self.destination.refhdg

        self.faf_lat, self.faf_lon = fn.get_point_at_distance(destination_lat, destination_lon, self.faf_distance,
                                                              fn.bound_angle_0_360(destination_hdg + 180))
        cw_bound = fn.bound_angle_0_360(destination_hdg + 180) + (self.iaf_angle / 2)
        ccw_bound = fn.bound_angle_0_360(destination_hdg + 180) - (self.iaf_angle / 2)

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
        current_pos = self.get_aircraft_position()
        if self.aircraft_positions:
            last_pos = self.aircraft_positions[-1]
            line_ac = Path(np.array([[last_pos.lat, last_pos.lon], [current_pos.lat, current_pos.lon]]))
            line_sink = Path(np.reshape(shapes["SINK"].coordinates, (len(shapes["SINK"].coordinates) // 2, 2)))
            line_restrict = Path(
                np.reshape(shapes["RESTRICT"].coordinates, (len(shapes["RESTRICT"].coordinates) // 2, 2)))

            if line_sink.intersects_path(line_ac):
                reward = self.config.successful_approach_reward
                reason = TerminationReason.SUCCESS
                terminated = True

            elif line_restrict.intersects_path(line_ac):
                reward = self.config.constraint_violation_reward
                reason = TerminationReason.FAILED_APPROACH
                terminated = True

        return reward, terminated, reason

    def _check_out_of_bounds(self) -> bool:
        return not self.check_inside_bounds(self.get_aircraft_position(), margin=0.0)

    def _generate_airport(self, np_random: np.random.Generator, options: dict) -> Position:
        map_sampling_config = self.config.map_sampling_config

        destination_lat = options.get("destination_lat", map_sampling_config.destination_lat_sampling.sample(np_random))
        destination_lon = options.get("destination_lon", map_sampling_config.destination_lon_sampling.sample(np_random))
        destination_hdg = options.get("destination_hdg", map_sampling_config.destination_hdg_sampling.sample(np_random))

        return bs_position(lat=destination_lat,
                           lon=destination_lon,
                           hdg=destination_hdg)

    def _generate_initial_position(self, np_random: np.random.Generator, options: dict) -> Position:
        map_sampling_config = self.config.map_sampling_config

        if "aircraft_lat" in options and "aircraft_lon" in options:
            return bs_position(lat=options["aircraft_lat"], lon=options["aircraft_lon"])
        elif "aircraft_lat" in options or "aircraft_lon" in options:
            raise ValueError("Both aircraft_lat and aircraft_lon must be provided in options if one is provided.")

        match map_sampling_config.aircraft_position_mode:
            case "relative":
                aircraft_lat = self.destination.lat + map_sampling_config.aircraft_lat_sampling.sample(np_random)
                aircraft_lon = self.destination.lon + map_sampling_config.aircraft_lon_sampling.sample(np_random)
            case "absolute":
                aircraft_lat = map_sampling_config.aircraft_lat_sampling.sample(np_random)
                aircraft_lon = map_sampling_config.aircraft_lon_sampling.sample(np_random)
            case _:
                raise ValueError("Unrecognized aircraft position mode")

        return bs_position(
            lat=aircraft_lat,
            lon=aircraft_lon
        )

    def check_inside_bounds(self, pos: Position, margin: float = 0.1) -> bool:
        delta_lat = self.lat_max - self.lat_min
        delta_lon = self.lon_max - self.lon_min
        lat_in_bounds = self.lat_min + (delta_lat * margin) <= pos.lat <= self.lat_max - (delta_lat * margin)
        lon_in_bounds = self.lon_min + (delta_lon * margin) <= pos.lon <= self.lon_max - (delta_lon * margin)
        return lat_in_bounds and lon_in_bounds

    def _save_telemetry(self) -> None:
        ac_alt = self.get_aircraft_altitude()
        ac_pos = self.get_aircraft_position()
        ac_tas = self.get_aircraft_tas()
        ac_mass = self.get_aircraft_mass()
        sim_dt = self.sim_dt
        self._telemetry_history.append(
            {"lat": ac_pos.lat, "lon": ac_pos.lon, "tas": ac_tas, "mass": ac_mass, "alt": ac_alt, "sim_dt": sim_dt})

    def get_telemetry_history(self) -> list[dict[str, float]]:
        return self._telemetry_history

    def lat_lon_to_pix(self, position: Position) -> tuple[int, int]:
        x_meters, y_meters = self.coordinate_transformer.transform(position.lon, position.lat)

        return self.meters_to_pix((x_meters, y_meters))

    def meters_to_pix(self, position_meters: tuple[float, float]) -> tuple[int, int]:
        norm_x = (position_meters[0] - self.x_min) / (self.x_max - self.x_min)
        norm_y = (position_meters[1] - self.y_min) / (self.y_max - self.y_min)
        screen_x = int(norm_x * self.window_size[0])
        screen_y = int((1 - norm_y) * self.window_size[1])
        return screen_x, screen_y

    def render(self):
        if self.render_mode is None:
            return None

        self.initialize_pygame(self.window_size)
        self.handle_pygame_events()
        canvas = pygame.Surface(self.window_size)

        for draw_function in self.get_render_layers():
            draw_function(canvas)

        return self.present_canvas(canvas)

    def get_render_layers(self) -> list[Callable]:
        """Return a list of functions that can be run to render the environment."""
        return [lambda canvas: canvas.fill(self.blue_background),
                self.draw_airport,
                self.draw_aircraft,
                self.draw_observation_text]

    def initialize_pygame(self, window_size: tuple[int, int]):
        """Checks if pygame is initialized properly. If not it will initialize."""
        if not pygame.get_init():
            pygame.init()
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(window_size)
            self.clock = pygame.time.Clock()
        return

    def present_canvas(self, canvas: pygame.Surface) -> None | np.ndarray:
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
        return None

    def handle_pygame_events(self) -> None:
        if self.window is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def draw_airport(self, canvas):
        airport_color = pygame.Color("black")
        red_dot_color = pygame.Color("red")

        airport_x_position, airport_y_position = self.lat_lon_to_pix(self.destination)
        shapes = bs.tools.areafilter.basic_shapes
        line_sink = np.reshape(shapes["SINK"].coordinates, (len(shapes["SINK"].coordinates) // 2, 2))
        line_restrict = np.reshape(shapes["RESTRICT"].coordinates, (len(shapes["RESTRICT"].coordinates) // 2, 2))

        pygame.draw.circle(canvas, red_dot_color, (int(airport_x_position), int(airport_y_position)), 5)

        self._draw_line_from_points(canvas, airport_color, list(line_sink))

        self._draw_line_from_points(canvas, airport_color, list(line_restrict))

    def draw_aircraft(self, canvas):
        aircraft_color = pygame.Color("black")
        ac_position = self.get_aircraft_position()
        ac_heading = self.get_aircraft_heading()

        red_line_color = pygame.Color("red")
        for point_1, point_2 in itertools.pairwise(self.aircraft_positions):
            x1, y1 = self.lat_lon_to_pix(point_1)
            x2, y2 = self.lat_lon_to_pix(point_2)
            pygame.draw.line(canvas, red_line_color, (x1, y1), (x2, y2), 2)

        ac_x_position, ac_y_position = self.lat_lon_to_pix(ac_position)

        heading_end_lat, heading_end_lon = fn.get_point_at_distance(ac_position.lat, ac_position.lon,
                                                                    self.aircraft_heading_length, ac_heading)
        heading_end_x, heading_end_y = self.lat_lon_to_pix(bs_position(lat=heading_end_lat, lon=heading_end_lon))

        pygame.draw.circle(canvas, aircraft_color, (int(ac_x_position), int(ac_y_position)), 5)

        pygame.draw.line(canvas,
                         aircraft_color,
                         (ac_x_position, ac_y_position),
                         (heading_end_x, heading_end_y),
                         width=2
                         )

    def _draw_line_from_points(self, canvas: pygame.Surface, color: pygame.Color, points: list[Position]) -> None:
        for point_1, point_2 in itertools.pairwise(points):
            x1, y1 = self.lat_lon_to_pix(bs_position(lat=point_1[0], lon=point_1[1]))
            x2, y2 = self.lat_lon_to_pix(bs_position(lat=point_2[0], lon=point_2[1]))
            pygame.draw.line(canvas, color, (x1, y1), (x2, y2), 2)

    def draw_observation_text(self, canvas):
        """Draw observation values as text in the upper-left corner."""
        font = pygame.font.Font(None, 24)
        text_color = pygame.Color("black")

        obs = self._get_obs()
        y_offset = 10
        obs = {**obs, "destination_bearing": np.array([self.destination.hdg])}

        for key, value in obs.items():
            if "distance" in key:
                text = f"{key}: {value[0] / 1000:.1f} [km]"
            else:
                text = f"{key}: {value[0]:.0f} [deg]"
            text_surface = font.render(text, True, text_color)
            canvas.blit(text_surface, (10, y_offset))
            y_offset += 30
