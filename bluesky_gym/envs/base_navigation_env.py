from dataclasses import dataclass

import bluesky as bs
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

ac_name = "KL001"
ac_type = "a320"
windows_size = (512, 512)
ac_initial_spd = 250


@dataclass
class Position:
    lat: float
    lon: float


@dataclass
class Airport:
    position: Position
    hdg: float


class BaseNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode):
        # pygame variables
        self.window_size = windows_size
        self.window: pygame.Surface | None = None
        self.clock = None
        self.xy_to_px: tuple | None = None

        self.observation_space = spaces.Dict(
            {
                "Delta_lat": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "Delta_lon": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "Delta_heading": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
            }
        )

        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')

        # variables for logging
        self.max_steps = 1000
        self.current_step: int | None = None

        self.fuel_used: float | None = None
        self.airport_details: Airport | None = None
        self.aircraft_initial_position: Position | None

    def reset(self, seed=None, options=None):
        bs.traf.reset()
        super().reset(seed=seed)

        self.fuel_used = 0.0
        self.aircraft_initial_position = Position(lat=0.0, lon=0.0)

        self.airport_details = Airport(
            Position(lat=np.random.normal(loc=0, scale=1),
                     lon=np.random.normal(loc=0, scale=1)),
            hdg=float(np.random.randint(low=1, high=36) * 10)
        )

        heading_to_airport = fn.get_hdg((self.aircraft_initial_position.lat, self.aircraft_initial_position.lon),
                                        (self.airport_details.position.lat, self.airport_details.position.lon))

        bs.traf.cre(ac_name, actype=ac_type, aclat=self.aircraft_initial_position.lat,
                    aclon=self.aircraft_initial_position.lon,
                    achdg=heading_to_airport, acspd=ac_initial_spd)

        self.xy_to_px = self._get_xy_to_px()

        if self.render_mode == "human":
            self._render_frame()
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        _, ac_hdg = self._get_aircraft_details()
        new_heading = (ac_hdg + action[0] * 180 + 360) % 360
        bs.stack.stack(f"HDG {ac_name} {new_heading}")

        for _ in range(10):
            bs.sim.step()
        self.current_step += 1

        observation = self._get_obs()
        reward, terminated = self._get_reward()
        truncated = self.current_step >= self.max_steps
        info = {}

        # bluesky reset?? bs.sim.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, truncated, info

    def _check_out_of_bounds(self) -> bool:
        ac_idx = bs.traf.id2idx(ac_name)
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]

        max_degrees_from_start = 5.0

        return np.sqrt((ac_lon - self.aircraft_initial_position.lon) ** 2 + (
                    ac_lat - self.aircraft_initial_position.lat) ** 2) >= max_degrees_from_start

    def _check_termination_at_airport(self) -> bool:
        ac_idx = bs.traf.id2idx(ac_name)
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]

        ac_hdg = bs.traf.hdg[ac_idx]

        reached_airport = abs(self.airport_details.position.lat - ac_lat) < 0.001 and abs(
            self.airport_details.position.lon - ac_lon) < 0.001
        correct_heading = abs(self.airport_details.hdg - ac_hdg) < 5
        return reached_airport and correct_heading

    def close(self) -> None:
        if self.render_mode == "human":
            pygame.quit()

    def _get_reward(self, fuel_coeff: float = 0.005):
        ac_idx = bs.traf.id2idx(ac_name)
        self.fuel_used += bs.traf.perf.fuelflow[ac_idx]

        penalty = 0
        if self._check_termination_at_airport():
            penalty = 5
            terminated = True
        elif self._check_out_of_bounds():
            penalty = -1
            terminated = True
        else:
            terminated = False

        return penalty - fuel_coeff * self.fuel_used, terminated

    def _get_aircraft_details(self) -> tuple[Position, float]:
        ac_idx = bs.traf.id2idx(ac_name)

        ac_hdg = bs.traf.hdg[ac_idx]
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]
        return Position(lat=ac_lat, lon=ac_lon), ac_hdg

    def _get_obs(self):
        ac_position, ac_hdg = self._get_aircraft_details()
        observation = {
            "Delta_lat": (self.airport_details.position.lat - ac_position.lat,),
            "Delta_lon": (self.airport_details.position.lon - ac_position.lon,),
            "Delta_heading": (self.airport_details.hdg - ac_hdg,),
        }

        return observation

    def _get_xy_to_px(self) -> tuple[float, float]:
        ac_position, _ = self._get_aircraft_details()

        delta_lat = abs(self.airport_details.position.lat - ac_position.lat)
        delta_lon = abs(self.airport_details.position.lon - ac_position.lon)

        return tuple(size / (3 * max(delta_lon, delta_lat)) for size in self.window_size)

    def _render_frame(self):
        canvas = self._initialize_pygame()

        self._draw_aircraft(canvas)

        self._draw_airport(canvas)

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_airport(self, canvas):
        airport_color = (0, 0, 0)
        airport_width = 10
        airport_length = 30

        faf_radius = 30
        faf_degrees = 60

        airport_x_position = (self.airport_details.position.lon - self.aircraft_initial_position.lon) * self.xy_to_px[
            0] + self.window_size[0] / 2
        airport_y_position = self.window_size[1] / 2 - (
                    self.airport_details.position.lat - self.aircraft_initial_position.lat) * self.xy_to_px[1]
        airport_heading = self.airport_details.hdg

        faf_x_position = airport_x_position - np.sin(np.deg2rad(airport_heading)) * airport_length / 2
        faf_y_position = airport_y_position + np.cos(np.deg2rad(airport_heading)) * airport_length / 2

        airport_surface = pygame.Surface((airport_width, airport_length), pygame.SRCALPHA)
        airport_surface.fill(airport_color)
        rotated_airport_surface = pygame.transform.rotate(airport_surface, -self.airport_details.hdg)
        airport_rect = rotated_airport_surface.get_rect(center=(airport_x_position, airport_y_position))
        canvas.blit(rotated_airport_surface, airport_rect)

        faf_arc_end = self.airport_details.hdg + (faf_degrees / 2)
        faf_arc_start = self.airport_details.hdg - (faf_degrees / 2)

        faf_arc_end_x = faf_x_position - np.sin(np.deg2rad(faf_arc_end)) * faf_radius
        faf_arc_end_y = faf_y_position + np.cos(np.deg2rad(faf_arc_end)) * faf_radius
        faf_arc_start_x = faf_x_position - np.sin(np.deg2rad(faf_arc_start)) * faf_radius
        faf_arc_start_y = faf_y_position + np.cos(np.deg2rad(faf_arc_start)) * faf_radius

        pygame.draw.line(canvas, airport_color, (faf_x_position, faf_y_position), (faf_arc_start_x, faf_arc_start_y), 2)
        pygame.draw.line(canvas, airport_color, (faf_x_position, faf_y_position), (faf_arc_end_x, faf_arc_end_y), 2)

    def _draw_aircraft(self, canvas):
        ac_length = 20
        ac_width = 10
        ac_color = (255, 255, 255)
        ac_heading_length = 50

        ac_position, ac_heading = self._get_aircraft_details()

        ac_x_position = (ac_position.lon - self.aircraft_initial_position.lon) * self.xy_to_px[0] + self.window_size[
            0] / 2
        ac_y_position = self.window_size[1] / 2 - (ac_position.lat - self.aircraft_initial_position.lat) * \
                        self.xy_to_px[1]

        heading_end_x = ac_x_position + np.sin(np.deg2rad(ac_heading)) * ac_heading_length
        heading_end_y = ac_y_position - np.cos(np.deg2rad(ac_heading)) * ac_heading_length

        ac_surface = pygame.Surface((ac_width, ac_length), pygame.SRCALPHA)
        ac_surface.fill(ac_color)
        rotated_ac_surface = pygame.transform.rotate(ac_surface, -ac_heading)
        ac_rect = rotated_ac_surface.get_rect(center=(ac_x_position, ac_y_position))
        canvas.blit(rotated_ac_surface, ac_rect)

        pygame.draw.line(canvas,
                         ac_color,
                         (ac_x_position, ac_y_position),
                         (heading_end_x, heading_end_y),
                         width=2
                         )

    def _initialize_pygame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))
        return canvas


def debug_env():
    # Instantiate the environment with rendering enabled
    env = BaseNavigationEnv(render_mode="human")

    try:
        # Reset the environment
        while True:
            obs, info = env.reset(seed=42)
            print(f"Initial Observation: {obs}")

            terminated = False
            truncated = False
            step_count = 0
            max_steps = 50  # Limit steps for debugging

            while not (terminated or truncated) and step_count < max_steps:
                # Sample a random action
                action = env.action_space.sample()

                # Take a step
                obs, reward, terminated, truncated, info = env.step(action)

                print(f"Step: {step_count + 1}")
                print(f"  Action: {action}")
                print(f"  Reward: {reward}")
                print(f"  Terminated: {terminated}")
                print(f"  Truncated: {truncated}")
                print("-" * 20)

                step_count += 1

    except Exception as e:
        print(f"An error occurred during execution: {e}")
    finally:
        # Cleanup
        env.close()


if __name__ == "__main__":
    debug_env()
