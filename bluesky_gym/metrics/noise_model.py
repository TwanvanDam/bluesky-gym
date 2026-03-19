from __future__ import annotations
from functools import lru_cache
import numpy as np
from bluesky.tools.aero import ft as ft_to_m
from pydantic import model_validator, ConfigDict, BaseModel


class NoiseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    noise_base_dba: float = 85  # Base noise level
    noise_cutoff_dba: float = 55  # Minimum noise level
    noise_resolution_m: float = 1000  # Noise resolution in meters
    w_0: float = 1e-12  # Reference sound power in watts (0 dBA corresponds to 1e-12 W)
    reward_scaling_factor: float = 2.0  # Scaling factor for noise reward
    base_distance: float = 1000 * ft_to_m # Reference distance in meters for noise calculation

    @model_validator(mode="after")
    def _validate_noise_bounds(self) -> NoiseConfig:
        if self.noise_resolution_m <= 0:
            raise ValueError("Noise resolution must be positive")
        if self.noise_base_dba <= self.noise_cutoff_dba:
            raise ValueError("Base noise level must be greater than cutoff noise level")
        return self

    def build(self) -> NoiseModel:
        return NoiseModel(config=self)

class NoiseModel:
    def __init__(self, config: NoiseConfig):
        self.config = config

    @property
    def base_noise_power(self) -> float:
        return self.config.w_0 * 10 ** (self.config.noise_base_dba / 10) # [ W ]

    @property
    def cutoff_noise_power(self) -> float:
        return self.config.w_0 * 10 ** (self.config.noise_cutoff_dba / 10) # [ W ]

    @property
    def base_noise_power_1m(self) -> float:
        return self.base_noise_power * (self.config.base_distance ** 2 ) # [ W ] at 1 meter

    @lru_cache
    def get_noise_power_kernel_shape_meters_and_pixels(self, altitude: float) -> tuple[tuple[float, float], tuple[int, int]]:
        """Get the shape of the noise power kernel in meters and pixels based on the altitude and noise parameters."""
        noise_radius = np.sqrt(self.base_noise_power_1m / self.cutoff_noise_power)  # [ m ] radius at which noise power drops to cutoff level
        noise_radius_ground = np.sqrt(noise_radius **2 - altitude ** 2) if noise_radius > altitude else self.config.noise_resolution_m  # [ m ]

        noise_radius_rounded = np.ceil(noise_radius_ground / self.config.noise_resolution_m) * self.config.noise_resolution_m
        kernel_shape_meters = tuple(float(2 * noise_radius_rounded) for _ in range(2))  # [ m ]
        kernel_shape_pixels = tuple(int(shape_meters / self.config.noise_resolution_m) + 1 for shape_meters in kernel_shape_meters)  # [ px ]
        return kernel_shape_meters, kernel_shape_pixels

    def get_noise_power_kernel(self, altitude: float) -> np.ndarray:
        kernel_shape_meters, _ = self.get_noise_power_kernel_shape_meters_and_pixels(altitude)

        x = np.arange(-kernel_shape_meters[0] / 2, kernel_shape_meters[0] / 2 + 1, self.config.noise_resolution_m)
        y = np.arange(-kernel_shape_meters[1] / 2, kernel_shape_meters[1] / 2 + 1, self.config.noise_resolution_m)
        xx, yy = np.meshgrid(x, y)
        distance_squared = xx**2 + yy**2 + altitude ** 2  # [ m ]

        noise_power = self.base_noise_power_1m / distance_squared  # [ W ]
        noise_power = np.clip(noise_power, self.cutoff_noise_power, None)
        return noise_power

    def calculate_mean_step_noise(self, population_map: np.ndarray, altitude: float, sim_dt: float) -> float:
        population_map = np.clip(population_map, 0, None)  # Ensure no negative population values
        mean_population_density = float(np.nanmean(population_map))
        mean_step_noise_power = self.step_total_noise(mean_population_density, altitude, sim_dt)
        return mean_step_noise_power

    def step_total_noise(self, population_map_extract: np.ndarray | float, altitude: float, sim_dt: float) -> float:
        noise_power_kernel = self.get_noise_power_kernel(altitude)
        assert isinstance(population_map_extract, float) or population_map_extract.shape == noise_power_kernel.shape, f"Population map extract {population_map_extract.shape} and noise kernel {noise_power_kernel.shape} must have the same shape"
        total_noise_power = np.sum(population_map_extract * noise_power_kernel)
        return total_noise_power * sim_dt

    def step_normalized_noise(self, population_map_extract: np.ndarray, altitude: float, mean_step_noise: float, sim_dt: float) -> float:
        step_total_noise = self.step_total_noise(population_map_extract, altitude, sim_dt)

        # Multiply by 2 to have the right magnitude compared to fuel consumption.
        normalized_noise = self.config.reward_scaling_factor * step_total_noise / mean_step_noise if mean_step_noise > 0 else 0
        return normalized_noise






