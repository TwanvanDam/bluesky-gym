import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import rasterio
import rasterio.features
import shapely
from gstools import CovModel, SumModel
from matplotlib import colors
import gstools as gs

@dataclass
class GeneratorBase(ABC):
    map_shape: tuple[int, int]
    map_range: tuple[float, float]

    @abstractmethod
    def regenerate(self, rng: np.random.Generator = None):
        ...

@dataclass
class PolygonGenerator(GeneratorBase):
    obstacle_size: int
    multiplier: float

    def regenerate(self, rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        num_obstacles = rng.integers(2,12)

        polygons = [self.generate_random_polygon(rng=rng) for _ in range(num_obstacles)]
        map = rasterio.features.rasterize(polygons, out_shape=self.map_shape)
        map *= self.multiplier
        return map, "people_per_pixel"

    def generate_random_polygon(self, rng: np.random.Generator):
        num_vertices = rng.integers(3,6)
        vertices = []
        centroid_x, centroid_y = rng.uniform() * self.map_shape[0], rng.uniform() * self.map_shape[1]
        for vertex in range(num_vertices):
            vertices.append((centroid_x + rng.uniform(-1,1) * self.obstacle_size / 2, centroid_y + rng.uniform(-1,1) * self.obstacle_size / 2))
        # Sort vertices by angle relative to centroid
        vertices.sort(key=lambda v: np.arctan2(v[1] - centroid_y, v[0] - centroid_x))
        return shapely.Polygon(vertices)

@dataclass
class CitiesGenerator(GeneratorBase):
    num_cities: int
    base_occupancy: float

    def regenerate(self, rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        # 1. Create the base grid and add "City Seeds"
        grid = np.zeros(self.map_shape)
        rows = rng.integers(0, self.map_shape[0], self.num_cities)
        cols = rng.integers(0, self.map_shape[1], self.num_cities)
        grid[rows, cols] = rng.exponential(scale=15.0, size=self.num_cities)

        # 2. Smooth the cities (The "Blur" effect)
        k_size = 61
        y, x = np.ogrid[-k_size // 2: k_size // 2 + 1, -k_size // 2: k_size // 2 + 1]
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * (k_size / 5) ** 2))

        # FFT Convolution
        grid_fft = np.fft.fft2(grid)
        kernel_fft = np.fft.fft2(kernel, s=self.map_shape)
        density_map = np.fft.ifft2(grid_fft * kernel_fft).real

        # 3. INCREASE BACKGROUND POPULATION
        # Instead of just light noise, we add a solid base + heavy Log-Normal noise
        # Log-normal creates that "sprawling rural" look where most areas have people
        rural_background = rng.lognormal(mean=-1.0, sigma=0.5, size=self.map_shape) * self.base_occupancy

        # Combine the two
        combined_map = density_map + rural_background

        # 4. Final Normalization
        combined_map = np.maximum(combined_map, 0)
        return combined_map, "people_per_pixel"

@dataclass
class PopulationDensityGenerator(GeneratorBase):
    covariance_models: dict[str, dict]
    target_mean: float

    def __post_init__(self):
        self.sum_model = None
        for cov in self.covariance_models.values():
            cov = cov.copy()
            cov_model = getattr(gs, cov.pop("cov_model"))
            model = cov_model(dim=2, **cov)
            self.sum_model = model if self.sum_model is None else self.sum_model + model

        self.ocean_model: SumModel = gs.Exponential(dim=2, len_scale=300, var=0.5) + gs.Gaussian(dim=2, len_scale=500, var=1)
        self.sum_srf = gs.SRF(self.sum_model)
        self.ocean_srf = gs.SRF(self.ocean_model)
        self.grid_x = np.linspace(0, self.map_range[0] / 1000, self.map_shape[0] + 1)
        self.grid_y = np.linspace(0, self.map_range[1] / 1000, self.map_shape[1] + 1)

    def sample_points_from_map(self, srf: gs.SRF, mean:float, rng: np.random.Generator = None) -> np.ndarray:
        return srf.structured((self.grid_x, self.grid_y), seed=rng.integers(0, 1e9) if rng else None)

    @staticmethod
    def winsorize_upper_tail(array: np.ndarray, percentile: float = 99.8) -> np.ndarray:
        """Clip the tail, so the influence of high values is minimized."""
        clip_above = np.nanpercentile(array, percentile)
        return np.clip(array, 0, clip_above)

    def scale_mean(self, array:np.ndarray) -> np.ndarray:
        mean = np.nanmean(array)
        return array.copy() * (self.target_mean / mean)

    def regenerate(self, rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()

        synthetic_log = self.sample_points_from_map(self.sum_srf, mean=1, rng=rng)
        synthetic = np.expm1(synthetic_log)
        synthetic  = self.winsorize_upper_tail(synthetic)
        synthetic = self.scale_mean(synthetic)

        ocean = self.sample_points_from_map(self.ocean_srf, mean=0, rng=rng)
        synthetic_masked = np.where(ocean < np.percentile(ocean, 10), -9999, synthetic)
        return synthetic_masked, "people_per_km2"

class ZeroPopulationGenerator(GeneratorBase):
    def regenerate(self, rng: np.random.Generator = None):
        return np.zeros(self.map_shape), "people_per_pixel"

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)


    # Fitted to 110 x 150 grid of each 4 km x 4 km cell, so length scales are in units of kms.
    covariance_models = {"cov_1" : {"cov_model" : "Gaussian", 'var':0.625, 'len_scale':60.3},
                         "cov_2" : {"cov_model" : "Gaussian", 'var':0.815, 'len_scale':2.63e2},
                         "cov_3" : {"cov_model" : "Integral", 'var':1.83, 'len_scale':37.0, 'nu': 0.233}}

    generator = PopulationDensityGenerator(covariance_models=covariance_models,
                                           map_shape=(128,128),
                                           map_range=(512,512),
                                           target_mean=361.60)


    def plot_distribution(density: np.ndarray):
        print(
            f"Min, Max, Mean, Median: {np.nanmin(density):.2f}, {np.nanmax(density):.2f}, {np.nanmean(density):.2f}, {np.nanmedian(density):.2f}")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(density[~np.isnan(density)].flatten(), bins=50, log=True)
        plt.title("Population Density Distribution")
        plt.xlabel("Population Density")
        plt.ylabel("Frequency (log scale)")

        plt.subplot(1, 2, 2)
        sorted_density = np.sort(density[~np.isnan(density)].flatten())
        cumulative = np.arange(len(sorted_density)) / len(sorted_density)
        plt.plot(sorted_density, cumulative)
        plt.xscale("log")
        plt.title("Cumulative Distribution of Population Density")
        plt.xlabel("Population Density (log scale)")
        plt.ylabel("Cumulative Probability")
        plt.tight_layout()
        plt.show()


    while True:
        start = time.time()
        pop_map,_ = generator.regenerate(rng)
        print(time.time() - start)
        pop_map = np.where(pop_map == -9999, np.nan, pop_map)
        plot_distribution(pop_map)
        im2 = plt.imshow(pop_map, cmap="Blues", origin="upper", norm=colors.Normalize(vmin=0, vmax=np.nanpercentile(pop_map, 99)))
        plt.title("Population Density (GRF)")
        plt.colorbar(im2)
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.show()
