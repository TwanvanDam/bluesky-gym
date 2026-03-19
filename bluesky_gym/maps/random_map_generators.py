import time

import numpy as np
import rasterio
import rasterio.features
import shapely
from gstools import CovModel
from matplotlib import colors
import gstools as gs

"""
TODO map generators should also return a resolution.
Refactor random generators and MapSource class to handle this.
"""

def generate_random_polygon(shape: tuple[int,int], obstacle_size:int, rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    num_vertices = rng.integers(3,6)
    vertices = []
    centroid_x, centroid_y = rng.uniform() * shape[0], rng.uniform() * shape[1]
    for vertex in range(num_vertices):
        vertices.append((centroid_x + rng.uniform(-1,1) * obstacle_size / 2, centroid_y + rng.uniform(-1,1) * obstacle_size / 2))
       # Sort vertices by angle relative to centroid
    vertices.sort(key=lambda v: np.arctan2(v[1] - centroid_y, v[0] - centroid_x))
    return shapely.Polygon(vertices)

def generate_random_shapes_map(shape:tuple[int,int]=(512, 512), obstacle_size:int=100, multiplier:float=50_000, rng: np.random.Generator = None) -> tuple[np.ndarray, str]:
    rng = rng or np.random.default_rng()
    num_obstacles = rng.integers(2,12)

    polygons = [generate_random_polygon(shape, obstacle_size, rng=rng) for _ in range(num_obstacles)]
    map = rasterio.features.rasterize(polygons, out_shape=shape)
    map *= multiplier
    return map, "people_per_pixel"

def generate_cities(shape=(512, 512), num_cities=100, base_occupancy=0.5, rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    # 1. Create the base grid and add "City Seeds"
    grid = np.zeros(shape)
    rows = rng.integers(0, shape[0], num_cities)
    cols = rng.integers(0, shape[1], num_cities)
    grid[rows, cols] = rng.exponential(scale=15.0, size=num_cities)

    # 2. Smooth the cities (The "Blur" effect)
    k_size = 61
    y, x = np.ogrid[-k_size // 2: k_size // 2 + 1, -k_size // 2: k_size // 2 + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * (k_size / 5) ** 2))

    # FFT Convolution
    grid_fft = np.fft.fft2(grid)
    kernel_fft = np.fft.fft2(kernel, s=shape)
    density_map = np.fft.ifft2(grid_fft * kernel_fft).real

    # 3. INCREASE BACKGROUND POPULATION
    # Instead of just light noise, we add a solid base + heavy Log-Normal noise
    # Log-normal creates that "sprawling rural" look where most areas have people
    rural_background = rng.lognormal(mean=-1.0, sigma=0.5, size=shape) * base_occupancy

    # Combine the two
    combined_map = density_map + rural_background

    # 4. Final Normalization
    combined_map = np.maximum(combined_map, 0)
    return combined_map, "people_per_pixel"

def sample_points_from_map(model: gs.CovModel, mean, output_shape: tuple[int,int], rng: np.random.Generator = None):
    srf = gs.SRF(model)
    map = srf.structured((np.arange(output_shape[0]), np.arange(output_shape[1])), seed=rng.integers(0, 1e9) if rng else None)
    return map


def generate_population_density(covariance_models: dict[str, dict], shape: tuple[int,int]=(128,128), rng: np.random.Generator = None) -> tuple[np.ndarray, str]:

    rng = rng or np.random.default_rng()
    mean = 1
    # Adjust length scales based on output shape to maintain similar spatial patterns across different resolutions
    len_factor = 512 / shape[0]
    rescale = 50_000

    sum_model = None
    for cov in covariance_models.values():
        cov_model = getattr(gs, cov.pop("cov_model"))
        model = cov_model(dim=2, **cov)
        sum_model = model if sum_model is None else sum_model + model

    synthetic = np.expm1(sample_points_from_map(sum_model, mean, output_shape=shape, rng=rng))

    ocean_model = gs.Exponential(dim=2, len_scale=300 / len_factor, var=0.5) + gs.Gaussian(dim=2, len_scale=500 / len_factor, var=1)
    ocean = sample_points_from_map(ocean_model, mean=0, output_shape=shape, rng=rng)
    synthetic_clipped = np.clip(synthetic, 0, np.percentile(synthetic, 99.9))
    synthetic_rescaled = synthetic_clipped / np.nanmax(synthetic_clipped) * rescale
    synthetic_masked = np.where(ocean < np.percentile(ocean, 10), -9999, synthetic_rescaled)
    return synthetic_masked, "people_per_km2"



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)
    covariance_models = {"cov_1" : {"cov_model" : "Matern", 'var':0.867, 'len_scale':1.23e2, 'nu':21},
                         "cov_2" : {"cov_model" : "Stable", 'var':4.97, 'len_scale':0.988, 'alpha':0.669},
                         "cov_3" : {"cov_model" : "Spherical", 'var':1.3, 'len_scale':1.12e2}}
    def plot_histogram(density: np.ndarray):
        plt.hist(density[~np.isnan(density)].flatten(), bins=50, log=True)
        plt.title("Population Density Distribution")
        plt.xlabel("Population Density")
        plt.ylabel("Frequency (log scale)")
        plt.show()


    while True:
        start = time.time()
        pop_map,_ = generate_population_density(covariance_models)
        print(time.time() - start)
        pop_map = np.where(pop_map == -9999, np.nan, pop_map)
        plot_histogram(pop_map)
        im2 = plt.imshow(pop_map, cmap="Blues", origin="upper", norm=colors.Normalize(vmin=0, vmax=np.nanpercentile(pop_map, 99)))
        plt.title("Population Density (GRF)")
        plt.colorbar(im2)
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.show()
