import numpy as np
import rasterio
import shapely
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

def generate_random_shapes_map(shape:tuple[int,int]=(512, 512), obstacle_size:int=100, rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    num_obstacles = rng.integers(2,12)

    polygons = [generate_random_polygon(shape, obstacle_size, rng=rng) for _ in range(num_obstacles)]
    map = rasterio.features.rasterize(polygons, out_shape=shape)

    return map

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
    return (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min())