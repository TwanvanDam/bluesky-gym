import numpy as np
import rasterio
import shapely


def generate_random_polygon(array_size: tuple[int,int], obstacle_size:int):
    num_vertices = np.random.randint(3,6)
    vertices = []
    centroid_x, centroid_y = np.random.uniform() * array_size[0], np.random.uniform() * array_size[1]
    for vertex in range(num_vertices):
        vertices.append((centroid_x + np.random.uniform(-1,1) * obstacle_size / 2, centroid_y + np.random.uniform(-1,1) * obstacle_size / 2))
       # Sort vertices by angle relative to centroid
    vertices.sort(key=lambda v: np.arctan2(v[1] - centroid_y, v[0] - centroid_x))
    return shapely.Polygon(vertices)

def generate_random_shapes_map(array_size:tuple[int,int], obstacle_size:int) -> np.ndarray:
    num_obstacles = np.random.randint(2,12)

    polygons = [generate_random_polygon(array_size, obstacle_size) for _ in range(num_obstacles)]

    map = rasterio.features.rasterize(polygons, out_shape=array_size)

    return map

def generate_population_density(shape=(512, 512), num_cities=100, base_occupancy=0.5):
    # 1. Create the base grid and add "City Seeds"
    grid = np.zeros(shape)
    rows = np.random.randint(0, shape[0], num_cities)
    cols = np.random.randint(0, shape[1], num_cities)
    grid[rows, cols] = np.random.exponential(scale=15.0, size=num_cities)

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
    rural_background = np.random.lognormal(mean=-1.0, sigma=0.5, size=shape) * base_occupancy

    # Combine the two
    combined_map = density_map + rural_background

    # 4. Final Normalization
    combined_map = np.maximum(combined_map, 0)
    return (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min())