import numpy as np
import rasterio
import rasterio.features
import shapely
from matplotlib import colors

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


# ---------------------------------------------------------------------------
# Multi-scale GRF-based population density generator
# ---------------------------------------------------------------------------

def _generate_grf(shape: tuple[int, int], correlation_length: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Generate a Gaussian Random Field via FFT spectral synthesis.

    The power spectrum is defined by a squared-exponential (Gaussian)
    covariance kernel with the given *correlation_length* (in pixels).

    Parameters
    ----------
    shape : tuple[int, int]
        (rows, cols) of the output field.
    correlation_length : float
        Characteristic spatial scale of the field (in pixels).
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    np.ndarray
        A 2-D array with zero-mean, unit-variance Gaussian statistics and
        spatial correlations governed by *correlation_length*.
    """
    rows, cols = shape

    # Frequency grids (centered at 0)
    freq_y = np.fft.fftfreq(rows)
    freq_x = np.fft.fftfreq(cols)
    fy, fx = np.meshgrid(freq_y, freq_x, indexing="ij")

    # Squared-exponential power spectrum  P(k) ∝ exp(-2 π² L² k²)
    # where L = correlation_length and k² = fx² + fy²
    power_spectrum = np.exp(
        -2.0 * (np.pi * correlation_length) ** 2 * (fx ** 2 + fy ** 2)
    )

    # Draw random Fourier coefficients (complex white noise) and colour them
    white_noise = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )
    field_fft = np.sqrt(power_spectrum) * white_noise
    field = np.fft.ifft2(field_fft).real

    # Standardise to zero-mean, unit-variance
    std = field.std()
    if std > 0:
        field = (field - field.mean()) / std

    return field


def generate_population_density(
    shape: tuple[int, int] = (512, 512),
    scales: list[tuple[float, float]] | None = None,
    noise_std: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate a synthetic population density map using multi-scale GRFs.

    The pipeline reproduces key properties of real population distributions:
    strong spatial clustering, multi-scale structure, smooth gradients,
    heavy-tailed densities, and local variability.

    Pipeline
    --------
    1. Sum several Gaussian Random Fields at different correlation lengths
       (large → metropolitan regions, medium → towns, small → neighborhoods).
    2. Add fine-grained Gaussian noise.
    3. Apply ``exp`` for a heavy-tailed (≈ lognormal) density.
    4. Compress with ``log1p`` and normalise to [0, 1].

    Parameters
    ----------
    shape : tuple[int, int], default (512, 512)
        Spatial dimensions (rows, cols) of the output map.
    scales : list of (correlation_length, weight) pairs, optional
        Each entry defines one GRF component.  Defaults to::

            [(50, 1.0), (20, 0.6), (8, 0.3)]

    noise_std : float, default 0.05
        Standard deviation of additive Gaussian noise applied after
        combining the GRF layers.
    rng : np.random.Generator, optional
        Seeded random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        2-D array of shape *shape* with values in [0, 1].
    """
    rng = rng or np.random.default_rng()

    if scales is None:
        scales = [
            (70, 1.0),   # Large cities / metropolitan regions
            (20, 0.2),   # Town clusters / suburbs
            (3,  0.1),   # Neighborhood-level variability
        ]

    # Step 1 & 2: weighted sum of multiscale GRFs
    field = np.zeros(shape, dtype=np.float64)
    for correlation_length, weight in scales:
        field += weight * _generate_grf(shape, correlation_length, rng)

    # Step 3: fine-grained noise for local heterogeneity
    field += rng.normal(0, noise_std, size=shape)

    # Step 4: exponential transform → heavy-tailed density
    population = field - np.percentile(field, 40)
    population = np.exp(population)

    population[population <= 1] = 1
    return population



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)
    while True:
        # 3. Population density (multi-scale GRF)
        pop_map = generate_population_density(rng=rng, shape=(1024,1024))
        im2 = plt.imshow(pop_map, cmap="Blues", origin="upper", norm=colors.LogNorm(vmin=pop_map.min(), vmax=pop_map.max()))
        plt.title("Population Density (GRF)")
        plt.colorbar(im2)
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.show()
