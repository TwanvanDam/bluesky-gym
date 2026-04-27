import gstools as gs
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


def get_dataset_info(file_path: str):
    """Print dataset information to help you determine correct bounds"""
    with rasterio.open(file_path) as dataset:
        print(f"CRS: {dataset.crs}")
        print(f"Bounds: {dataset.bounds}")
        print(f"Size: {dataset.width} x {dataset.height} pixels")
        print(f"Resolution: {dataset.res}")


def load_tiff_to_numpy(file_path: str, lat_min=None, lat_max=None, lon_min=None, lon_max=None) -> np.ndarray:
    """
    Load TIFF as numpy array.
    If bounds are provided, they must be in the dataset's native CRS (not lat/lon).
    If no bounds provided, loads the entire dataset.
    """
    with rasterio.open(file_path) as dataset:
        bounds = dataset.bounds
        res_x, res_y = dataset.res
        crs = dataset.crs

        print(f"Dataset CRS: {crs}")
        print(f"Dataset bounds: {bounds}")

        # If no bounds specified, use entire dataset
        if lat_min is None or lat_max is None or lon_min is None or lon_max is None:
            print("No bounds specified; loading entire dataset")
            data = dataset.read(1)
        else:
            print(f"Requested bounds: x=[{lon_min}, {lon_max}], y=[{lat_min}, {lat_max}]")
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            lon_min, lat_min = transformer.transform(lon_min, lat_min)
            lon_max, lat_max = transformer.transform(lon_max, lat_max)
            print(f"Transformed bounds to dataset CRS: x=[{lon_min}, {lon_max}], y=[{lat_min}, {lat_max}]")
            # Calculate pixel coordinates
            col_min = max(0, int((lon_min - bounds.left) / res_x))
            col_max = min(dataset.width, int((lon_max - bounds.left) / res_x))
            row_min = max(0, int((bounds.top - lat_max) / res_y))
            row_max = min(dataset.height, int((bounds.top - lat_min) / res_y))

            print(f"Clipped pixel coordinates: col_min={col_min}, col_max={col_max}, row_min={row_min}, row_max={row_max}")

            if col_min >= col_max or row_min >= row_max:
                raise ValueError(
                    f"Invalid bounds after clipping. The coordinates don't intersect the dataset. "
                    f"Dataset CRS is {crs} with bounds {bounds}. "
                    f"Call get_dataset_info('{file_path}') to see available bounds."
                )

            # Read the data
            width = col_max - col_min
            height = row_max - row_min
            window = Window(col_min, row_min, width, height)
            data = dataset.read(1, window=window)

        print(f"Data shape: {data.shape}")
        if data.size > 0:
            print(f"Data min/max: {np.nanmin(data)}/{np.nanmax(data)}")
        print(f"Data dtype: {data.dtype}")

        # Clip negative / nodata values to 0
        clipped_data = np.clip(data, 0, np.inf)
        clipped_data = np.log1p(clipped_data)
        plt.imshow(clipped_data, cmap="Blues", origin="upper")
        plt.title("Clipped and Log-Transformed Population Density")
        plt.colorbar(label="Log(1 + Population Density)")
        plt.xlabel("X coordinate (pixels)")
        plt.ylabel("Y coordinate (pixels)")
        plt.show()
    return clipped_data

def make_variogram(data: np.ndarray, max_dist: float, n_lags: int, max_samples: int = 10_000, seed: int = 42):
    nrows, ncols = data.shape
    values = data.flatten().astype(float)

    print(f"Total flattened values: {len(values)}")
    print(f"Flattened data min/max: {np.nanmin(values)}/{np.nanmax(values)}")

    # Mask out nodata / NaN values
    mask = np.isfinite(values) & (values >= 0)
    valid_indices = np.nonzero(mask)[0]
    values = values[valid_indices]

    print(f"Valid points after filtering: {len(values)}")
    print(f"Valid values min/max: {np.min(values)}/{np.max(values)}")

    # Subsample to avoid O(N^2) blowup on large rasters
    n_valid = len(values)
    if n_valid > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_valid, size=max_samples, replace=False)
        valid_indices = valid_indices[idx]
        values = values[idx]
        print(f"Subsampled to {len(values)} points")

    # Compute x/y coordinates only for the sampled points (avoids full meshgrid)
    y_flat = (valid_indices // ncols).astype(float)
    x_flat = (valid_indices % ncols).astype(float)

    pos = np.array([x_flat, y_flat])

    # Create variogram
    bin_edges = np.linspace(0, max_dist, n_lags + 1)
    variogram = gs.vario_estimate(pos, values, bin_edges=bin_edges)

    return variogram

def plot_variogram_and_model(variogram):
    import matplotlib.pyplot as plt
    bin_centers, gamma = variogram

    # Single models
    single_models = {
        'Spherical': gs.Spherical(dim=2),
        'Exponential': gs.Exponential(dim=2),
        'Matern': gs.Matern(dim=2),
        'Stable': gs.Stable(dim=2),
        'Linear': gs.Linear(dim=2),
    }

    # 2-component summed models
    summed_models = {
        'Spherical + Linear': gs.Spherical(dim=2) + gs.Linear(dim=2),
        'Exponential + Linear': gs.Exponential(dim=2) + gs.Linear(dim=2),
        'Matern + Linear': gs.Matern(dim=2) + gs.Linear(dim=2),
        'Spherical + Spherical': gs.Spherical(dim=2) + gs.Spherical(dim=2),
        'Exponential + Exponential': gs.Exponential(dim=2) + gs.Exponential(dim=2),
    }

    # 3-component multi-scale models — mirrors the manual GRF's 3 spatial scales
    # (large cities / towns / neighborhoods)
    multiscale_models = {
        'Gaussian x3 (multiscale)': gs.Gaussian(dim=2) + gs.Gaussian(dim=2) + gs.Gaussian(dim=2),
        'Exponential x3 (multiscale)': gs.Exponential(dim=2) + gs.Exponential(dim=2) + gs.Exponential(dim=2),
        'Matern x3 (multiscale)': gs.Matern(dim=2) + gs.Matern(dim=2) + gs.Matern(dim=2),
        'Gaussian + Exponential + Linear': gs.Gaussian(dim=2) + gs.Exponential(dim=2) + gs.Linear(dim=2),
        'Matern + Gaussian + Linear': gs.Matern(dim=2) + gs.Gaussian(dim=2) + gs.Linear(dim=2),
    }

    all_models = {**single_models, **summed_models, **multiscale_models}

    # Create x values for model evaluation
    x_max = bin_centers[-1]
    x = np.linspace(0, x_max, 300)

    results = {}
    plt.figure(figsize=(14, 8))
    plt.scatter(bin_centers, gamma, color='orange', zorder=5, label='Empirical', s=60, edgecolors='black', linewidth=1.5)

    for name, m in all_models.items():
        try:
            m.fit_variogram(bin_centers, gamma, nugget=True)
            residuals = gamma - m.variogram(bin_centers)
            rmse = np.sqrt((residuals ** 2).mean())
            results[name] = rmse

            # Use different line styles for summed vs single models
            linestyle = '--' if '+' in name else '-'
            linewidth = 2.5 if '+' in name else 2
            plt.plot(x, m.variogram(x), label=f'{name} (RMSE={rmse:.4f})', linestyle=linestyle, linewidth=linewidth)
        except Exception as e:
            print(f"{name} failed: {e}")

    plt.legend(loc='best', fontsize=10)
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Semivariance', fontsize=12)
    plt.title('Variogram Model Comparison (including summed models)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if results:
        single_results   = {k: v for k, v in results.items() if '+' not in k}
        multiscale_results = {k: v for k, v in results.items() if k.endswith('(multiscale)') or k.count('+') >= 2}
        summed_results   = {k: v for k, v in results.items() if '+' in k and k not in multiscale_results}

        print("\n=== Single Models ===")
        for name, rmse in sorted(single_results.items(), key=lambda x: x[1]):
            print(f"{name}: RMSE={rmse:.4f}")

        if summed_results:
            print("\n=== 2-Component Summed Models ===")
            for name, rmse in sorted(summed_results.items(), key=lambda x: x[1]):
                print(f"{name}: RMSE={rmse:.4f}")

        if multiscale_results:
            print("\n=== 3-Component Multi-scale Models ===")
            for name, rmse in sorted(multiscale_results.items(), key=lambda x: x[1]):
                print(f"{name}: RMSE={rmse:.4f}")

        best = min(results, key=results.get)
        if best in multiscale_results:
            best_type = "Multi-scale"
        elif '+' in best:
            best_type = "Summed"
        else:
            best_type = "Single"
        print(f"\n✓ Best fit: {best} ({best_type})")
        print(f"  RMSE: {results[best]:.4f}")
    else:
        print("\nNo models fitted successfully")

def fit_best_model(variogram):
    """Fit all models and return the best one"""
    bin_centers, gamma = variogram

    single_models = {
        'Gaussian':    gs.Gaussian(dim=2),
        'Spherical':   gs.Spherical(dim=2),
        'Exponential': gs.Exponential(dim=2),
        'Matern':      gs.Matern(dim=2),
        'Stable':      gs.Stable(dim=2),
        'Linear':      gs.Linear(dim=2),
    }

    summed_models = {
        'Gaussian + Linear':    gs.Gaussian(dim=2) + gs.Linear(dim=2),
        'Spherical + Linear':   gs.Spherical(dim=2) + gs.Linear(dim=2),
        'Exponential + Linear': gs.Exponential(dim=2) + gs.Linear(dim=2),
        'Matern + Linear':      gs.Matern(dim=2) + gs.Linear(dim=2),
        'Spherical + Spherical':         gs.Spherical(dim=2) + gs.Spherical(dim=2),
        'Exponential + Exponential':     gs.Exponential(dim=2) + gs.Exponential(dim=2),
    }

    # 3-component models — mirrors manual GRF's large/medium/small scale hierarchy
    multiscale_models = {
        'Gaussian x3 (multiscale)':          gs.Gaussian(dim=2) + gs.Gaussian(dim=2) + gs.Gaussian(dim=2),
        'Exponential x3 (multiscale)':       gs.Exponential(dim=2) + gs.Exponential(dim=2) + gs.Exponential(dim=2),
        'Matern x3 (multiscale)':            gs.Matern(dim=2) + gs.Matern(dim=2) + gs.Matern(dim=2),
        'Gaussian + Exponential + Linear':   gs.Gaussian(dim=2) + gs.Exponential(dim=2) + gs.Linear(dim=2),
        'Matern + Gaussian + Linear':        gs.Matern(dim=2) + gs.Gaussian(dim=2) + gs.Linear(dim=2),
    }

    all_models = {**single_models, **summed_models, **multiscale_models}
    results = {}

    for name, m in all_models.items():
        try:
            m.fit_variogram(bin_centers, gamma, nugget=True)
            residuals = gamma - m.variogram(bin_centers)
            rmse = np.sqrt((residuals ** 2).mean())
            results[name] = (m, rmse)
        except Exception as e:
            print(f"{name} failed: {e}")

    if results:
        best_name = min(results, key=lambda x: results[x][1])
        best_model, best_rmse = results[best_name]
        print(f"✓ Best model: {best_name} (RMSE={best_rmse:.4f})")
        return best_model, best_name, best_rmse
    else:
        raise ValueError("No models fitted successfully")

def sample_from_model(model, size: int = 512, seed: int = 42) -> np.ndarray:
    """
    Sample a random field from the fitted variogram model

    Args:
        model: Fitted gstools covariance model
        size: Size of the output map (size x size)
        seed: Random seed for reproducibility

    Returns:
        Sampled field as numpy array of shape (size, size)
    """
    # Create a regular grid
    x = np.arange(0, size, dtype=float)
    y = np.arange(0, size, dtype=float)
    pos = np.meshgrid(x, y)

    # Create a random field generator using SRF
    print(f"Sampling from model with parameters: {model}")
    sampler = gs.SRF(model, seed=seed)

    # Generate the random field
    field = sampler(pos)

    # Reshape to (size, size) if needed
    if field.ndim == 1:
        field = field.reshape((size, size))

    return field

def sample_and_save(model, output_path: str = None, size: int = 512, seed: int = 42, plot: bool = True):
    """
    Sample from model and optionally save/display.

    Post-processing matches the manual generate_population_density pipeline:
      1. Shift field by the 40th percentile  (same threshold as manual GRF)
      2. Apply exp  (heavy-tailed density, same as manual GRF)
      3. Clip minimum to 1  (no zeros, same as manual GRF)
    """
    print(f"\n=== Sampling {size}x{size} map from model ===")
    field = sample_from_model(model, size=size, seed=seed)

    # Standardise to zero-mean, unit-variance — matching _generate_grf in map_generators.py
    field = (field - field.mean()) / (field.std() + 1e-8)

    # Mirror the manual GRF post-processing exactly
    field = field - np.percentile(field, 40)   # step 1: shift
    sampled_field = np.exp(field)               # step 2: heavy-tailed density
    sampled_field[sampled_field <= 1] = 1       # step 3: no zeros

    print(f"Sampled field shape: {sampled_field.shape}")
    print(f"Sampled field min/max: {np.min(sampled_field):.4f}/{np.max(sampled_field):.4f}")
    print(f"Sampled field mean: {np.mean(sampled_field):.4f}")
    print(f"Sampled field std: {np.std(sampled_field):.4f}")
    print(f"Sampled field median: {np.median(sampled_field):.4f}")

    if output_path:
        from rasterio.transform import Affine
        transform = Affine.identity()
        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=sampled_field.shape[0], width=sampled_field.shape[1],
            count=1, dtype=rasterio.float32, transform=transform,
        ) as dst:
            dst.write(sampled_field.astype(rasterio.float32), 1)
        print(f"✓ Saved to {output_path}")

    if plot:
        from matplotlib.colors import LogNorm
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(sampled_field, cmap='viridis',
                       norm=LogNorm(vmin=sampled_field.min(), vmax=sampled_field.max()))
        ax.set_title(f'Sampled {size}x{size} Population Map', fontsize=14)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Population density (log scale)', rotation=270, labelpad=20)
        plt.tight_layout()
        plt.show()

    return sampled_field

def compare_models(fitted_model, size: int = 512, seed: int = 42):
    """
    Compare the fitted variogram model against the manual GRF population generator

    Args:
        fitted_model: Fitted gstools model
        size: Map size (size x size)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (fitted_map, manual_map)
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    # Import the manual generator
    from bluesky_gym.maps.map_generators import generate_population_density

    print("\n" + "="*80)
    print("COMPARING FITTED MODEL vs MANUAL GRF GENERATOR")
    print("="*80)

    # Generate from fitted model (same post-processing as manual GRF)
    print("\n--- Sampling from Fitted Variogram Model ---")
    fitted_map = sample_from_model(fitted_model, size=size, seed=seed)
    fitted_map = (fitted_map - fitted_map.mean()) / (fitted_map.std() + 1e-8)  # standardise
    fitted_map = fitted_map - np.percentile(fitted_map, 40)
    fitted_map = np.exp(fitted_map)
    fitted_map[fitted_map <= 1] = 1

    # Generate from manual GRF
    print("\n--- Sampling from Manual GRF Generator ---")
    manual_map = generate_population_density(shape=(size, size), rng=np.random.default_rng(seed))

    # Compute statistics
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)

    stats_comparison = {
        'Metric': [],
        'Fitted Model': [],
        'Manual GRF': [],
        'Difference': []
    }

    metrics = {
        'Min': (np.min, lambda x, y: x - y),
        'Max': (np.max, lambda x, y: x - y),
        'Mean': (np.mean, lambda x, y: x - y),
        'Median': (np.median, lambda x, y: x - y),
        'Std Dev': (np.std, lambda x, y: x - y),
        'Skewness': (lambda x: stats.skew(x.flatten()), lambda x, y: x - y),
        'Kurtosis': (lambda x: stats.kurtosis(x.flatten()), lambda x, y: x - y),
        'Coverage (>0)': (lambda x: np.sum(x > 0) / x.size * 100, lambda x, y: y - x),  # percentage
    }

    for name, (func, diff_func) in metrics.items():
        fitted_val = func(fitted_map)
        manual_val = func(manual_map)
        diff = diff_func(fitted_val, manual_val)

        stats_comparison['Metric'].append(name)
        stats_comparison['Fitted Model'].append(f"{fitted_val:.4f}")
        stats_comparison['Manual GRF'].append(f"{manual_val:.4f}")
        stats_comparison['Difference'].append(f"{diff:.4f}")

        print(f"{name:20s} | Fitted: {fitted_val:10.4f} | Manual: {manual_val:10.4f} | Diff: {diff:+.4f}")

    # Spatial correlation analysis
    print("\n" + "="*80)
    print("SPATIAL CORRELATION ANALYSIS")
    print("="*80)

    # Compute empirical variograms from generated maps
    from scipy.spatial import distance_matrix

    # Sample points for variogram computation
    n_sample_points = 1000
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(size * size, size=min(n_sample_points, size * size), replace=False)

    fitted_values = fitted_map.flatten()[sample_indices]
    manual_values = manual_map.flatten()[sample_indices]

    sample_coords = np.array(np.unravel_index(sample_indices, (size, size))).T
    distances = distance_matrix(sample_coords, sample_coords)

    # Bin distances
    dist_bins = np.linspace(0, np.sqrt(2) * size / 2, 20)

    fitted_vario_empirical = []
    manual_vario_empirical = []
    bin_centers = []

    for i in range(len(dist_bins) - 1):
        mask = (distances >= dist_bins[i]) & (distances < dist_bins[i+1])
        if np.sum(mask) > 0:
            bin_centers.append((dist_bins[i] + dist_bins[i+1]) / 2)
            fitted_diff = fitted_values[:, None] - fitted_values[None, :]
            manual_diff = manual_values[:, None] - manual_values[None, :]
            fitted_vario_empirical.append(np.mean(fitted_diff[mask] ** 2 / 2))
            manual_vario_empirical.append(np.mean(manual_diff[mask] ** 2 / 2))

    print(f"Empirical variogram correlation at different lags:")
    print(f"{'Distance':>12} | {'Fitted Vario':>15} | {'Manual Vario':>15} | {'Ratio':>10}")
    for i, (dist, fv, mv) in enumerate(zip(bin_centers, fitted_vario_empirical, manual_vario_empirical)):
        ratio = fv / mv if mv > 0 else 0
        print(f"{dist:12.2f} | {fv:15.4f} | {mv:15.4f} | {ratio:10.2f}")

    # Visualization
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Maps
    im0 = axes[0, 0].imshow(fitted_map, cmap='viridis')
    axes[0, 0].set_title('Fitted Model (log-transformed back)', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(manual_map, cmap='viridis')
    axes[0, 1].set_title('Manual GRF Generator', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1])

    difference_map = fitted_map - manual_map
    im2 = axes[0, 2].imshow(difference_map, cmap='RdBu_r')
    axes[0, 2].set_title('Difference (Fitted - Manual)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Distributions
    axes[1, 0].hist(fitted_map.flatten(), bins=50, alpha=0.7, label='Fitted', color='blue', edgecolor='black')
    axes[1, 0].hist(manual_map.flatten(), bins=50, alpha=0.7, label='Manual', color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Population Density')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Value Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')

    # Log-scale distributions
    fitted_nonzero = fitted_map[fitted_map > 0].flatten()
    manual_nonzero = manual_map[manual_map > 0].flatten()

    axes[1, 1].hist(np.log1p(fitted_nonzero), bins=50, alpha=0.7, label='Fitted', color='blue', edgecolor='black')
    axes[1, 1].hist(np.log1p(manual_nonzero), bins=50, alpha=0.7, label='Manual', color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Log(1 + Population Density)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Log-scale Distribution Comparison')
    axes[1, 1].legend()

    # Empirical variograms
    axes[1, 2].plot(bin_centers, fitted_vario_empirical, 'o-', label='Fitted Model', linewidth=2, markersize=6)
    axes[1, 2].plot(bin_centers, manual_vario_empirical, 's-', label='Manual GRF', linewidth=2, markersize=6)
    axes[1, 2].set_xlabel('Distance (pixels)')
    axes[1, 2].set_ylabel('Semivariance')
    axes[1, 2].set_title('Empirical Variogram from Generated Maps')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Fitted Model Map:  mean={np.mean(fitted_map):.4f}, std={np.std(fitted_map):.4f}, range=[{np.min(fitted_map):.4f}, {np.max(fitted_map):.4f}]")
    print(f"Manual GRF Map:    mean={np.mean(manual_map):.4f}, std={np.std(manual_map):.4f}, range=[{np.min(manual_map):.4f}, {np.max(manual_map):.4f}]")
    print(f"\nMaps generated successfully for visual and statistical comparison.")

    return fitted_map, manual_map

if __name__ == "__main__":
    # Example usage
    file_path = "scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff"

    # First, check dataset info
    print("=== Dataset Info ===")
    get_dataset_info(file_path)

    print("\n=== Loading data ===")
    # Load full dataset (no bounds specified)
    data = load_tiff_to_numpy(file_path, lat_min=49, lat_max=56, lon_min=0, lon_max=7)  # Example bounds for Europe; adjust as needed

    print("\n=== Computing variogram ===")
    variogram = make_variogram(data, max_dist=300, n_lags=50)

    print("\n=== Plotting ===")
    plot_variogram_and_model(variogram)

    print("\n=== Fitting best model ===")
    best_model, best_name, best_rmse = fit_best_model(variogram)

    print("\n=== Sampling 512x512 map from best model ===")
    sampled_map = sample_and_save(best_model, size=512, seed=42)

    print("\n=== COMPARISON: Fitted Model vs Manual GRF ===")
    fitted_map, manual_map = compare_models(best_model, size=512, seed=42)

