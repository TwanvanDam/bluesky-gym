import time
from itertools import combinations_with_replacement

import gstools as gs
import matplotlib.pyplot as plt
from matplotlib import colors
import rasterio
import numpy as np
from tqdm.auto import tqdm


def read_population_map(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        density = src.read(1).astype(np.float32)
    return density

def mask_empty_values(density: np.ndarray, threshold: float = -1) -> np.ndarray:
    masked_density = np.where(density < threshold, np.nan, density)
    return masked_density

def plot_map(density: np.ndarray):
    plt.imshow(density, cmap="Blues", origin="upper", norm=norm)
    plt.colorbar(label="Population Density")
    plt.title("Population Density Map")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.show()

def plot_distribution(density: np.ndarray):
    print(f"Min, Max, Mean, Median: {np.nanmin(density):.2f}, {np.nanmax(density):.2f}, {np.nanmean(density):.2f}, {np.nanmedian(density):.2f}")
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


def plot_distributution(density: np.ndarray):
    """Backward-compatible alias for the old typo'ed function name."""
    plot_distribution(density)

def compare_distributions(original: np.ndarray, synthetic: np.ndarray):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(original[~np.isnan(original)].flatten(), bins=50, log=True, alpha=0.5, label="Original")
    plt.hist(synthetic[~np.isnan(synthetic)].flatten(), bins=50, log=True, alpha=0.5, label="Synthetic")
    plt.title("Population Density Distribution")
    plt.xlabel("Population Density")
    plt.ylabel("Frequency (log scale)")
    plt.legend()

    plt.subplot(1, 2, 2)
    sorted_original = np.sort(original[~np.isnan(original)].flatten())
    cumulative_original = np.arange(len(sorted_original)) / len(sorted_original)
    sorted_synthetic = np.sort(synthetic[~np.isnan(synthetic)].flatten())
    cumulative_synthetic = np.arange(len(sorted_synthetic)) / len(sorted_synthetic)
    plt.plot(sorted_original, cumulative_original, label="Original")
    plt.plot(sorted_synthetic, cumulative_synthetic, label="Synthetic")
    plt.xscale("log")
    plt.title("Cumulative Distribution of Population Density")
    plt.xlabel("Population Density (log scale)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

def subsample_array(array: np.ndarray, factor: int, method: str = "mean") -> np.ndarray:
    """Subsample a 2D array by a given factor using the specified method."""
    array = array[:array.shape[0] - array.shape[0] % factor, :array.shape[1] - array.shape[1] % factor] # Trim to be divisible by factor
    if method == "mean":
        return array.reshape(array.shape[0] // factor, factor, array.shape[1] // factor, factor).mean(axis=(1, 3))
    elif method == "median":
        return np.median(array.reshape(array.shape[0] // factor, factor, array.shape[1] // factor, factor), axis=(1, 3))
    else:
        raise ValueError(f"Unsupported subsampling method: {method}")


def compute_zero_fraction(array: np.ndarray, zero_threshold: float = 0.0) -> float:
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid <= zero_threshold))


def winsorize_upper_tail(array: np.ndarray, upper_quantile: float = 99.8) -> np.ndarray:
    x = np.array(array, copy=True)
    valid = ~np.isnan(x)
    if not np.any(valid):
        return x
    cap = np.nanpercentile(x, upper_quantile)
    x[valid] = np.minimum(x[valid], cap)
    return x


def quantile_map_to_target(source: np.ndarray, target: np.ndarray, n_quantiles: int = 512) -> np.ndarray:
    """Map source values to target distribution via empirical CDF matching."""
    out = np.array(source, copy=True)
    src_valid = ~np.isnan(source)
    tgt_valid = ~np.isnan(target)
    if not np.any(src_valid) or not np.any(tgt_valid):
        return out

    q = np.linspace(0.0, 1.0, n_quantiles)
    src_q = np.quantile(source[src_valid], q)
    tgt_q = np.quantile(target[tgt_valid], q)

    # np.interp expects strictly increasing x-coordinates.
    src_q_unique, unique_idx = np.unique(src_q, return_index=True)
    tgt_q_unique = tgt_q[unique_idx]
    if src_q_unique.size == 1:
        out[src_valid] = tgt_q_unique[0]
        return out

    out[src_valid] = np.interp(
        source[src_valid],
        src_q_unique,
        tgt_q_unique,
        left=tgt_q_unique[0],
        right=tgt_q_unique[-1],
    )
    return out


def calibrate_zero_inflation(array: np.ndarray, target_zero_fraction: float, zero_threshold: float = 0.0) -> np.ndarray:
    """Force a target fraction of near-zero cells by thresholding the lowest values."""
    x = np.array(array, copy=True)
    valid = ~np.isnan(x)
    if not np.any(valid):
        return x

    target_zero_fraction = float(np.clip(target_zero_fraction, 0.0, 1.0))
    if target_zero_fraction <= 0.0:
        return x

    cutoff = np.nanquantile(x[valid], target_zero_fraction)
    x[(x <= cutoff) & valid] = zero_threshold
    return x


def print_distribution_summary(label: str, array: np.ndarray, zero_threshold: float = 0.0) -> None:
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        print(f"{label}: empty")
        return
    p50, p90, p99 = np.percentile(valid, [50, 90, 99])
    print(
        f"{label}: min={valid.min():.3f}, mean={valid.mean():.3f}, p50={p50:.3f}, "
        f"p90={p90:.3f}, p99={p99:.3f}, max={valid.max():.3f}, "
        f"zero_frac={np.mean(valid <= zero_threshold):.3f}"
    )


def calibrate_synthetic_distribution(
    synthetic: np.ndarray,
    target: np.ndarray,
    *,
    upper_quantile: float = 99.8,
    n_quantiles: int = 512,
    zero_threshold: float = 0.0,
) -> np.ndarray:
    """Post-process SRF output to better match target sparsity and heavy tail."""
    x = np.clip(synthetic, 0, None)
    x = winsorize_upper_tail(x, upper_quantile=upper_quantile)
    x = quantile_map_to_target(x, target, n_quantiles=n_quantiles)
    target_zero_fraction = compute_zero_fraction(target, zero_threshold=zero_threshold)
    x = calibrate_zero_inflation(x, target_zero_fraction=target_zero_fraction, zero_threshold=zero_threshold)
    # A second pass helps recover upper-tail shape after zero-thresholding.
    x = quantile_map_to_target(x, target, n_quantiles=n_quantiles)
    return np.clip(x, 0, None)

# def quantile_scale(array: np.ndarray, target_median: float, target_q99: float) -> np.ndarray:
#     x = np.clip(array, 0, None)
#     p50, p99 = np.nanpercentile(x, [50, 99])
#     if p99 == p50:
#         return x
#     a = (target_q99 - target_median) / (p99 - p50)
#     b = target_median - a * p50
#     return np.clip(a * x + b, 0, None)
#
# def min_quantile_scale(array: np.ndarray, target_min: float, target_q95: float) -> np.ndarray:
#     x = np.clip(array, 0, None)
#     p0 = np.nanpercentile(x, 0)
#     p95 = np.nanpercentile(x, 95)
#     if p95 == p0:
#         return x
#     a = (target_q95 - target_min) / (p95 - p0)
#     b = target_min - a * p0
#     return np.clip(a * x + b, 0, None)
#
#
# def softplus(array: np.ndarray, beta: float) -> np.ndarray:
#     return np.log1p(np.exp(beta * array)) / beta
#
# def z(array: np.ndarray) -> np.ndarray:
#     mean = np.nanmean(array)
#     std = np.nanstd(array)
#     return (array - mean) / (std + 1e-8)

if __name__ == '__main__':
    from gstools import vario_estimate_unstructured
    path = "scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff"
    density = read_population_map(path)
    density = mask_empty_values(density)
    # density = np.log1p(density)
    norm = colors.Normalize(vmin=0, vmax=np.nanpercentile(density, 99.5))
    benelux = density[2000:2450, 2700:3300]
    print(norm.vmax)

    print(f"shape: {benelux.shape}, minimum:  {np.nanmin(benelux)}, maximum: {np.nanmax(benelux)}, empty: {np.isnan(benelux).sum() / benelux.size * 100:.2f}%")
    print(f"Mean population density: {np.nanmean(benelux):.2f} people/km^2")
    plot_map(benelux)
    # plot_distributution(benelux)
    factor = 4
    subsampled_map = subsample_array(benelux, factor=factor, method="mean")
    print(f"Subsampled shape: {subsampled_map.shape}, minimum:  {np.nanmin(subsampled_map)}, maximum: {np.nanmax(subsampled_map)}, empty: {np.isnan(subsampled_map).sum() / subsampled_map.size * 100:.2f}%")
    plot_map(subsampled_map)
    plot_distribution(subsampled_map)
    normalized_subsampled = np.log1p(subsampled_map)

    grid_x, grid_y = factor * np.mgrid[0:normalized_subsampled.shape[0], 0:normalized_subsampled.shape[1]]

    values_non_masked = normalized_subsampled[~np.isnan(normalized_subsampled)].flatten()
    # np.random.seed(42)
    # random_indices = np.random.randint(0, len(values_non_masked), size=10000)
    x_samples = grid_x[~np.isnan(normalized_subsampled)].flatten()
    y_samples = grid_y[~np.isnan(normalized_subsampled)].flatten()
    #
    bin_center, gamma = vario_estimate_unstructured([x_samples, y_samples], values_non_masked)
    best_rmse = float('inf')
    best_model = None
    cov_models = [gs.Gaussian, gs.Integral] # gs.Exponential,  gs.Spherical, gs.Matern
    for model_components in tqdm(combinations_with_replacement(cov_models, 3)):
        print(f"Fitting model: {model_components}")
        model = model_components[0](dim=2)
        for model_component in model_components[1:]:
            model = model + model_component(dim=2)
        try:
            model.fit_variogram(bin_center, gamma, nugget=False, weights="inv")
        except:
            print("Fitting failed for model:", model_components)
            continue

        residuals = gamma - model.variogram(bin_center)
        rmse = np.sqrt((residuals ** 2).mean())
        if rmse < best_rmse:
            best_rmse, best_model = rmse, model
        # print(f"Model: {model_components}, RMSE: {rmse:.4f}")
        # print(f"Fitted parameters: {best_model.models}, RMSE: {best_rmse:.4f}")

    print("Fitted model:", best_model)
    while True:
        start = time.time()
        mean = np.nanmean(normalized_subsampled)
        print(mean)
        srf = gs.SRF(best_model, mean=mean)
        synthetic_log = srf.structured((np.arange(0, benelux.shape[0], factor),
                                           np.arange(0, benelux.shape[1], factor)))
        print(time.time() - start)
        # synthetic = np.expm1(synthetic_log)
        # np.clip(synthetic, 0, np.percentile(synthetic, 99.9), out=synthetic)
        # np.clip(synthetic_log, 0, None, out=synthetic_log)
        synthetic = np.expm1(synthetic_log)
        synthetic = calibrate_synthetic_distribution(
            synthetic,
            subsampled_map,
            upper_quantile=99.8,
            n_quantiles=100,
            zero_threshold=0.0,
        )
        print_distribution_summary("Target", benelux)
        print_distribution_summary("Synthetic", subsampled_map)
        ocean_model = gs.Exponential(dim=2, len_scale=300, var=0.5) + gs.Gaussian(dim=2, len_scale=300, var=1)
        ocean_srf = gs.SRF(ocean_model)
        ocean = ocean_srf.structured((np.arange(0, benelux.shape[0], factor), np.arange(0, benelux.shape[1], factor)))

        synthetic_masked = np.where(ocean < np.percentile(ocean, 10), np.nan, synthetic)

        plot_map(synthetic_masked)
        compare_distributions(subsampled_map, synthetic)
