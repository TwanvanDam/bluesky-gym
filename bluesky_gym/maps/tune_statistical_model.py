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
    plt.imshow(np.log1p(density), cmap="Blues", origin="upper", norm=norm)
    plt.colorbar(label="Population Density")
    plt.title("Population Density Map")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.show()

def plot_histogram(density: np.ndarray):
    plt.hist(density[~np.isnan(density)].flatten(), bins=50, log=True)
    plt.title("Population Density Distribution")
    plt.xlabel("Population Density")
    plt.ylabel("Frequency (log scale)")
    plt.show()

if __name__ == '__main__':
    from gstools import vario_estimate_unstructured
    path = "scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff"
    density = read_population_map(path)
    density = mask_empty_values(density)
    norm = colors.Normalize(vmin=0, vmax=np.log1p(np.nanpercentile(density, 99.9)))
    benelux = density[2000:2450, 2700:3300]
    print(norm.vmax)

    print(density.shape, np.nanmin(benelux),np.nanmax(benelux), np.isnan(benelux).sum() / benelux.size)
    plot_map(benelux)
    plot_histogram(benelux)
    grid_x, grid_y = np.mgrid[0:benelux.shape[0], 0:benelux.shape[1]]

    values_non_masked = benelux[~np.isnan(benelux)].flatten()
    np.random.seed(42)
    random_indices = np.random.randint(0, len(values_non_masked), size=10000)
    values_sampled = np.log1p(values_non_masked[random_indices])
    x_samples = grid_x[~np.isnan(benelux)].flatten()[random_indices]
    y_samples = grid_y[~np.isnan(benelux)].flatten()[random_indices]

    bin_center, gamma = vario_estimate_unstructured((x_samples, y_samples), values_sampled)
    best_rmse = float('inf')
    best_model = None
    cov_models = [gs.Exponential, gs.Gaussian, gs.Matern, gs.Integral, gs.Stable, gs.Spherical]
    for model_components in tqdm(combinations_with_replacement(cov_models, 3)):
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
        mean = np.nanmean(np.log1p(benelux))
        print(mean)
        srf = gs.SRF(best_model, mean=np.nanmean(np.log1p(benelux)))
        synthetic_log = srf.structured((np.arange(benelux.shape[0]),
                                            np.arange(benelux.shape[1])))
        print(time.time() - start)
        synthetic = np.expm1(synthetic_log)
        np.clip(synthetic, 0, np.percentile(synthetic, 99.9), out=synthetic)

        ocean_model = gs.Exponential(dim=2, len_scale=300) + gs.Gaussian(dim=2, len_scale=300)
        ocean_srf = gs.SRF(ocean_model)
        ocean = ocean_srf.structured((np.arange(benelux.shape[0]), np.arange(benelux.shape[1])))

        synthetic_masked = np.where(ocean < np.percentile(ocean, 25), np.nan, synthetic)

        plot_map(synthetic_masked)
        plot_histogram(synthetic_masked)
