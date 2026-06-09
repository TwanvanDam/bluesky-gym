"""Calibrate and visualise the domain-randomization value transforms.

Anchors each value transform (gamma / floor / scale) so the transformed sim
population field matches the EDDF reference *median*, and picks the gamma range
from a mean-inflation budget. All statistics use the full valid population
(zeros included, nodata/sea excluded) to match how the noise reward and
``MapSource.mean_value`` treat pixels.

Run: ``python -m scripts.apply_transforms``
"""
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer
from rasterio.windows import from_bounds

from bluesky_gym.maps.map_transforms import GammaCorrection, FloorRaise, ScaleValues

# --- Configuration ----------------------------------------------------------
MAP_PATH = Path("scripts/population_maps/europe_3035_1km.tif")
EDDF = (50.0379, 8.5622)          # reference airport (lat, lon)
SIM_HALF_KM = 400.0               # half-extent of the reference box (full box = 2x on a side)
CLIP_PERCENTILE = 99.9            # population clip; matches the reward's clip_noise_reward path
GAMMA_GRID = np.linspace(0.5, 1.0, 51)
INFLATION_BAND = (1.0, 2.0)       # (lower, upper) allowed mean-inflation factor for the gamma range
SEED = 42


# --- Raster I/O -------------------------------------------------------------
def _valid_pixels(data: np.ndarray, nodata) -> np.ndarray:
    """Flatten to finite, non-negative pixels (drops the nodata sentinel / sea)."""
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data[np.isfinite(data) & (data >= 0)]


def read_full_dataset(src) -> np.ndarray:
    """All valid pixels of the raster, in source units."""
    return _valid_pixels(src.read(1).astype(np.float64), src.nodata)


def read_box(src, left, bottom, right, top) -> np.ndarray:
    """Valid pixels within (left, bottom, right, top) in source CRS."""
    win = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
    return _valid_pixels(src.read(1, window=win).astype(np.float64), src.nodata)


def latlon_box_to_native(transformer, lat_c, lon_c, half_km):
    """Square box of half-width ``half_km`` around a lat/lon point, in the source CRS (metres)."""
    x_c, y_c = transformer.transform(lon_c, lat_c)
    half_m = half_km * 1000.0
    return x_c - half_m, y_c - half_m, x_c + half_m, y_c + half_m


# --- Calibration ------------------------------------------------------------
def calibrate_gamma(sim: np.ndarray, target: np.ndarray, clip_percentile: float) -> tuple[float, float]:
    """Gamma range (low, high) whose mean-inflation stays within ``INFLATION_BAND``.

    Under the median-matched power law ``c = target_median / sim_median^γ``, the
    median maps to the target for *every* γ — so γ only controls the distribution
    shape, and hence the mean inflation, which rises monotonically with γ. The
    per-episode range therefore runs from γ_low (mildest, inflation ≈ lower bound)
    to γ_high (most aggressive, inflation ≈ upper bound). Numerator and denominator
    are both clipped: the reward always clips the population map, so the clean
    baseline it is compared against is clipped too.
    """
    lower, upper = INFLATION_BAND
    clip = np.percentile(sim, clip_percentile)
    target_median = np.median(target)
    sim_median = np.median(sim)
    base_mean = np.mean(np.clip(sim, None, clip))

    gamma_low: float | None = None   # smallest γ with inflation ≥ lower bound
    gamma_high: float | None = None  # largest γ with inflation ≤ upper bound
    for gamma in GAMMA_GRID:
        c = target_median / (sim_median ** gamma)
        inflation = np.mean(np.clip(np.power(sim, gamma) * c, 0, clip)) / base_mean
        if inflation <= upper:
            gamma_high = gamma
        if gamma_low is None and inflation >= lower:
            gamma_low = gamma

    return gamma_low, gamma_high


def calibrate_scale(sim: np.ndarray, target: np.ndarray) -> float:
    """Multiplier mapping the sim median onto the target median."""
    return float(np.median(target) / np.median(sim))


def calibrate_floor(sim: np.ndarray, target: np.ndarray) -> float:
    """Additive offset mapping the sim median onto the target median."""
    return float(np.median(target) - np.median(sim))


# --- Transform helpers ------------------------------------------------------
def param_range(transform) -> tuple[float, float]:
    """The (low, high) per-episode sampling range of a value transform."""
    if isinstance(transform, GammaCorrection):
        return transform.gamma
    if isinstance(transform, ScaleValues):
        return transform.factor
    if isinstance(transform, FloorRaise):
        return transform.floor
    raise TypeError(f"no known range field for {type(transform).__name__}")


def value_fn_at(transform, param: float):
    """Concrete pixel function with the transform's range collapsed to ``param``,
    so f(x) is deterministic (used to draw range endpoints on the tone curve)."""
    if isinstance(transform, GammaCorrection):
        t = transform.model_copy(update={"gamma": (param, param)})
        t.resolve(transform._base_value)
    elif isinstance(transform, ScaleValues):
        t = transform.model_copy(update={"factor": (param, param)})
    elif isinstance(transform, FloorRaise):
        t = transform.model_copy(update={"floor": (param, param)})
    else:
        raise TypeError(f"unsupported transform {type(transform).__name__}")
    return t._sample(np.random.default_rng())


def sample_transformed(transform, data: np.ndarray, rng: np.random.Generator, clip: float) -> np.ndarray:
    """Apply one per-episode draw of ``transform`` to ``data`` and clip (as the reward does)."""
    return np.clip(transform._sample(rng)(data), 0, clip)


def apply_extreme(transform, data: np.ndarray, clip: float) -> np.ndarray:
    """Apply ``transform`` at the high (most aggressive) end of its range and clip.

    The high end is the worst case for every value transform: largest γ (most mean
    inflation), largest scale factor, largest floor. Deterministic — no RNG draw."""
    high = param_range(transform)[1]
    return np.clip(value_fn_at(transform, high)(data), 0, clip)


# --- Plotting ---------------------------------------------------------------
def plot_cdf(distributions: list[tuple[str, np.ndarray, str]], clip: float) -> None:
    """Empirical CDFs (log-x) of each distribution, annotated with median/mean/p75.

    Statistics use the full valid population (zeros included, nodata excluded
    upstream) — matching the reward / ``mean_value`` — so the median annotation
    equals the calibration anchor. Zeros fall left of the log x-limit and simply
    lift the curve's starting fraction.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, vals, color in distributions:
        vals_plot = np.clip(vals, 0, clip)
        sorted_v = np.sort(vals_plot)
        ecdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        med, mean, p75 = np.median(vals_plot), np.mean(vals_plot), np.percentile(vals_plot, 75)
        label = f"{name}\n  median={med:,.1f}  mean={mean:,.1f}  p75={p75:,.1f}  n={len(vals_plot):,}"
        ax.plot(sorted_v, ecdf, color=color, linewidth=2.2, label=label)
        ax.plot([med], [0.5], "o", color=color, markersize=7, markeredgecolor="k", markeredgewidth=0.8)

    for y, text in ((0.5, "median (50%)"), (0.75, "75th pct")):
        ax.axhline(y, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.text(clip * 0.98, y + 0.01, text, ha="right", va="bottom", fontsize=9, color="gray")
    ax.axvline(clip, color="gray", linestyle="--", linewidth=1.2, label=f"p{CLIP_PERCENTILE} clip ({clip:,.0f})")

    ax.set_xscale("log")
    ax.set_xlim(1, clip * 1.1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Population per km²")
    ax.set_ylabel("Fraction of pixels ≤ value  (empirical CDF)")
    ax.set_title("Population density distribution — effect of value transform")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/augmentation/transformed_cdf.png")
    plt.show()


def plot_tone_curves(transforms: list[tuple[object, str]], sim_median: float,
                     target_median: float, clip: float) -> None:
    """Input→output tone curve per transform, drawn at the extreme (high) end of its range."""
    x = np.logspace(0, 5, num=500)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, x, label="Identity", color="black")
    for transform, color in transforms:
        high = param_range(transform)[1]
        ax.plot(x, value_fn_at(transform, high)(x), color=color,
                label=f"{transform.type} (max={high:.2g})")
    ax.scatter([sim_median], [target_median], zorder=5,
               label=f"f(sim_median)=target ({sim_median:.1f}→{target_median:.1f})")
    ax.axvline(sim_median, color="gray", linewidth=0.8)
    ax.axhline(target_median, color="gray", linewidth=0.8)
    ax.axhline(clip, color="grey", label=f"p{CLIP_PERCENTILE} clip ({clip:,.0f})")
    ax.set(xscale="log", yscale="log", xlabel="Input ppl/km²", ylabel="Output ppl/km²")
    ax.set_title("Value-transform tone curves")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/augmentation/tone_curves.png")
    plt.show()


# --- Entry point ------------------------------------------------------------
def main() -> None:
    dataset = rasterio.open(MAP_PATH)
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    sim = read_full_dataset(dataset)
    eddf = read_box(dataset, *latlon_box_to_native(transformer, *EDDF, half_km=SIM_HALF_KM))

    clip = np.percentile(sim, CLIP_PERCENTILE)
    sim_median, eddf_median = np.median(sim), np.median(eddf)

    sim_mean_clipped = np.mean(np.clip(sim, None, clip))
    print(f"sim mean (clipped)  {sim_mean_clipped:.1f}")
    print(f"eddf mean (clipped) {np.mean(np.clip(eddf, None, clip)):.1f}")

    # Calibrate each transform against the EDDF reference.
    gamma_low, gamma_high = calibrate_gamma(sim, eddf, CLIP_PERCENTILE)
    factor_far = calibrate_scale(sim, eddf)
    floor_far = calibrate_floor(sim, eddf)

    print(f"\nγ range with inflation in {INFLATION_BAND}×: [{gamma_low:.3f}, {gamma_high:.3f}]")
    gamma = GammaCorrection(percentile=50, gamma=(gamma_low, gamma_high), target_value=eddf_median)
    gamma.resolve(sim_median)

    print(f"Scaling within the range [1.0, {factor_far:.2f}]")
    scale = ScaleValues(factor=(1.0, factor_far))   # low end = identity (no-op)
    print(f"Fixed Floor within the range [0.0, {floor_far:.2f}]")
    floor = FloorRaise(floor=(0.0, floor_far))       # low end = identity (no-op)

    # name -> (transform, color), shared by both plots.
    transforms = {
        "gamma correction": (gamma, "C2"),
        "floor raise": (floor, "C3"),
        "scale values": (scale, "C4"),
    }

    transformed = {name: apply_extreme(t, sim, clip) for name, (t, _) in transforms.items()}
    for name, vals in transformed.items():
        transformed_mean_clipped = np.mean(np.clip(vals, None, clip))
        print(f"transformed mean (clipped) ({name}): {transformed_mean_clipped:.1f}, reward inflation {transformed_mean_clipped / sim_mean_clipped}")

    plot_cdf(
        [("Sim (training field)", sim, "C0"), ("EDDF box", eddf, "C1")]
        + [(f"Sim + {name}", transformed[name], color) for name, (_, color) in transforms.items()],
        clip=clip,
    )
    plot_tone_curves(list(transforms.values()), sim_median, eddf_median, clip)


if __name__ == "__main__":
    main()
