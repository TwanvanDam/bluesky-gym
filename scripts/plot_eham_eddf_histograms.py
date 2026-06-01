"""Histograms of population density around EHAM, EDDF, and the full simulation environment.

Bounds taken from a resolution_sweep_2 config:
- Per-episode sim bounds: ±400 km around destination (800 km square) — config value is half-extent
  (see base_navigation_env.py:187-190).
- Training destination box: lat [35, 55], lon [-5, 45].
- Population source: scripts/population_maps/europe_3035_1km.tif (EPSG:3035, 1 km/px).
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import matplotlib.pyplot as plt

from scripts.common.colors import REGION_COLORS

TIF_PATH = Path(__file__).parent / "population_maps" / "europe_3035_1km.tif"
OUT_PATH = Path(__file__).parent / "population_maps" / "histograms_eham_eddf_sim.png"

SIM_HALF_KM = 400.0  # half-extent in km; full box is 2 * SIM_HALF_KM on a side
EHAM = (52.308, 4.764)
EDDF = (50.0379, 8.5622)
DEST_BOX_LAT = (35.0, 55.0)
DEST_BOX_LON = (-5.0, 45.0)


def latlon_box_to_native(transformer, lat_c, lon_c, half_km):
    """Approximate a square box around a lat/lon point in the source CRS (meters)."""
    x_c, y_c = transformer.transform(lon_c, lat_c)
    half_m = half_km * 1000.0
    return x_c - half_m, y_c - half_m, x_c + half_m, y_c + half_m


def latlon_corners_to_native_bbox(transformer, lat_lo, lat_hi, lon_lo, lon_hi):
    """Project the WGS84 box corners and take the axis-aligned bounding box in source CRS."""
    lats = [lat_lo, lat_lo, lat_hi, lat_hi]
    lons = [lon_lo, lon_hi, lon_lo, lon_hi]
    xs, ys = transformer.transform(lons, lats)
    return min(xs), min(ys), max(xs), max(ys)


def read_box(src, left, bottom, right, top):
    """Read the raster window covering (left, bottom, right, top) in source CRS."""
    win = from_bounds(left, bottom, right, top, transform=src.transform)
    win = win.round_offsets().round_lengths()
    data = src.read(1, window=win).astype(np.float64)
    nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    valid = data[np.isfinite(data) & (data >= 0)]
    return valid


def stats_block(name, vals):
    p = np.percentile(vals, [50, 75, 90, 99, 99.9])
    return (
        f"{name}\n"
        f"  n pixels       : {len(vals):>10,}\n"
        f"  mean           : {vals.mean():>10.1f}\n"
        f"  median         : {p[0]:>10.1f}\n"
        f"  75th pct       : {p[1]:>10.1f}\n"
        f"  90th pct       : {p[2]:>10.1f}\n"
        f"  99th pct       : {p[3]:>10.1f}\n"
        f"  99.9th pct     : {p[4]:>10.1f}\n"
        f"  max            : {vals.max():>10.1f}\n"
        f"  frac zero/empty: {(vals == 0).mean():>10.1%}\n"
    )


def main():
    with rasterio.open(TIF_PATH) as src:
        print(f"Source: {TIF_PATH.name}  CRS={src.crs}  res={src.res}")
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        eham_bbox = latlon_box_to_native(transformer, *EHAM, half_km=SIM_HALF_KM)
        eddf_bbox = latlon_box_to_native(transformer, *EDDF, half_km=SIM_HALF_KM)
        sim_bbox = latlon_corners_to_native_bbox(
            transformer, DEST_BOX_LAT[0], DEST_BOX_LAT[1], DEST_BOX_LON[0], DEST_BOX_LON[1]
        )

        eham_vals = read_box(src, *eham_bbox)
        eddf_vals = read_box(src, *eddf_bbox)
        sim_vals = read_box(src, *sim_bbox)

    regions = [
        ("EHAM (52.31°N, 4.76°E)", eham_vals, REGION_COLORS[0]),
        ("EDDF (50.04°N, 8.56°E)", eddf_vals, REGION_COLORS[1]),
        ("Sim env (training destination box)", sim_vals, REGION_COLORS[2]),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    side_km = 2 * SIM_HALF_KM
    fig.suptitle(
        "Population density distribution — fraction of pixels at or below value\n"
        f"Airport boxes: ±{SIM_HALF_KM:.0f} km ({side_km:.0f}×{side_km:.0f} km, matching simulation_bounds_size) · "
        f"sim env = destination box lat[{DEST_BOX_LAT[0]:.0f},{DEST_BOX_LAT[1]:.0f}] × lon[{DEST_BOX_LON[0]:.0f},{DEST_BOX_LON[1]:.0f}]",
        fontsize=11,
        fontweight="bold",
    )

    # ECDF on linear axes — read percentiles straight off the y-axis.
    # x clipped to the 90th percentile of the densest region so the bulk is legible.
    x_max = max(eham_vals)
    for name, vals, color in regions:
        sorted_v = np.sort(vals)
        ecdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        med = np.median(vals)
        p75 = np.percentile(vals, 75)
        frac_above_xmax = (vals > x_max).mean()
        label = (
            f"{name}\n"
            f"   median={med:,.0f}   p75={p75:,.0f}   "
            f"frac > {x_max:,.0f} = {frac_above_xmax:.0%}   n={len(vals):,}"
        )
        ax.plot(sorted_v, ecdf, color=color, linewidth=2.2, label=label)
        # Mark median on the curve.
        ax.plot([med], [0.5], "o", color=color, markersize=7, markeredgecolor="k", markeredgewidth=0.8)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(x_max * 0.985, 0.51, "median (50%)", ha="right", va="bottom", fontsize=9, color="gray")
    ax.axhline(0.75, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(x_max * 0.985, 0.76, "75th percentile", ha="right", va="bottom", fontsize=9, color="gray")

    ax.set_xlim(1, x_max)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Population per km²")
    ax.set_ylabel("Fraction of pixels ≤ value  (empirical CDF)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.semilogx()
    # Short interpretive caption rather than a verbose stats table.
    ratio = float(np.median(eham_vals) / max(np.median(sim_vals), 1e-9))
    caption = (
        f"How to read this: a curve passing through (x={int(np.median(eham_vals))}, y=0.5) means half the pixels in that region are\n"
        f"below {int(np.median(eham_vals))} ppl/km². Curves shifted RIGHT are denser. "
        f"EHAM's median is ~{ratio:.0f}× the sim-environment median —\n"
        f"the agent rarely trains on terrain as dense as the area it must navigate at EHAM."
    )
    fig.text(
        0.5, -0.02, caption, ha="center", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )

    plt.tight_layout(rect=[0, 0.0, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT_PATH}")

    print("\nQuick summary (median ppl/km²):")
    for name, vals, _ in regions:
        print(f"  {name:50s}  median={np.median(vals):>7.1f}  mean={vals.mean():>7.1f}  n={len(vals):,}")


if __name__ == "__main__":
    main()
