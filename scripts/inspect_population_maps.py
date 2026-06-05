"""
Inspect all population map TIFFs in scripts/population_maps/.
- Saves one plot per file showing the covered area (log-scaled density)
- Writes a summary text file with CRS, bounds, resolution, and basic stats

Options:
  --exclusion LAT LON RADIUS_KM   Overlay an exclusion zone as a red shaded
                                   circle (may be repeated for multiple zones)
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
import rasterio.enums
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Transformer

from bluesky_gym.utils.sampling_config import ExclusionZone

MAP_DIR = Path(__file__).parent / "population_maps"
OUT_DIR = Path("./plots/population_maps")

MAX_PIXELS = 4_000_000  # downsample files larger than this on read


def wgs84_bounds(dataset):
    """Return (lon_min, lat_min, lon_max, lat_max) in WGS84."""
    try:
        transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
        b = dataset.bounds
        lon_min, lat_min = transformer.transform(b.left, b.bottom)
        lon_max, lat_max = transformer.transform(b.right, b.top)
        return lon_min, lat_min, lon_max, lat_max
    except Exception:
        return None


def wgs84_corners(dataset):
    """Return WGS84 (lon, lat) of each native-bounds corner.

    The four corners are transformed individually because a projected CRS
    (e.g. EPSG:3035) has curved edges in lon/lat, so they do not share
    common longitudes/latitudes.
    """
    try:
        transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
        b = dataset.bounds
        return {
            "top-left": transformer.transform(b.left, b.top),
            "top-right": transformer.transform(b.right, b.top),
            "bottom-left": transformer.transform(b.left, b.bottom),
            "bottom-right": transformer.transform(b.right, b.bottom),
        }
    except Exception:
        return None


def inspect_file(tiff_path: Path, exclusion_zones: list[ExclusionZone] | None = None) -> dict:
    with rasterio.open(tiff_path) as ds:
        # Downsample very large rasters to avoid OOM
        total_px = ds.width * ds.height
        if total_px > MAX_PIXELS:
            scale = (MAX_PIXELS / total_px) ** 0.5
            out_h = max(1, int(ds.height * scale))
            out_w = max(1, int(ds.width * scale))
            print(f"  downsampling {ds.width}×{ds.height} → {out_w}×{out_h}")
            data = ds.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.average).astype(np.float64)
        else:
            data = ds.read(1).astype(np.float64)
        nodata = ds.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        valid = data[np.isfinite(data) & (data >= 0)]
        info = {
            "file": tiff_path.name,
            "crs": str(ds.crs),
            "bounds_native": ds.bounds,
            "size_px": (ds.width, ds.height),
            "resolution": ds.res,
            "nodata": nodata,
            "min": float(np.nanmin(data)) if valid.size else float("nan"),
            "max": float(np.nanmax(data)) if valid.size else float("nan"),
            "mean": float(np.nanmean(data)) if valid.size else float("nan"),
            "nonzero_frac": float(np.sum(valid > 0) / valid.size) if valid.size else 0.0,
        }
        info["bounds_wgs84"] = wgs84_bounds(ds)

        # Reproject the (downsampled) array to Web Mercator for display
        src_transform = ds.transform * rasterio.Affine.scale(
            ds.width / data.shape[1], ds.height / data.shape[0]
        )
        dst_crs = "EPSG:3857"
        dst_transform, dst_w, dst_h = calculate_default_transform(
            ds.crs, dst_crs, data.shape[1], data.shape[0], *ds.bounds
        )
        merc = np.full((dst_h, dst_w), np.nan, dtype=np.float64)
        reproject(
            source=data, destination=merc,
            src_transform=src_transform, src_crs=ds.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.average,
        )

        # Plot (Web Mercator; axes annotated with a lon/lat graticule)
        display = np.where(np.isfinite(merc) & (merc > 0), np.log1p(merc), np.nan)
        left = dst_transform.c
        top = dst_transform.f
        right = left + dst_w * dst_transform.a
        bottom = top + dst_h * dst_transform.e
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            display, cmap="Blues", origin="upper",
            extent=(left, right, bottom, top), aspect="equal",
        )
        plt.colorbar(im, ax=ax, label="log(1 + population density)")

        # lon/lat graticule (Mercator x depends only on lon, y only on lat)
        to_merc = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        lon_ticks = list(range(-30, 60, 10))
        lat_ticks = list(range(30, 70, 10))
        ax.set_xticks([to_merc.transform(lon, 0)[0] for lon in lon_ticks])
        ax.set_xticklabels([f"{lon}°" for lon in lon_ticks])
        ax.set_yticks([to_merc.transform(0, lat)[1] for lat in lat_ticks])
        ax.set_yticklabels([f"{lat}°" for lat in lat_ticks])
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Annotate the WGS84 lon/lat of each native-bounds corner, placed at
        # the corner's true Mercator position (data edges are tilted here)
        corners = wgs84_corners(ds)
        if corners:
            align = {
                "top-left": ("left", "bottom"),
                "top-right": ("right", "bottom"),
                "bottom-left": ("left", "top"),
                "bottom-right": ("right", "top"),
            }
            for name, (lon, lat) in corners.items():
                x, y = to_merc.transform(lon, lat)
                ha, va = align[name]
                ax.annotate(
                    f"{lat:.2f}°N, {lon:.2f}°E",
                    xy=(x, y), xycoords="data",
                    ha=ha, va=va, fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.8),
                )

        if exclusion_zones:
            for zone in exclusion_zones:
                x, y = to_merc.transform(zone.lon, zone.lat)
                # Web Mercator inflates ground distance by 1/cos(lat)
                r = zone.radius_km * 1000.0 / np.cos(np.deg2rad(zone.lat))
                circle = mpatches.Circle(
                    (x, y), r,
                    facecolor="red", edgecolor="darkred",
                    alpha=0.35, linewidth=1.5,
                )
                ax.add_patch(circle)

        plot_path = OUT_DIR / f"{tiff_path.stem}_coverage.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  saved plot → {plot_path.name}")

    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--exclusion", metavar=("LAT", "LON", "RADIUS_KM"),
        nargs=3, action="append", type=float, default=[],
        help="Exclusion zone to shade red (lat lon radius_km). Repeatable.",
    )
    args = parser.parse_args()

    exclusion_zones = [
        ExclusionZone(lat=lat, lon=lon, radius_km=r)
        for lat, lon, r in args.exclusion
    ]

    tiffs = sorted(p for p in MAP_DIR.iterdir() if p.suffix.lower() in {".tif", ".tiff"})
    if not tiffs:
        print(f"No TIFF files found in {MAP_DIR}")
        return

    all_info = []
    for path in tiffs:
        print(f"\nInspecting {path.name} ...")
        info = inspect_file(path, exclusion_zones=exclusion_zones or None)
        all_info.append(info)

    summary_path = OUT_DIR / "map_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Population map summary\n")
        f.write("=" * 60 + "\n\n")
        for info in all_info:
            f.write(f"File:       {info['file']}\n")
            f.write(f"CRS:        {info['crs']}\n")
            b = info["bounds_native"]
            f.write(f"Bounds:     left={b.left:.2f}  right={b.right:.2f}  "
                    f"bottom={b.bottom:.2f}  top={b.top:.2f}  (native CRS)\n")
            wgs = info["bounds_wgs84"]
            if wgs:
                f.write(f"WGS84:      lon [{wgs[0]:.3f}, {wgs[2]:.3f}]  "
                        f"lat [{wgs[1]:.3f}, {wgs[3]:.3f}]\n")
            f.write(f"Size:       {info['size_px'][0]} x {info['size_px'][1]} px\n")
            f.write(f"Resolution: {info['resolution'][0]:.1f} x {info['resolution'][1]:.1f} (native units)\n")
            f.write(f"No-data:    {info['nodata']}\n")
            f.write(f"Value range: {info['min']:.2f} – {info['max']:.2f}\n")
            f.write(f"Mean value: {info['mean']:.2f}\n")
            f.write(f"Non-zero fraction: {info['nonzero_frac']:.1%}\n")
            f.write("\n")

    print(f"\nSummary written → {summary_path}")


if __name__ == "__main__":
    main()
