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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Transformer

from bluesky_gym.utils.sampling_config import ExclusionZone

MAP_DIR = Path(__file__).parent / "population_maps"
OUT_DIR = MAP_DIR

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


def wgs84_to_pixel(lat, lon, ds_crs, ds_transform, img_w, orig_w, img_h, orig_h):
    """Convert WGS84 lat/lon to (col, row) in the displayed (possibly downsampled) image."""
    transformer = Transformer.from_crs("EPSG:4326", ds_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    col_orig = (x - ds_transform.c) / ds_transform.a
    row_orig = (y - ds_transform.f) / ds_transform.e
    return col_orig * img_w / orig_w, row_orig * img_h / orig_h


def km_to_pixel_radius(lat, lon, radius_km, ds_crs, ds_transform, img_w, orig_w):
    """Convert radius_km to pixel radius in the displayed image via eastward offset."""
    lon_offset = lon + radius_km / (111.0 * np.cos(np.deg2rad(lat)))
    transformer = Transformer.from_crs("EPSG:4326", ds_crs, always_xy=True)
    x0, _ = transformer.transform(lon, lat)
    x1, _ = transformer.transform(lon_offset, lat)
    r_orig = abs(x1 - x0) / abs(ds_transform.a)
    return r_orig * img_w / orig_w


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

        # Plot
        display = np.where(np.isfinite(data) & (data > 0), np.log1p(data), np.nan)
        img_h, img_w = display.shape
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(display, cmap="Blues", origin="upper", aspect="auto")
        plt.colorbar(im, ax=ax, label="log(1 + population density)")

        if exclusion_zones:
            for zone in exclusion_zones:
                col, row = wgs84_to_pixel(
                    zone.lat, zone.lon, ds.crs, ds.transform,
                    img_w, ds.width, img_h, ds.height,
                )
                r_px = km_to_pixel_radius(
                    zone.lat, zone.lon, zone.radius_km, ds.crs, ds.transform,
                    img_w, ds.width,
                )
                circle = mpatches.Circle(
                    (col, row), r_px,
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
