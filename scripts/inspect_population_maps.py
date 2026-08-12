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
from rasterio.transform import array_bounds, xy
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, Resampling
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Geod, Transformer

from bluesky_gym.utils.sampling_config import ExclusionZone
from scripts.common.colors import BACKGROUND_COLOR

MAP_DIR = Path(__file__).parent / "population_maps"
OUT_DIR = Path("./plots/population_maps")

PLOT_CRS = "ESRI:54009"  # Mollweide equal-area
WGS84_CRS = "EPSG:4326"
PLOT_CMAP = "Blues"

FIGSIZE = (10, 8)
DPI = 150

ANNOTATION_PT = 6.0
ANNOTATION_LW = 0.8
# Thinner than the matplotlib default, so the exclusion hatching reads as texture
# rather than as a second solid fill.
matplotlib.rcParams["hatch.linewidth"] = 0.4

MAX_PIXELS = 4_000_000  # downsample files larger than this on read


def wgs84_bounds(dataset):
    """Return (lon_min, lat_min, lon_max, lat_max) in WGS84."""
    try:
        b = dataset.bounds
        return transform_bounds(dataset.crs, WGS84_CRS, b.left, b.bottom, b.right, b.top)
    except Exception:
        return None


def draw_exclusion_zone(ax, x: float, y: float, r: float) -> None:
    ax.add_patch(mpatches.Circle(
        (x, y), r, facecolor=BACKGROUND_COLOR, edgecolor="none", alpha=0.8, zorder=3,
    ))
    ax.add_patch(mpatches.Circle(
        (x, y), r, facecolor="none", edgecolor="darkred", hatch="////",
        linewidth=ANNOTATION_LW, linestyle=(0, (5, 3)), zorder=4,
    ))

def inspect_file(tiff_path: Path, exclusion_zones: list[ExclusionZone] | None = None,
                 figsize: tuple[float, float] = FIGSIZE, dpi: int = DPI,
                 scale_bar: bool = False, label_exclusions: bool = False) -> dict:
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
            "min": float(valid.min()) if valid.size else float("nan"),
            "max": float(valid.max()) if valid.size else float("nan"),
            "mean": float(valid.mean()) if valid.size else float("nan"),
            "nonzero_frac": float(np.sum(valid > 0) / valid.size) if valid.size else 0.0,
        }
        info["bounds_wgs84"] = wgs84_bounds(ds)

        # Reproject the (downsampled) array to the display CRS
        src_transform = ds.transform * rasterio.Affine.scale(
            ds.width / data.shape[1], ds.height / data.shape[0]
        )
        dst_transform, dst_w, dst_h = calculate_default_transform(
            ds.crs, PLOT_CRS, data.shape[1], data.shape[0], *ds.bounds
        )
        proj = np.full((dst_h, dst_w), np.nan, dtype=np.float64)
        reproject(
            source=data, destination=proj,
            src_transform=src_transform, src_crs=ds.crs,
            dst_transform=dst_transform, dst_crs=PLOT_CRS,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.average,
        )

        # determine pixels that are not nodata
        finite = np.isfinite(proj)
        display = np.where(finite, np.log1p(np.clip(proj, 0, None)), np.nan)
        left, bottom, right, top = array_bounds(dst_h, dst_w, dst_transform)
        cmap = matplotlib.colormaps[PLOT_CMAP].copy()

        # make nodata values grey
        cmap.set_bad(color=BACKGROUND_COLOR)
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(
            display, cmap=cmap, origin="upper",
            extent=(left, right, bottom, top), aspect="equal",
        )

        # Crop the axes to the data footprint
        if finite.any():
            cols = np.where(finite.any(axis=0))[0]
            rows = np.where(finite.any(axis=1))[0]
            x_left, _ = xy(dst_transform, 0, cols[0], offset="ul")
            x_right, _ = xy(dst_transform, 0, cols[-1], offset="ur")
            _, y_top = xy(dst_transform, rows[0], 0, offset="ul")
            _, y_bottom = xy(dst_transform, rows[-1], 0, offset="ll")
            ax.set_xlim(x_left, x_right)
            ax.set_ylim(y_bottom, y_top)

        if exclusion_zones:
            to_plot_crs = Transformer.from_crs(WGS84_CRS, PLOT_CRS, always_xy=True)
            geod = Geod(ellps="WGS84")
            for zone in exclusion_zones:
                x, y = to_plot_crs.transform(zone.lon, zone.lat)
                lon_edge, lat_edge, _ = geod.fwd(zone.lon, zone.lat, 90.0, zone.radius_km * 1000.0)
                x_edge, y_edge = to_plot_crs.transform(lon_edge, lat_edge)
                r = np.hypot(x_edge - x, y_edge - y)
                draw_exclusion_zone(ax, x, y, r, zone.radius_km, label=label_exclusions)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = OUT_DIR / f"{tiff_path.stem}_coverage.png"
        ax.set_axis_off()
        fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"  saved plot → {plot_path.name}")

    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--exclusion", metavar=("LAT", "LON", "RADIUS_KM"),
        nargs=3, action="append", type=float, default=[],
        help="Exclusion zone to mark as excluded (lat lon radius_km). Repeatable.",
    )
    parser.add_argument(
        "--figsize", metavar=("WIDTH_IN", "HEIGHT_IN"), nargs=2, type=float, default=FIGSIZE,
        help="Figure size in inches. Set this to the LaTeX slot for a figure that goes in the paper.",
    )
    parser.add_argument("--dpi", type=int, default=DPI, help="Raster resolution of the saved PNG.")
    parser.add_argument("--scale_bar", action="store_true", help="Draw a scale bar in the lower left.")
    parser.add_argument("--label_exclusions", action="store_true",
                        help="Annotate each exclusion zone in the figure instead of leaving it to the caption.")
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
        info = inspect_file(path, exclusion_zones=exclusion_zones or None,
                            figsize=tuple(args.figsize), dpi=args.dpi,
                            scale_bar=args.scale_bar, label_exclusions=args.label_exclusions)
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
