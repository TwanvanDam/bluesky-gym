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
# Default is a large inspection figure. Pass --figsize/--dpi to build one at the
# size of its LaTeX slot instead: the in-figure text is set in points, so a
# figure that gets rescaled by \includegraphics has its labels rescaled with it,
# and a 10 in figure dropped into a 5 cm slot leaves them illegible.
FIGSIZE = (10, 8)
DPI = 150

SCALE_BAR_KM = 500
# Annotations sit below the 9 pt body size: this is a locator map, so the
# exclusion label and the scale bar should be readable without competing with the
# caption under them.
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


def draw_exclusion_zone(ax, x: float, y: float, r: float, radius_km: float,
                        label: bool = False) -> None:
    """Mark a zone as *removed from* the sampling domain rather than as a highlight.

    A saturated filled disc reads as a region of interest. The fill here is the
    same grey the colormap uses for no-data, so grey means "outside the sampling
    domain" everywhere in the figure; the hatching and dashed edge then say the
    exclusion is a rule, not missing data. Fill and hatch are separate patches
    because ``alpha`` applies to a patch as a whole, edge and hatching included.
    """
    ax.add_patch(mpatches.Circle(
        (x, y), r, facecolor=BACKGROUND_COLOR, edgecolor="none", alpha=0.8, zorder=3,
    ))
    ax.add_patch(mpatches.Circle(
        (x, y), r, facecolor="none", edgecolor="darkred", hatch="////",
        linewidth=ANNOTATION_LW, linestyle=(0, (5, 3)), zorder=4,
    ))
    if label:
        # Placed up and to the left, which is open sea on the European map and so
        # does not cover any density the figure is meant to show. Off by default:
        # the caption carries the explanation for the figure as used in the paper.
        ax.annotate(
            f"No airports\n({radius_km:g} km)",
            xy=(x - r * 0.71, y + r * 0.71), xytext=(x - r * 2.4, y + r * 2.1),
            color="darkred", ha="center", va="center", zorder=5, fontsize=ANNOTATION_PT,
            arrowprops=dict(arrowstyle="->", color="darkred", linewidth=ANNOTATION_LW, shrinkB=0),
        )


def draw_scale_bar(ax, length_km: float = SCALE_BAR_KM, x0: float = 0.04, y0: float = 0.04) -> None:
    """Scale bar in the lower-left, sized by the true ground distance there.

    ``PLOT_CRS`` is Mollweide, whose linear scale over Europe runs from about
    0.99 to 1.18 east-west: a bar of ``length_km`` nominal projected metres would
    be wrong by up to a fifth. The bar is therefore drawn as the projected length
    of a real ``length_km`` geodesic starting at the bar's own position, so it is
    exact where it sits and only drifts away from it.
    """
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    x_start = x_left + x0 * (x_right - x_left)
    y = y_bottom + y0 * (y_top - y_bottom)

    to_wgs84 = Transformer.from_crs(PLOT_CRS, WGS84_CRS, always_xy=True)
    to_plot_crs = Transformer.from_crs(WGS84_CRS, PLOT_CRS, always_xy=True)
    lon, lat = to_wgs84.transform(x_start, y)
    lon_end, lat_end, _ = Geod(ellps="WGS84").fwd(lon, lat, 90.0, length_km * 1000.0)
    x_end, _ = to_plot_crs.transform(lon_end, lat_end)

    ax.plot([x_start, x_end], [y, y],
            color="black", linewidth=1.4, solid_capstyle="butt", zorder=6)
    ax.text((x_start + x_end) / 2, y + 0.012 * (y_top - y_bottom), f"{length_km:g} km",
            ha="center", va="bottom", color="black", zorder=6, fontsize=ANNOTATION_PT)


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

        # Only no-data pixels (NaN after reprojection) are left masked; genuine
        # zero-population land stays a real value so it draws with the colormap.
        # Masked pixels render grey, matching the Population wrapper's grey
        # background fill behind transparent no-data areas.
        finite = np.isfinite(proj)
        display = np.where(finite, np.log1p(np.clip(proj, 0, None)), np.nan)
        left, bottom, right, top = array_bounds(dst_h, dst_w, dst_transform)
        cmap = matplotlib.colormaps[PLOT_CMAP].copy()
        cmap.set_bad(color=BACKGROUND_COLOR)
        fig, ax = plt.subplots(figsize=figsize)
        # Axes fill the canvas, so ``figsize`` is the size of the saved map and
        # not of a map plus margins. Combined with the zero-pad tight bbox below,
        # that makes --figsize the actual output size.
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(
            display, cmap=cmap, origin="upper",
            extent=(left, right, bottom, top), aspect="equal",
        )

        # Crop the axes to the data footprint so the surrounding no-data padding
        # (introduced when the source rectangle is warped to the display CRS) is
        # not shown as a grey border.
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
                # Project a true geodesic offset point rather than assuming a
                # fixed scale factor, so the radius is correct in whatever
                # PLOT_CRS is (currently LAEA, not Web Mercator).
                lon_edge, lat_edge, _ = geod.fwd(zone.lon, zone.lat, 90.0, zone.radius_km * 1000.0)
                x_edge, y_edge = to_plot_crs.transform(lon_edge, lat_edge)
                r = np.hypot(x_edge - x, y_edge - y)
                draw_exclusion_zone(ax, x, y, r, zone.radius_km, label=label_exclusions)

        if scale_bar:
            draw_scale_bar(ax)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = OUT_DIR / f"{tiff_path.stem}_coverage.png"
        # No axis decorations and a zero-pad tight bbox: the figure is included
        # at a fixed height in the paper, so any white margin saved here just
        # shrinks the map on the page. Was being cropped by hand before.
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
