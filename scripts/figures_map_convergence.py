"""Generate a visual explainer of the map-convergence bug and its fix.

Produces two PNGs:
  1. convergence_cause.png  - WHY it happens: the EPSG:3035 graticule fans out, so
     "grid-north" (straight up on the flat map) is not "true-north" (toward the pole).
     A compass inset quantifies the twist gamma at three sites.
  2. convergence_effect.png - WHAT it does: the sampled observation window before vs
     after the fix at a high-convergence site, plus the before/after alignment error
     across all European longitudes with the eval sites marked.

No BlueSky sim or population raster is needed; everything is pure projection geometry.

    python -m scripts.figures_map_convergence [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyproj
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyArrow, Wedge

from bluesky_gym.maps.raster_sampler import RasterSampler, MapObservationConfig
from bluesky_gym.envs.common.functions import get_point_at_distance, get_hdg

DEST_CRS = "EPSG:3035"
OBS = MapObservationConfig(shape=(128, 128), range=(120_000.0, 120_000.0), position="forward")

RED = "#d1495b"     # before / wrong
GREEN = "#2a9d8f"   # after / corrected
BLUE = "#264653"    # true north / heading
GREY = "#8d99ae"

SAMPLER = RasterSampler(map_source=None, resampling="cubic_spline", destination_crs=DEST_CRS)
WGS84_TO_DEST = SAMPLER.wgs84_to_dest
DEST_TO_WGS84 = pyproj.Transformer.from_crs(DEST_CRS, "wgs84", always_xy=True)


def pos(lat, lon):
    return SimpleNamespace(lat=lat, lon=lon, alt=0.0)


def true_north_vec_dest(lat, lon, length_m=1.0):
    """Unit-ish vector pointing to true north, expressed in dest-CRS meters at (lat, lon)."""
    x0, y0 = WGS84_TO_DEST.transform(lon, lat)
    x1, y1 = WGS84_TO_DEST.transform(lon, lat + 1e-3)
    v = np.array([x1 - x0, y1 - y0])
    return v / np.linalg.norm(v) * length_m


def heading_vec_dest(lat, lon, hdg_deg, length_m=1.0):
    """Vector along the true compass heading, in dest-CRS meters at (lat, lon)."""
    lat2, lon2 = get_point_at_distance(lat, lon, 1.0, hdg_deg)
    x0, y0 = WGS84_TO_DEST.transform(lon, lat)
    x1, y1 = WGS84_TO_DEST.transform(lon2, lat2)
    v = np.array([x1 - x0, y1 - y0])
    return v / np.linalg.norm(v) * length_m


def window_corners(lat, lon, hdg, corrected: bool):
    """Corners of the sampled window in dest-CRS meters. corrected=False reproduces the
    old behaviour by cancelling the fix's gamma subtraction."""
    gamma = SAMPLER._meridian_convergence(lon, lat)
    orientation = hdg if corrected else hdg + gamma
    return SAMPLER.get_view_corners(pos(lat, lon), orientation, OBS)


# --------------------------------------------------------------------------------------
# Figure 1: the cause
# --------------------------------------------------------------------------------------
def fig_cause(out: Path):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # --- graticule of Europe in EPSG:3035 ---
    lons = np.arange(-20, 41, 5)
    lats = np.arange(34, 67, 4)
    lat_dense = np.linspace(34, 66, 100)
    lon_dense = np.linspace(-20, 40, 100)
    for lon in lons:
        xs, ys = WGS84_TO_DEST.transform(np.full_like(lat_dense, lon), lat_dense)
        ax.plot(xs, ys, color=GREY, lw=1.0, zorder=1)
    for lat in lats:
        xs, ys = WGS84_TO_DEST.transform(lon_dense, np.full_like(lon_dense, lat))
        ax.plot(xs, ys, color=GREY, lw=0.8, ls=":", zorder=1)

    # central meridian highlighted
    xs, ys = WGS84_TO_DEST.transform(np.full_like(lat_dense, 10.0), lat_dense)
    ax.plot(xs, ys, color=BLUE, lw=2.2, zorder=2, label="central meridian (10°E) — grid-N = true-N here")

    # Per-site label offsets to avoid overlap (dx, dy in metres, text ha)
    site_offsets = {
        "Schiphol":  (-1_100_000,  80_000, "left"),
        "Frankfurt": (   100_000,  80_000, "left"),
        "Crimea":    (   100_000,  80_000, "left"),
    }
    sites = [("Schiphol", 52.31, 4.76), ("Frankfurt", 50.11, 8.68), ("Crimea", 45.0, 34.0)]
    for name, lat, lon in sites:
        x, y = WGS84_TO_DEST.transform(lon, lat)
        gamma = SAMPLER._meridian_convergence(lon, lat)
        L = 850_000
        tn = true_north_vec_dest(lat, lon, L)
        ax.annotate("", xy=(x, y + L), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0), zorder=4)
        ax.annotate("", xy=(x + tn[0], y + tn[1]), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0), zorder=4)
        ax.scatter([x], [y], s=45, color="black", zorder=5)
        dx, dy, ha = site_offsets[name]
        ax.text(x + dx, y + dy, f"{name}\nγ = {gamma:+.0f}°",
                fontsize=10, fontweight="bold", ha=ha, zorder=6)

    ax.plot([], [], color=RED, lw=2.0, label="grid-north (straight up on flat map)")
    ax.plot([], [], color=BLUE, lw=2.0, label="true-north (toward the pole)")
    ax.set_title("WGS84 graticule in EPSG:3035\ngrid-north ≠ true-north away from 10°E",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("EPSG:3035 (ETRS89-LAEA Europe, central meridian 10°E)",
                  fontsize=8.5, color=GREY)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Figure 2: the effect
# --------------------------------------------------------------------------------------
def draw_window(ax, lat, lon, hdg, corrected, color, label):
    corners = window_corners(lat, lon, hdg, corrected)
    x0, y0 = WGS84_TO_DEST.transform(lon, lat)
    pts = [(cx - x0, cy - y0) for cx, cy in corners]
    poly = MplPolygon(pts, closed=True, fill=False, edgecolor=color, lw=2.4, label=label, zorder=3)
    ax.add_patch(poly)
    # mark the 'forward / top edge' midpoint so the aiming is obvious
    top_mid = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
    ax.plot([0, top_mid[0]], [0, top_mid[1]], color=color, lw=1.3, ls=":", zorder=3)


def fig_effect(out: Path):
    fig, axw = plt.subplots(figsize=(6.5, 6.0))

    # --- window before vs after at Crimea ---
    lat, lon, hdg = 45.0, 34.0, 50.0
    gamma = SAMPLER._meridian_convergence(lon, lat)
    draw_window(axw, lat, lon, hdg, corrected=False, color=RED,
                label=f"window BEFORE (aimed {gamma:+.0f}° off)")
    draw_window(axw, lat, lon, hdg, corrected=True, color=GREEN,
                label="window AFTER (aligned to heading)")

    hd = heading_vec_dest(lat, lon, hdg, 95_000)
    axw.annotate("", xy=(hd[0], hd[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.6), zorder=5)
    axw.text(hd[0] * 1.02, hd[1] * 1.02, "true flight heading", color=BLUE,
             fontsize=10, fontweight="bold")
    axw.scatter([0], [0], s=60, color="black", zorder=6)
    axw.text(6000, -9000, "aircraft", fontsize=9)

    axw.set_aspect("equal")
    axw.set_xlabel("metres east of aircraft (EPSG:3035 grid)")
    axw.set_ylabel("metres north of aircraft (EPSG:3035 grid)")
    axw.set_title(f"Forward window at Crimea (34°E)\n"
                  f"old code aims it {abs(gamma):.0f}° off the flight path",
                  fontsize=12, fontweight="bold")
    axw.legend(loc="lower center", fontsize=9, framealpha=0.95)
    axw.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures") / "map_convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    f1 = fig_cause(out_dir / "convergence_cause.png")
    f2 = fig_effect(out_dir / "convergence_effect.png")
    print(f"wrote {f1}")
    print(f"wrote {f2}")


if __name__ == "__main__":
    main()
