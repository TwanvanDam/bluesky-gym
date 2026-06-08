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
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[0, 1])

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
    ax.plot(xs, ys, color=BLUE, lw=2.2, zorder=2, label="central meridian (10°E)\ngrid-N = true-N here")

    sites = [("Schiphol", 52.31, 4.76), ("Frankfurt", 50.11, 8.68), ("Crimea", 45.0, 34.0)]
    for name, lat, lon in sites:
        x, y = WGS84_TO_DEST.transform(lon, lat)
        gamma = SAMPLER._meridian_convergence(lon, lat)
        L = 850_000
        tn = true_north_vec_dest(lat, lon, L)
        # grid north is straight up in dest CRS
        ax.annotate("", xy=(x, y + L), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0), zorder=4)
        ax.annotate("", xy=(x + tn[0], y + tn[1]), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0), zorder=4)
        ax.scatter([x], [y], s=45, color="black", zorder=5)
        ax.text(x + 90_000, y + 60_000, f"{name}\nγ = {gamma:+.0f}°",
                fontsize=10, fontweight="bold", zorder=6)

    ax.plot([], [], color=RED, lw=2.0, label="grid-north (straight up on flat map)")
    ax.plot([], [], color=BLUE, lw=2.0, label="true-north (toward the pole)")
    ax.set_title("EPSG:3035 graticule — meridians fan out\n"
                 "grid-north ≠ true-north away from 10°E", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    # --- compass inset: the twist at Crimea ---
    lat, lon, hdg = 45.0, 34.0, 50.0
    gamma = SAMPLER._meridian_convergence(lon, lat)
    tn = true_north_vec_dest(lat, lon, 1.0)
    hd_true = heading_vec_dest(lat, lon, hdg, 1.0)
    # grid-space version of the heading (what the buggy code sampled): rotate heading by +gamma
    # in grid -> equivalently the grid 'up' the old window used. Show its direction.
    theta_grid_wrong = np.radians(hdg)  # old code treated heading as a grid angle from grid-N(+y)
    hd_wrong = np.array([np.sin(theta_grid_wrong), np.cos(theta_grid_wrong)])

    axc.annotate("", xy=(0, 1), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4))
    axc.text(0.03, 1.02, "grid-north", color=RED, fontsize=10, fontweight="bold")
    axc.annotate("", xy=(tn[0], tn[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.4))
    axc.text(tn[0] - 0.62, tn[1] + 0.02, "true-north", color=BLUE, fontsize=10, fontweight="bold")

    # wedge for gamma between grid-N and true-N
    ang_grid = 90.0
    ang_true = np.degrees(np.arctan2(tn[1], tn[0]))
    axc.add_patch(Wedge((0, 0), 0.32, min(ang_grid, ang_true), max(ang_grid, ang_true),
                        facecolor=GREY, alpha=0.4))
    axc.text(0.16, 0.40, f"γ = {gamma:.0f}°", fontsize=11, fontweight="bold")

    # heading: where the plane actually flies (true) vs where the old window pointed (grid)
    axc.annotate("", xy=(hd_true[0] * 0.82, hd_true[1] * 0.82), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4))
    axc.text(hd_true[0] * 0.86, hd_true[1] * 0.86, "flight\nheading", color=GREEN, fontsize=9,
             fontweight="bold", ha="left")
    axc.annotate("", xy=(hd_wrong[0] * 0.82, hd_wrong[1] * 0.82), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0, ls="--"))
    axc.text(hd_wrong[0] * 0.86, hd_wrong[1] * 0.70, "where the\nwindow looked\n(old code)",
             color=RED, fontsize=9, ha="left")

    axc.set_xlim(-1.15, 1.15); axc.set_ylim(-0.25, 1.2)
    axc.set_aspect("equal")
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_title("Compass at Crimea (γ = 18°)\nthe old window was aimed γ off the heading",
                  fontsize=12, fontweight="bold")

    fig.suptitle("CAUSE — a flat map of a round Earth is twisted; the twist (γ) grows toward the edges",
                 fontsize=13, fontweight="bold", y=1.0)
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
    fig, (axw, axe) = plt.subplots(1, 2, figsize=(13, 6.0))

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
    axw.set_xlabel("metres east of aircraft (grid)")
    axw.set_ylabel("metres north of aircraft (grid)")
    axw.set_title(f"EFFECT — the 'look-ahead' window at Crimea\n"
                  f"old code points it {gamma:.0f}° off the real flight path",
                  fontsize=12, fontweight="bold")
    axw.legend(loc="lower center", fontsize=9, framealpha=0.95)
    axw.grid(alpha=0.25)

    # --- alignment error vs longitude, before vs after ---
    lon_sweep = np.linspace(-15, 35, 120)
    lat_ref = 50.0
    err_before, err_after = [], []
    for lo in lon_sweep:
        g = SAMPLER._meridian_convergence(lo, lat_ref)
        # use heading 90 (E-W) where the non-conformal residual is largest, worst-case after
        def measured(orient):
            t = SAMPLER._get_dst_transform_from_center(pos(lat_ref, lo), orient, OBS)
            cols, rows = OBS.shape
            cx, cy = t * (cols / 2, rows / 2)
            fx, fy = t * (cols / 2, rows / 2 - 1)
            clon, clat = DEST_TO_WGS84.transform(cx, cy)
            flon, flat = DEST_TO_WGS84.transform(fx, fy)
            return get_hdg(np.array([clat, clon]), np.array([flat, flon]))
        def aerr(a, b):
            return abs((a - b + 180) % 360 - 180)
        err_before.append(aerr(measured(90.0 + g), 90.0))
        err_after.append(aerr(measured(90.0), 90.0))

    axe.plot(lon_sweep, err_before, color=RED, lw=2.4, label="before fix")
    axe.plot(lon_sweep, err_after, color=GREEN, lw=2.4, label="after fix")
    axe.axvline(10.0, color=BLUE, ls="--", lw=1.2)
    axe.text(10.3, axe.get_ylim()[1] * 0.92 if False else 17, "10°E\n(central)", color=BLUE, fontsize=8)
    for name, lo in [("Frankfurt", 8.68), ("Schiphol", 4.76)]:
        g = SAMPLER._meridian_convergence(lo, lat_ref)
        axe.scatter([lo], [abs(g)], color="black", zorder=5, s=30)
        axe.annotate(f"{name}\n(eval site)", (lo, abs(g)), textcoords="offset points",
                     xytext=(-2, 8), fontsize=8, ha="right")
    axe.set_xlabel("longitude (°E)  —  training spanned all of Europe")
    axe.set_ylabel("window mis-alignment (degrees)")
    axe.set_title("Mis-alignment vs longitude\nbug is ~0 in the centre, large at the edges",
                  fontsize=12, fontweight="bold")
    axe.legend(loc="upper center", fontsize=9)
    axe.grid(alpha=0.25)

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
