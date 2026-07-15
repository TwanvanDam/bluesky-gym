"""
Visualize forward vs centered observation window modes (schematic, no background map).

Shows how the 16×16 px (1 km/pixel) observation grid is positioned
relative to the aircraft for both 'centered' and 'forward' modes.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from scripts.common.colors import *

PLOTS_DIR = Path(__file__).parent.parent / "plots"

# Observation parameters (from centered_1.yaml / forward_1.yaml)
SHAPE_PX = (16, 16)          # (cols, rows)
RANGE_M = (64_000, 64_000)   # meters
KM_PER_PX = RANGE_M[0] / SHAPE_PX[0] / 1_000   # 4.0 km/pixel
OBS_KM = SHAPE_PX[0] * KM_PER_PX               # 16 km

# Context area (km) around aircraft shown in each panel
CTX_KM = 80.0

AC_COLOR = "#333333"
BG_COLOR = "white"
PANEL_BG = "white"
TEXT_COLOR = "#222222"


def draw_aircraft(ax, x_km, y_km, size_km=5, zorder=8):
    nose  = np.array([x_km,              y_km + size_km * 0.7])
    left  = np.array([x_km - size_km * 0.45, y_km - size_km * 0.45])
    right = np.array([x_km + size_km * 0.45, y_km - size_km * 0.45])
    tail  = np.array([x_km,              y_km - size_km * 0.18])
    poly = plt.Polygon([nose, left, tail, right], closed=True,
                       facecolor=AC_COLOR, edgecolor="#555", linewidth=0.8, zorder=zorder)
    ax.add_patch(poly)


def draw_obs_window(ax, ac_x, ac_y, mode: str):
    """Draw pixel grid and bounding box of the observation window."""
    half_w = OBS_KM / 2
    left   = ac_x - half_w
    bottom = ac_y if mode == "forward" else ac_y - OBS_KM / 2
    color  = FORWARD_COLOR if mode == "forward" else CENTERED_COLOR

    # Pixel fill (subtle tint)
    ax.add_patch(mpatches.Rectangle(
        (left, bottom), OBS_KM, OBS_KM,
        facecolor=color, alpha=0.08, linewidth=0, zorder=2))

    # Pixel grid lines
    for i in range(SHAPE_PX[0] + 1):
        xg = left + i * KM_PER_PX
        ax.plot([xg, xg], [bottom, bottom + OBS_KM],
                color=color, lw=0.5, alpha=0.6, zorder=3)
    for j in range(SHAPE_PX[1] + 1):
        yg = bottom + j * KM_PER_PX
        ax.plot([left, left + OBS_KM], [yg, yg],
                color=color, lw=0.5, alpha=0.6, zorder=3)

    # Bold border
    ax.add_patch(mpatches.Rectangle(
        (left, bottom), OBS_KM, OBS_KM,
        linewidth=2.2, edgecolor=color, facecolor="none", zorder=5))

    return left, bottom


def add_dim_annotations(ax, left, bottom, mode):
    """Width and height dimension lines."""
    color  = FORWARD_COLOR if mode == "forward" else CENTERED_COLOR
    ann_y = bottom - 5
    ax.annotate("", xy=(left + OBS_KM, ann_y), xytext=(left, ann_y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1), zorder=6)
    ax.text(left + OBS_KM / 2, ann_y - 2,
            f"{int(OBS_KM)} km  ({SHAPE_PX[0]} px)",
            ha="center", va="top", color=color, zorder=6)

    ann_x = left + OBS_KM + 5
    ax.annotate("", xy=(ann_x, bottom + OBS_KM), xytext=(ann_x, bottom),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1), zorder=6)
    ax.text(ann_x + 2, bottom + OBS_KM / 2,
            f"{int(OBS_KM)} km\n({SHAPE_PX[1]} px)",
            ha="left", va="center", color=color, zorder=6)


def plot_panel(ax, mode: str):
    ac_x, ac_y = 0.0, 0.0

    left, bottom = draw_obs_window(ax, ac_x, ac_y, mode)
    draw_aircraft(ax, ac_x, ac_y)
    add_dim_annotations(ax, left, bottom, mode)

    # Limits derived from the actual window position so both modes stay in
    # frame; span is mode-independent, so panels crop to identical size.
    ax.set_xlim(left - 6, left + OBS_KM + 16)
    ax.set_ylim(bottom - 8, bottom + OBS_KM + 6)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------

out_dir = PLOTS_DIR / "population_maps"

for mode, suffix in [
    ("centered", "centered"),
    ("forward",  "forward"),
]:
    size = 0.49 * 0.75 * TEXTWIDTH_IN
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor(BG_COLOR)

    plot_panel(ax, mode)
    fig.tight_layout(pad=1.5)

    out_path = out_dir / f"observation_mode_{suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")

plt.close()
