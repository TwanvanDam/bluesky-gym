import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scripts.common.colors import MODE_COLORS

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

def draw_aircraft(ax, x_km, y_km, size_km=3, zorder=8):
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
    color  = MODE_COLORS[mode]

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
    color = MODE_COLORS[mode]
    ann_y = bottom - 4.0
    ax.annotate("", xy=(left + OBS_KM, ann_y), xytext=(left, ann_y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.8), zorder=6)
    ax.text(left + OBS_KM / 2, ann_y - 1.5,
            f"{int(OBS_KM)} km  ({SHAPE_PX[0]} px)",
            ha="center", va="top", fontsize=11, color=color, zorder=6)

    ann_x = left + OBS_KM + 3.0
    ax.annotate("", xy=(ann_x, bottom + OBS_KM), xytext=(ann_x, bottom),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.8), zorder=6)
    ax.text(ann_x + 1.0, bottom + OBS_KM / 2,
            f"{int(OBS_KM)} km\n({SHAPE_PX[1]} px)",
            ha="left", va="center", fontsize=11, color=color, zorder=6)

def plot_panel(ax, mode: str):
    ac_x, ac_y = 0.0, 0.0
    left, bottom = draw_obs_window(ax, ac_x, ac_y, mode)
    draw_aircraft(ax, ac_x, ac_y)
    add_dim_annotations(ax, left, bottom, mode)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if mode == "centered":
        ax.set_xlim(-42, 54)
        ax.set_ylim(-46, 42)
    else:  # forward: window spans ac_y=0 to OBS_KM=64
        ax.set_xlim(-42, 54)
        ax.set_ylim(-14, 72)


def save_panel(mode: str, out_path: str):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    plot_panel(ax, mode)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    save_panel("centered", "plots/population_maps/observation_modes_centered.pdf")
    save_panel("forward",  "plots/population_maps/observation_modes_forward.pdf")

