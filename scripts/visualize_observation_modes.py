"""
Visualize forward vs centered observation window modes (schematic, no background map).

Shows how the 16×16 px (4 km/pixel) observation grid is positioned
relative to the aircraft for both 'centered' and 'forward' modes.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from scripts.common.colors import *
from scripts.common.figures import H_STRIP, W_HALF, paper_axes, save

PLOTS_DIR = Path(__file__).parent.parent / "plots"

# The panels carry no ticks or axis labels, so they need only enough margin to
# keep the dimension arrows and the unclipped "64 km" callout off the canvas edge.
PANEL_MARGIN_IN = 0.10

# Dimension callouts match the 9 pt of the tick labels in every other figure,
# and of the sibling schematic in figures/1D_vector_definition_top_view.tex.
DIM_LABEL_PT = 9

# Horizontal pad (km) on either side of the observation window. Sized to hold
# the fore/aft dimension labels on the right, and mirrored on the left.
DIM_PAD_KM = 26

# Observation parameters (from centered_1.yaml / forward_1.yaml)
SHAPE_PX = (16, 16)          # (cols, rows)
RANGE_M = (64_000, 64_000)   # meters
KM_PER_PX = RANGE_M[0] / SHAPE_PX[0] / 1_000   # 4.0 km/pixel
OBS_KM = SHAPE_PX[0] * KM_PER_PX               # 64 km

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


def add_dim_annotations(ax, left, bottom, ac_y, mode):
    """Width dimension line, plus fore/aft ranges measured from the aircraft."""
    color  = FORWARD_COLOR if mode == "forward" else CENTERED_COLOR
    ann_y = bottom - 5
    ax.annotate("", xy=(left + OBS_KM, ann_y), xytext=(left, ann_y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1), zorder=6)
    ax.text(left + OBS_KM / 2, ann_y - 2,
            f"{int(OBS_KM)} km  ({SHAPE_PX[0]} px)",
            ha="center", va="top", color=color, fontsize=DIM_LABEL_PT, zorder=6)

    # The fore/aft split is what distinguishes the two modes, so the vertical
    # extent is dimensioned from the aircraft rather than as one total height.
    ann_x = left + OBS_KM + 5
    top = bottom + OBS_KM
    ahead, behind = top - ac_y, ac_y - bottom
    if ahead > 32:
        px = 16
    else:
        px = 8
    ax.annotate("", xy=(ann_x, top), xytext=(ann_x, ac_y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1), zorder=6)
    # Kilometres only: the width arrow already gives the pixel count, and the
    # longer form is too wide to fit beside a centred window.
    ax.text(ann_x + 2, (ac_y + top) / 2, f"ahead\n{int(ahead)} km\n({px} px)",
            ha="left", va="center", color=color, fontsize=DIM_LABEL_PT, zorder=6)

    if behind > 0:
        ax.annotate("", xy=(ann_x, ac_y), xytext=(ann_x, bottom),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.1), zorder=6)
        ax.text(ann_x + 2, (bottom + ac_y) / 2, f"behind\n{int(behind)} km \n(8 px)",
                ha="left", va="center", color=color, fontsize=DIM_LABEL_PT, zorder=6)
    else:
        ax.text(ann_x + 2, ac_y, "behind\n0 km\n(0 px)",
                ha="left", va="center", color=color, fontsize=DIM_LABEL_PT, zorder=6)

    # Datum through the aircraft, so the split reads as measured from it.
    ax.plot([left, ann_x], [ac_y, ac_y], color=color, lw=0.8,
            ls=(0, (4, 3)), alpha=0.9, zorder=6)


def plot_panel(ax, mode: str):
    ac_x, ac_y = 0.0, 0.0

    left, bottom = draw_obs_window(ax, ac_x, ac_y, mode)
    draw_aircraft(ax, ac_x, ac_y)
    add_dim_annotations(ax, left, bottom, ac_y, mode)

    # Limits derived from the actual window position so both modes stay in
    # frame; span is mode-independent, so panels crop to identical size.
    # The left pad matches the right one that holds the fore/aft labels, so the
    # window itself sits centred in the panel and the two grids line up on the
    # page. Both panels are height-limited, so the wider span costs no size.
    ax.set_xlim(left - DIM_PAD_KM, left + OBS_KM + DIM_PAD_KM)
    ax.set_ylim(bottom - 8, bottom + OBS_KM + 6)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------

out_dir = PLOTS_DIR / "population_maps"

for mode, suffix in [
    ("centered", "centered"),
    ("forward",  "forward"),
]:
    # One panel per 0.49\textwidth subfigure at width=\textwidth, so the saved PDF
    # is exactly its LaTeX slot and the 9 pt labels above are the 9 pt that reaches
    # the page; a tight bbox would undo that. W_HALF x H_STRIP is the same slot as
    # every boxplot panel. The 86:78 km span drawn below is far narrower than the
    # axes box that leaves, so equal aspect fits the drawing to the height and
    # centres it, which is what keeps the schematic itself compact.
    fig, ax = paper_axes(W_HALF, H_STRIP,
                         left=PANEL_MARGIN_IN, right=PANEL_MARGIN_IN,
                         bottom=PANEL_MARGIN_IN, top=PANEL_MARGIN_IN)
    fig.patch.set_facecolor(BG_COLOR)

    plot_panel(ax, mode)

    out_path = out_dir / f"observation_mode_{suffix}.pdf"
    save(fig, out_path, facecolor=fig.get_facecolor())

plt.close()
