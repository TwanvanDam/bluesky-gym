"""Shared categorical colors for plotting scripts.

Single source of truth for the discrete colors used across the plotting
scripts. Everything is pulled from matplotlib's standard ``Dark2`` qualitative
colormap so series colors are consistent across every figure.

Sequential colormaps for continuous raster/density data (e.g. ``cmap="Blues"``
in ``imshow``) are intentionally *not* defined here — qualitative colormaps do
not apply to continuous data.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DARK2 = plt.colormaps["Dark2"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
    "mathtext.fontset": "stix",   # Times-like math
    "font.size": 9,              # base size = body text
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,         # AIAA captions/small text ~9pt
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "pdf.fonttype": 42,           # TrueType, not Type 3 (searchable, no checker complaints)
})

def _qual(i: int):
    """Discrete color ``i`` from the Dark2 qualitative colormap (wraps at 8)."""
    return DARK2(i % DARK2.N)

def get_pygame_color(matplotlib_color_name):
    """Converts any Matplotlib color name or hex string to a Pygame RGBA tuple."""
    rgba_float = mcolors.to_rgba(matplotlib_color_name)
    return tuple(int(channel * 255) for channel in rgba_float)

TRAJECTORY_COLOR = "black"
SINK_COLOR = "green"
RESTRICT_COLOR = "red"
BACKGROUND_COLOR = "grey"
OBSERVATION_WINDOW_COLOR = "red"
HEATMAP_COLORS = "Blues"

# Semantic assignments — keep these consistent across all scripts.
BASELINE_COLOR = _qual(7)                                 # Dark2's gray
FORWARD_COLOR = _qual(0) # teal
CENTERED_COLOR =  _qual(1) # orange
POWER_COLOR = _qual(2)
SCALE_COLOR = _qual(3)
FLOOR_COLOR = _qual(4)
HIGHLIGHT_COLOR = _qual(5)
MULTI_SCALE_COLOR = _qual(2)
SEED_COLORS = [_qual(i) for i in (2,3,4,5,6)]          # per-seed cycle
TRANSFORMS_COLOR =_qual(4)
UNKNOWN_COLOR = _qual(6)                                  # config matched no color rule

BOXPLOT_ALPHA = 0.6
BOXPLOT_ALPHA_LIGHT = 0.4
TEXTWIDTH_PT = 469.755        # \the\textwidth of new-aiaa, in TeX points
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27
# Size subcaptions are set in, for panel captions drawn by matplotlib instead of
# by `subcaption`. Measure it the same way as TEXTWIDTH_PT — put
# \makeatletter\typeout{CAPSIZE=\f@size}\makeatother inside a \caption{} and
# read the log — and correct this if the class disagrees.
CAPTION_PT = 8.0

METRIC_TO_AXIS_REVERS = {
    "fuel" : False,
    "noise": False,
    "normalized_fuel": False,
    "normalized_noise": False,
    "combined": False,
    "reward": False,
    "reward_unclipped": False
}

# Axis label per metric — what goes on the y axis wherever that metric is drawn,
# standalone panel or grid cell.
METRICS = {
    "fuel": "Fuel [kg]",
    "noise": "Noise [W·s]",
    "normalized_fuel": r"$F_\text{ep}$ [-]",
    "normalized_noise": r"$N_\text{ep}$ [-]",
    "combined": "Fuel + Noise",
    "reward": "Reward (clipped)",
    "reward_unclipped": r"$R_\text{ep}$ [-]",
}

# The metric grid: which metrics get a panel, in reading order, and the short
# name used for the (a), (b), … panel caption and the LaTeX \subref list. The
# caption is deliberately shorter than the axis label — the symbol and unit are
# already on the axis, so repeating them above the panel only adds noise.
METRIC_TO_CAPTION = {
    "reward_unclipped": "Reward",
    "normalized_noise": "Noise",
    "normalized_fuel": "Fuel",
}

REASON_HATCH = {
    "success":         "",
    "failed_approach": "////",
    "max_steps":       "....",
    "out_of_bounds":   "xxxx",
}

FILLED_REASONS = {"success", "failed_approach"}

