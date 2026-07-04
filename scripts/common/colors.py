"""Shared categorical colors for plotting scripts.

Single source of truth for the discrete colors used across the plotting
scripts. Everything is pulled from matplotlib's standard ``Dark2`` qualitative
colormap so series colors are consistent across every figure.

Sequential colormaps for continuous raster/density data (e.g. ``cmap="Blues"``
in ``imshow``) are intentionally *not* defined here — qualitative colormaps do
not apply to continuous data.
"""

import matplotlib.pyplot as plt

DARK2 = plt.colormaps["Dark2"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
    "mathtext.fontset": "stix",   # Times-like math
    "font.size": 10,              # base size = body text
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,         # AIAA captions/small text ~9pt
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": True,
})

def _qual(i: int):
    """Discrete color ``i`` from the Dark2 qualitative colormap (wraps at 8)."""
    return DARK2(i % DARK2.N)


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
TEXTWIDTH_PT = 469
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.7

METRIC_TO_AXIS_REVERS = {
    "fuel" : True,
    "noise": True,
    "normalized_fuel": False,
    "normalized_noise": False,
    "combined": True,
    "reward": False,
    "reward_unclipped": False
}

METRICS = [
    ("fuel", "fuel [kg]"),
    ("noise", "noise [W·s]"),
    ("normalized_fuel", "normalized fuel"),
    ("normalized_noise", "normalized noise"),
    ("combined", "normalized fuel + noise"),
    ("reward", "reward"),
    ("reward_unclipped", "reward (no noise clipping)"),
]

REASON_HATCH = {
    "success":         "",
    "failed_approach": "////",
    "max_steps":       "....",
    "out_of_bounds":   "xxxx",
}

