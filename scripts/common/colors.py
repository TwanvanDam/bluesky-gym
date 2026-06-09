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


def qual(i: int):
    """Discrete color ``i`` from the Dark2 qualitative colormap (wraps at 8)."""
    return DARK2(i % DARK2.N)


# Semantic assignments — keep these consistent across all scripts.
MODE_COLORS = {"forward": qual(0), "centered": qual(1)}  # teal / orange
BASELINE_COLOR = qual(7)                                 # Dark2's gray
SEED_COLORS = [qual(i) for i in range(DARK2.N)]          # per-seed cycle
REGION_COLORS = [qual(0), qual(1), qual(2)]              # EHAM / EDDF / sim env
COMPARE_COLORS = (qual(0), qual(1))                      # run A vs run B

# Termination-reason stacked bars (success is drawn separately by mode color).
REASON_COLORS = {
    "failed_approach": qual(5),
    "max_steps": qual(7),
    "out_of_bounds": qual(3),
}
FALLBACK_REASON_COLOR = qual(6)
