"""Shared figure geometry for every plot that ends up in the paper.

The point of this module is that the *saved* PDF is exactly as wide as the
LaTeX slot it gets included in, and that every figure in a slot shares one
axes box. LaTeX then never rescales anything, so the 10 pt / 9 pt text set in
``common.colors`` is the text size that actually reaches the page.
"""

from collections.abc import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mtext

from scripts.common.colors import CAPTION_PT, TEXTWIDTH_IN


# Widths (Latex \textwidth).
W_FULL = 1.00
W_WIDE = 0.85
W_THREEQ = 0.75
W_HALF = 0.49    # two per row
W_THIRD = 0.32   # three per row

H_STRIP = 0.33   # short strip: training curves, wide category bars
H_PLOT = 0.45    # default single-panel plot
H_TALL = 0.55    # long tick labels or an annotation band under the axes

PLOT_TYPE_TO_SIZE = {
    "run_reward" : (W_THREEQ, H_STRIP),
    "alignment" : (W_HALF, H_STRIP),
    "transforms" : (W_THREEQ, H_STRIP),
    "weird_plot" : (W_THREEQ, H_STRIP),        # boxplots + legend strip
    "sweep_metric" : (W_HALF, H_STRIP),      # one boxplot panel, two per row
    "sweep_breakdown" : (W_WIDE, H_STRIP),   # stacked bars + legend strip
    "sweep_frontier" : (W_THREEQ, H_PLOT),   # fuel-noise frontier + two legends
}

OUTCOME_Y_LIMITS = ((0.8,1.01), (0.0,1.05))
OUTCOME_Y_TICKS = 5


def outcome_ylim(ax, min_seed_rate: float) -> None:
    """Y-axis for an episode-outcome / success plot.

    Zooms into the top of the range when every per-seed dot stays above it,
    otherwise shows the full 0-100%. Ticks are evenly spaced up to 100% in
    both cases so that outcome panels across the paper share tick positions.
    """
    zoomed, full = OUTCOME_Y_LIMITS
    lo, hi = zoomed if min_seed_rate > zoomed[0] else full
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.linspace(lo, 1.00, OUTCOME_Y_TICKS))

# Margins in inches, identical for every figure so that axes boxes, tick text
# and y-labels line up when two figures share a page.
MARGIN_IN = {"left": 0.62, "right": 0.10, "bottom": 0.42, "top": 0.08}

# Vertical space taken by the pieces under a panel, in inches. Used to reserve
# margins for panel captions typeset by matplotlib instead of by `subcaption`.
CAPTION_SKIP_IN = 10 / 72   # `caption` package default skip between box and caption
CAPTION_LINE_IN = 0.16      # one line of caption text with leading
AXIS_LABEL_IN = 0.20        # an axis label line
TICK_LABEL_IN = 0.16        # a row of tick labels


def paper_axes(width: float = W_HALF, height: float = H_PLOT, **margin_overrides):
    """Figure + single axes sized for a LaTeX slot.

    ``width`` and ``height`` are both fractions of ``\\textwidth`` and give the
    size of the whole figure (use the ``W_*`` / ``H_*`` constants). The axes box
    is what is left after the margins, which default to :data:`MARGIN_IN`;
    override individual ones in inches to reserve space for an outside legend,
    e.g. ``paper_axes(W_THREEQ, H_PLOT, right=1.25)``.
    """
    margins = {**MARGIN_IN, **margin_overrides}
    fig_w = width * TEXTWIDTH_IN
    fig_h = height * TEXTWIDTH_IN
    axes_w = fig_w - margins["left"] - margins["right"]
    axes_h = fig_h - margins["bottom"] - margins["top"]
    if axes_w <= 0 or axes_h <= 0:
        raise ValueError(f"margins {margins} leave no room in a "
                         f"{fig_w:.2f} x {fig_h:.2f} in figure")

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        margins["left"] / fig_w,
        margins["bottom"] / fig_h,
        axes_w / fig_w,
        axes_h / fig_h,
    ])
    return fig, ax


def paper_grid(ncols: int, nrows: int = 1, width: float = W_FULL,
               panel_aspect: float = 1.0, wspace_in: float = 0.07,
               hspace_in: float | Sequence[float] = 0.07, **margin_overrides):
    """Figure + a grid of panels that all share one axes size.

    Unlike :func:`paper_axes` the height is not given: it follows from the panel
    width and ``panel_aspect`` (panel height / panel width, 1.0 = square), so
    map panels come out undistorted instead of matplotlib padding the axes box
    to satisfy ``aspect="equal"``. Margins work as in :func:`paper_axes`;
    reserve room for a shared colorbar, legend or captions through them.

    ``hspace_in`` is either one gap for every row boundary or a sequence of
    ``nrows - 1`` gaps, top to bottom. Per-row gaps matter when text lives
    between the rows: with a single gap, every row has to reserve room for the
    tallest caption in the figure, which shows up as slack around the short ones.

    Returns ``(fig, axes)`` with ``axes`` an ``(nrows, ncols)`` object array,
    row 0 on top.

    Building one figure per LaTeX slot — rather than one PDF per panel dropped
    into ``subfigure``s — is what keeps the panels the same size: only the left
    column carries tick labels, and the space that costs comes out of the shared
    margin instead of out of that panel's map.
    """
    margins = {**MARGIN_IN, **margin_overrides}
    gaps = ([float(hspace_in)] * (nrows - 1) if np.isscalar(hspace_in)
            else [float(gap) for gap in hspace_in])
    if len(gaps) != nrows - 1:
        raise ValueError(f"hspace_in needs {nrows - 1} gaps for {nrows} rows, got {len(gaps)}")

    fig_w = width * TEXTWIDTH_IN
    grid_w = fig_w - margins["left"] - margins["right"] - wspace_in * (ncols - 1)
    panel_w = grid_w / ncols
    if panel_w <= 0:
        raise ValueError(f"margins {margins} leave no room for {ncols} panels "
                         f"in a {fig_w:.2f} in figure")
    panel_h = panel_w * panel_aspect
    fig_h = margins["bottom"] + margins["top"] + nrows * panel_h + sum(gaps)

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        # Height of everything below this row: the rows under it plus their gaps.
        below = (nrows - 1 - row) * panel_h + sum(gaps[row:])
        for col in range(ncols):
            left = margins["left"] + col * (panel_w + wspace_in)
            bottom = margins["bottom"] + below
            axes[row, col] = fig.add_axes([
                left / fig_w, bottom / fig_h, panel_w / fig_w, panel_h / fig_h,
            ])
    return fig, axes


# ---------------------------------------------------------------- metric grids

# The combined sweep figure: the metric panels that used to be a 2x2 block of
# `subfigure`s in the paper, drawn as one matplotlib figure instead. The cell
# the odd panel count leaves free holds the legend, and the (a), (b), …
# captions sit above the panels rather than under them.
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
GRID_CAPTION_WEIGHT = "bold"     # set to "normal" for plain captions
GRID_CAPTION_PAD_IN = 0.08       # gap between a caption and the panel under it
# Every panel carries its own y label and tick labels, so the column gap has to
# pay for that block plus free space. It is *not* the figure's left margin: that
# one is padded so axes boxes line up across figures, and paying for the padding
# again between the columns shows up as a visible gutter. Y_LABEL_BLOCK_IN is
# measured — the widest sweep panel puts 0.43 in left of its axes box.
Y_LABEL_BLOCK_IN = AXIS_LABEL_IN + TICK_LABEL_IN + 0.09
GRID_COLUMN_GAP_IN = Y_LABEL_BLOCK_IN + 0.10
# Between two rows sits the upper row's x tick labels and axis label, then the
# lower row's caption.
GRID_ROW_GAP_IN = (TICK_LABEL_IN + AXIS_LABEL_IN + 0.10
                   + CAPTION_LINE_IN + GRID_CAPTION_PAD_IN)


def _sweep_panel_aspect() -> float:
    """Axes-box shape of a standalone ``sweep_metric`` panel.

    Grid panels keep it, so the boxes read the same whether a metric is shown
    on its own or in the combined figure.
    """
    width, height = PLOT_TYPE_TO_SIZE["sweep_metric"]
    return ((height * TEXTWIDTH_IN - MARGIN_IN["bottom"] - MARGIN_IN["top"])
            / (width * TEXTWIDTH_IN - MARGIN_IN["left"] - MARGIN_IN["right"]))


def metric_grid(npanels: int, ncols: int = 2, width: float = W_FULL,
                legend: bool = True):
    """Figure for ``npanels`` metric panels plus one spare cell for the legend.

    Returns ``(fig, panel_axes, legend_ax)`` with ``panel_axes`` in reading
    order. Every cell past the panels is hidden, so a grid that is not exactly
    filled shows white space rather than an empty axes box; ``legend_ax`` is the
    first of those (``None`` with ``legend=False``, which also drops the cell
    reserved for it from the row count). Pair with :func:`grid_caption` and
    :func:`legend_in_cell`.
    """
    ncells = npanels + (1 if legend else 0)   # one cell per panel plus the legend's
    nrows = -(-ncells // ncols)
    fig, axes = paper_grid(
        ncols, nrows, width=width, panel_aspect=_sweep_panel_aspect(),
        wspace_in=GRID_COLUMN_GAP_IN, hspace_in=GRID_ROW_GAP_IN,
        top=MARGIN_IN["top"] + CAPTION_LINE_IN + GRID_CAPTION_PAD_IN,
    )
    flat = [axes[index // ncols, index % ncols] for index in range(nrows * ncols)]
    for ax in flat[npanels:]:
        ax.set_visible(False)
    return fig, flat[:npanels], (flat[npanels] if legend else None)


def grid_caption(ax, letter: str, text: str):
    """Panel caption above ``ax``, in the size and weight `subcaption` would use."""
    return ax.set_title(f"({letter}) {text}", size=CAPTION_PT,
                        weight=GRID_CAPTION_WEIGHT, pad=GRID_CAPTION_PAD_IN * 72)


def grid_latex_snippet(save_path, labels: Sequence[str], width: float = W_FULL) -> str:
    r"""The ``figure`` environment to paste, with phantom subcaptions for ``\subref``.

    The captions are drawn by matplotlib now, so the ``subfigure``s exist only
    to own the ``(a)``, ``(b)``, … labels the text refers to — same trick as in
    plot_trajectory_figure.
    """
    stem = save_path.stem
    phantoms = "\n".join(
        f"  \\begin{{subfigure}}{{0pt}}\\phantomsubcaption\\label{{fig:{stem}-{letter}}}\\end{{subfigure}}%"
        for letter, _ in zip(PANEL_LETTERS, labels)
    )
    include_width = "\\textwidth" if width >= 0.999 else f"{width:g}\\textwidth"
    caption = "... " + "; ".join(
        f"(\\subref{{fig:{stem}-{letter}}}) {label}"
        for letter, label in zip(PANEL_LETTERS, labels)) + "."
    return (
        "\\begin{figure}[tb]\n"
        "  \\centering\n"
        f"{phantoms}\n"
        f"  \\includegraphics[width={include_width}]{{{save_path.name}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{stem}}}\n"
        "\\end{figure}"
    )


def figure_bbox(ax):
    """``ax``'s bounding box *including* its tick labels, in figure coords."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    return ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())


def row_caption_y(axes, extra_in: float = 0.0, skip_in: float = CAPTION_SKIP_IN) -> float:
    """Shared caption baseline for one row of panels, in figure coords.

    Measured from the lowest tick label in the row so every caption in the row
    sits on the same line, and offset by ``extra_in`` to clear anything drawn
    below the panels that matplotlib does not know about (a shared axis label).
    """
    axes = [ax for ax in np.ravel(axes) if ax is not None]
    fig = axes[0].figure
    fig.canvas.draw()
    bottom = min(figure_bbox(ax).y0 for ax in axes)
    return bottom - (skip_in + extra_in) / fig.get_figheight()


def panel_caption(ax, text: str, y: float, size: float = CAPTION_PT, **kwargs):
    """Draw a panel caption centred under ``ax`` at baseline ``y``.

    Stands in for a ``subcaption`` under a ``subfigure``: same font, same size,
    same default 10 pt skip. ``text`` goes through matplotlib's mathtext, so
    ``$\\lambda$`` works but LaTeX macros do not.
    """
    fig = ax.figure
    pos = ax.get_position()
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "top")
    return fig.text(0.5 * (pos.x0 + pos.x1), y, text, size=size, **kwargs)


def grid_labels(fig, axes, xlabel: str | None = None, ylabel: str | None = None,
                x_pad_in: float = 0.05, y_pad_in: float = 0.05) -> None:
    """One x/y label for a whole :func:`paper_grid`, centred on the panel block.

    ``fig.supxlabel`` positions itself against the figure and drifts as soon as
    the margins are asymmetric; this anchors on the axes boxes instead. Pass the
    axes that actually carry tick labels (bottom row / left column).
    """
    axes = [ax for ax in np.ravel(axes) if ax is not None]
    fig.canvas.draw()
    boxes = [figure_bbox(ax) for ax in axes]
    positions = [ax.get_position() for ax in axes]
    if xlabel is not None:
        x_mid = 0.5 * (min(p.x0 for p in positions) + max(p.x1 for p in positions))
        fig.text(x_mid, min(b.y0 for b in boxes) - x_pad_in / fig.get_figheight(),
                 xlabel, ha="center", va="top")
    if ylabel is not None:
        y_mid = 0.5 * (min(p.y0 for p in positions) + max(p.y1 for p in positions))
        fig.text(min(b.x0 for b in boxes) - y_pad_in / fig.get_figwidth(), y_mid,
                 ylabel, ha="right", va="center", rotation="vertical")


def legend_in_cell(fig, ax, handles, **kwargs):
    """Draw a legend centred in an unused :func:`paper_grid` cell.

    Hides ``ax`` and puts the legend where that panel would have been, so a grid
    with one cell to spare pays for its legend out of the hole in the grid
    instead of out of a reserved margin. Anchored on the cell's centre with
    ``loc="center"``, which is what centres the legend box on that point.
    """
    ax.set_visible(False)
    pos = ax.get_position()
    kwargs.setdefault("loc", "center")
    kwargs.setdefault("edgecolor", "k")
    return fig.legend(handles=handles,
                      bbox_to_anchor=(0.5 * (pos.x0 + pos.x1), 0.5 * (pos.y0 + pos.y1)),
                      **kwargs)


def right_margin_x(ax, pad_in: float = 0.10) -> float:
    """``bbox_to_anchor`` x that puts an artist in the reserved right margin.

    In axes coordinates, so it pairs with a ``loc="… left"`` legend. Use it
    directly when several artists share the margin (e.g. two legends stacked at
    y=1 and y=0); :func:`legend_right` is the single-legend shorthand.
    """
    return 1 + pad_in / (ax.get_position().width * ax.figure.get_figwidth())


def legend_right(ax, pad_in: float = 0.10, **kwargs):
    """Legend centred in the right margin reserved by :func:`paper_axes`."""
    kwargs.setdefault("loc", "center left")
    kwargs.setdefault("bbox_to_anchor", (right_margin_x(ax, pad_in), 0.5))
    return ax.legend(**kwargs)


def _undrawn_tick_labels(fig) -> set[int]:
    """Ids of tick labels matplotlib keeps for locations outside the view.

    A locator often hands back one tick past each end of the axis; matplotlib
    holds on to those labels but never draws them. They still report a window
    extent — far outside the canvas for a panel near the figure edge — so
    :func:`save` has to leave them out of its overflow check.
    """
    undrawn: set[int] = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not low <= tick.get_loc() <= high:
                    undrawn.update({id(tick.label1), id(tick.label2)})
    return undrawn


def save(fig, path, dpi: int = 150, **kwargs) -> None:
    """Save at exactly the requested figure size, warning about clipped artists.

    Deliberately does not accept ``bbox_inches``: the whole point of
    :func:`paper_axes` is that the saved width is known up front.
    """
    kwargs.pop("bbox_inches", None)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Text is checked separately: a colorbar label long enough to run off the
    # top of a short figure does not show up in fig.get_tightbbox(), so the
    # clipping it causes would otherwise go unreported.
    boxes = [fig.get_tightbbox(renderer)]
    to_inches = fig.dpi_scale_trans.inverted()
    undrawn = _undrawn_tick_labels(fig)
    for text in fig.findobj(mtext.Text):
        if text.get_visible() and text.get_text().strip() and id(text) not in undrawn:
            boxes.append(text.get_window_extent(renderer).transformed(to_inches))

    overflow = max(
        max(-box.x0, -box.y0, box.x1 - fig.get_figwidth(), box.y1 - fig.get_figheight())
        for box in boxes
    )
    if overflow > 0.01:
        print(f"  ! {path}: content overflows the canvas by {overflow:.2f} in — "
              f"widen the corresponding margin in paper_axes()")
    fig.savefig(path, dpi=dpi, **kwargs)
    print(f"Saved → {path} ({fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in)")
