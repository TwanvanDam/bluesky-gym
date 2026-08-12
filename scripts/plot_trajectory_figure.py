r"""Compose several trajectory panels into one paper figure, captions and all.

The figure is described by a small text file: one line per panel, a blank line
starts a new row. Everything after ``#`` is a comment.

    # Block A — observation resolution (EDDF)
    runs/resolution_sweep_2/sweep_2_centered_4_seed00/trajectories/EDDF_RW25R/trajectories.csv,  4 px
    runs/resolution_sweep_2/sweep_2_centered_16_seed00/trajectories/EDDF_RW25R/trajectories.csv, 16 px
    runs/resolution_sweep_2/sweep_2_centered_32_seed00/trajectories/EDDF_RW25R/trajectories.csv, 32 px

    runs/resolution_sweep_2/sweep_2_forward_16_seed00/trajectories/EDDF_RW25R/trajectories.csv,  Forward, $\kappa = 4$

Run it with::

    python -m scripts.plot_trajectory_figure figures/resolution.txt --out figures/resolution.pdf

Each line is the path to a ``trajectories.csv``, a comma, and the panel caption.
Only the first comma separates the two, so captions may contain commas. The
``(a)``, ``(b)``, … prefix is added automatically in reading order.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import bluesky as bs
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser

from bluesky_gym.maps.map_sources import TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler
from scripts.common import figures as fg
from scripts.common.colors import CAPTION_PT
from scripts.present_trajectories import (
    MARGIN_M, PLOT_CRS, TrajectoryPanel, add_colorbar, build_norm, density_cmap,
    density_label, draw_background, draw_terminal_geometry, draw_trajectories,
    km_ticks, legend_handles, load_panel, make_transformer, panel_half_width,
    project_panel, scale_bar,
)

PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
CAPTION_WEIGHT = "bold"   # set to "normal" for plain captions

# Room reserved around the grid, in inches. Each entry is only paid for when the
# corresponding element is actually drawn.
YTICK_LABEL_IN = 0.38   # tick marks + a 4-digit label at 9 pt
COLORBAR_IN = 0.90      # gap + bar + tick labels + label
LEGEND_LINE_IN = 0.20   # one row of legend entries
XTICK_OVERHANG_IN = 0.18  # the outermost x tick label sticks out past the axes box
YTICK_OVERHANG_IN = 0.10  # ditto for the topmost y tick label
EDGE_IN = 0.06
PANEL_GAP_IN = 0.16       # free space between panels, the same in x and y


def canvas_width(width: float, draw_legend: bool) -> float:
    r"""Width of the saved canvas, as a fraction of ``\textwidth``.

    ``--width`` sizes the panels and the colorbar. The legend is text about the
    figure rather than part of it, so it gets the full text width: with a legend
    the canvas stays full width and the panel block is centred inside it. LaTeX
    then includes the figure at ``\textwidth`` — scaling it down instead would
    shrink the legend along with the panels, and its labels are set in the same
    9 pt as the rest of the paper.
    """
    return fg.W_FULL if draw_legend else width


def legend_layout(handles, available_in: float) -> tuple[int, int, float]:
    """Pick the widest legend shape that still fits in ``available_in``.

    The number of legend rows has to be known *before* the figure is created,
    because it decides the top margin — so this estimates the width from the
    label lengths (Times averages about half an em per character) instead of
    measuring a legend that does not exist yet. Returns ``(ncol, nrows, width)``.
    """
    char_in = 0.5 * plt.rcParams["legend.fontsize"] / 72
    entries = sorted((len(h.get_label()) * char_in + 0.45 for h in handles), reverse=True)
    for nrow in range(1, len(handles) + 1, 1):
        ncol = len(handles) // nrow
        width = sum(entries[:ncol])
        if width <= available_in or ncol == 1:
            return ncol, nrow, width
    raise AssertionError("unreachable: ncol=1 always returns")


@dataclass
class PanelSpec:
    csv_path: Path
    caption: str


def text_width_in(text: str, size: float = CAPTION_PT, weight: str = "normal") -> float:
    """Rendered width of ``text`` in inches, mathtext included."""
    fig = plt.figure(figsize=(1, 1))
    artist = fig.text(0, 0, text, size=size, weight=weight)
    fig.canvas.draw()
    width = artist.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    plt.close(fig)
    return width


def split_words(text: str) -> list[str]:
    """Split on spaces, but never inside ``$…$`` — that would break the maths."""
    words, current, in_math = [], "", False
    for char in text:
        if char == "$":
            in_math = not in_math
        if char == " " and not in_math:
            if current:
                words.append(current)
            current = ""
        else:
            current += char
    if current:
        words.append(current)
    return words


def normalize_caption(text: str) -> str:
    r"""Resolve ``\\`` in a spec caption, differently in and out of maths.

    Outside ``$…$`` it means what it means in LaTeX — a forced line break.
    Inside, it is the doubled backslash you get from copying an escaped string
    (``$\\kappa$``), which mathtext would read as a break and reject, so it
    collapses back to one.
    """
    parts, in_math = [], False
    for part in re.split(r"(\$)", text):
        if part == "$":
            in_math = not in_math
            parts.append(part)
        else:
            parts.append(part.replace("\\\\", "\\" if in_math else "\n"))
    return "".join(parts)


def wrap_caption(text: str, max_width_in: float, weight: str = "normal") -> str:
    """Greedy word wrap to ``max_width_in``, measuring the real rendered width.

    A caption wider than its panel would otherwise run into the neighbouring
    panel's caption — ``subcaption`` wraps for you, matplotlib does not. Breaks
    already in the text (from ``\\\\``) are kept and each part wrapped on its own.
    """
    wrapped = []
    for forced_line in text.split("\n"):
        lines, current = [], ""
        for word in split_words(forced_line):
            candidate = f"{current} {word}".strip()
            if current and text_width_in(candidate, weight=weight) > max_width_in:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
        wrapped.extend(lines)
    return "\n".join(wrapped)


def check_caption(caption: str, spec_path: Path, line_number: int) -> str:
    """Fail early, and with the spec line, on maths matplotlib cannot render."""
    for text_line in caption.split("\n"):   # mathtext parses one line at a time
        try:
            MathTextParser("agg").parse(text_line, dpi=72, prop=FontProperties(size=CAPTION_PT))
        except ValueError as error:
            raise SystemExit(
                f"{spec_path}:{line_number}: cannot render caption {caption!r}\n"
                f"Captions use matplotlib mathtext, not LaTeX — write maths as in a "
                f".tex file (one backslash, e.g. $\\kappa = 4$) and avoid macros.\n{error}"
            ) from None
    return caption


def parse_spec(spec_path: Path) -> list[list[PanelSpec]]:
    """Parse the spec file into rows of panels."""
    spec_rows: list[list[PanelSpec]] = []
    current: list[PanelSpec] = []
    for line_number, raw_line in enumerate(spec_path.read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            if current:
                spec_rows.append(current)
                current = []
            continue
        # Only the first comma separates path from caption, so captions like
        # "Centered, 16 px" survive intact.
        path, _, caption = line.partition(",")
        caption = normalize_caption(caption.strip())
        current.append(PanelSpec(Path(path.strip()),
                                 check_caption(caption, spec_path, line_number)))
    if current:
        spec_rows.append(current)

    if not spec_rows:
        raise SystemExit(f"{spec_path}: no panels found")
    return spec_rows


def trajectory_dir(csv_path: Path) -> Path:
    """Directory holding trajectories.csv + details, from either as a reference."""
    directory = csv_path.parent if csv_path.suffix == ".csv" else csv_path
    if not (directory / "trajectories.csv").exists():
        raise SystemExit(f"{csv_path}: no trajectories.csv here")
    return directory


def load_panels(spec_rows: list[list[PanelSpec]], transformer) -> list[list[TrajectoryPanel]]:
    loaded_rows: list[list[TrajectoryPanel]] = []
    for spec_row in spec_rows:
        loaded = []
        for spec in spec_row:
            directory = trajectory_dir(spec.csv_path)
            panel = load_panel(directory, caption=spec.caption)
            if panel is None:
                raise SystemExit(f"{directory}: missing details.json/details.pkl")
            loaded.append(project_panel(panel, transformer))
        loaded_rows.append(loaded)
    return loaded_rows


def plot_figure(panels: list[list[TrajectoryPanel]], save_path: Path, background_map: Path,
                width: float = fg.W_FULL, normalization_mode: str = "log",
                normalization_percentile: float = 99.9, range_km: float | None = 250,
                draw_axes: bool = True, axis_labels: bool = True, draw_colorbar: bool = True,
                draw_legend: bool = True, captions: str = "below",
                panel_gap: float = PANEL_GAP_IN) -> None:
    """Draw the whole figure: one square panel per spec line.

    ``captions`` places the ``(a) …`` labels ``"below"`` each panel (what
    ``subcaption`` does), ``"above"`` them as panel titles, or ``"inside"`` the
    panel as a bare letter with the descriptions left to the main LaTeX caption.
    Above and inside keep the captions clear of the tick labels entirely, which
    is worth a lot once the figure has more than one row.
    """
    map_source = TiffMapSourceConfig(file_path=background_map).build()
    raster_sampler = RasterSampler(map_source, resampling="cubic_spline", destination_crs=PLOT_CRS)
    transformer = make_transformer(raster_sampler.destination_crs)
    v_max = map_source.get_normalization_value(normalization_percentile)
    norm = build_norm(normalization_mode, v_max)
    cmap = density_cmap()

    # One window size for every panel: same km-per-inch everywhere, so the panels
    # can actually be compared against each other.
    if range_km:
        half_width_m = range_km * 1000 + MARGIN_M
    else:
        half_width_m = max(panel_half_width(panel) for row in panels for panel in row)

    nrows = len(panels)
    ncols = max(len(row) for row in panels)

    tick_block = fg.TICK_LABEL_IN if draw_axes else 0.0
    xticks_every_row = captions == "below"
    label_block = fg.AXIS_LABEL_IN + 0.06 if (draw_axes and axis_labels) else 0.0

    canvas = canvas_width(width, draw_legend)
    side_pad = 0.5 * (canvas - width) * fg.TEXTWIDTH_IN

    handles = legend_handles()
    legend_ncol, legend_rows, legend_width = legend_layout(
        handles, canvas * fg.TEXTWIDTH_IN - 2 * EDGE_IN)
    legend_block = legend_rows * LEGEND_LINE_IN + 0.08

    empty_cells = [(row, col) for row in range(nrows)
                   for col in range(len(panels[row]), ncols)]
    colorbar_cell = empty_cells[0] if (draw_colorbar and empty_cells) else None

    left = side_pad + ((YTICK_LABEL_IN + (fg.AXIS_LABEL_IN if axis_labels else 0.0))
                       if draw_axes else EDGE_IN)
    right = side_pad + (COLORBAR_IN if draw_colorbar and colorbar_cell is None
                        else (XTICK_OVERHANG_IN if draw_axes else EDGE_IN))

    wspace = panel_gap
    panel_width_in = (canvas * fg.TEXTWIDTH_IN - left - right - wspace * (ncols - 1)) / ncols
    row_lines = [1] * nrows
    if captions in ("below", "above"):
        letters = iter(PANEL_LETTERS)
        for row_index, row in enumerate(panels):
            for panel in row:
                panel.caption = wrap_caption(f"({next(letters)}) {panel.caption}".strip(),
                                             panel_width_in, weight=CAPTION_WEIGHT)
                row_lines[row_index] = max(row_lines[row_index], panel.caption.count("\n") + 1)

    def caption_block(row_index: int) -> float:
        height = row_lines[row_index] * fg.CAPTION_LINE_IN
        if captions == "below":
            return fg.CAPTION_SKIP_IN + height
        if captions == "above":
            return height + 0.08   # the title's pad above the axes
        return 0.0

    gaps = [(tick_block if xticks_every_row else 0.0)
            + (caption_block(row) if captions == "below" else caption_block(row + 1))
            + wspace
            for row in range(nrows - 1)]

    edge = max(EDGE_IN, YTICK_OVERHANG_IN if colorbar_cell is None and draw_colorbar else 0.0)
    fig, axes = fg.paper_grid(
        ncols, nrows, width=canvas, panel_aspect=1.0, wspace_in=wspace, hspace_in=gaps,
        left=left, right=right,
        top=(YTICK_OVERHANG_IN if draw_axes else edge)
            + (caption_block(0) if captions == "above" else 0.0),
        bottom=tick_block + (caption_block(nrows - 1) if captions == "below" else 0.0)
               + label_block + edge + (legend_block if draw_legend else 0.0),
    )

    used_axes: list = []
    image = None
    for row_index, row in enumerate(panels):
        for col_index in range(ncols):
            ax = axes[row_index, col_index]
            if col_index >= len(row):
                ax.set_visible(False)  # ragged last row
                continue
            panel = row[col_index]
            used_axes.append(ax)

            image = draw_background(ax, raster_sampler, panel.center, half_width_m, norm, cmap)
            draw_trajectories(ax, panel)
            draw_terminal_geometry(ax, panel.destination, transformer)
            if draw_axes:
                # Ticks on the start distance itself: the outer ring is where the
                # aircraft spawn, so the label carries meaning a round number does not.
                km_ticks(ax, panel.center, half_width_m, step_km=range_km,
                         show_x=xticks_every_row or row_index == nrows - 1,
                         show_y=col_index == 0)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                if row_index == 0 and col_index == 0:
                    scale_bar(ax, half_width_m)

    # panel.caption already carries its "(a) …" prefix, wrapped to the panel.
    letters = iter(PANEL_LETTERS)
    last_caption_y = 1.0
    for row_index, row in enumerate(panels):
        row_axes = [axes[row_index, col] for col in range(len(row))]
        # One shared baseline per row, measured from the lowest tick label in
        # that row, so the panel-to-caption distance is identical everywhere.
        if captions == "below":
            last_caption_y = fg.row_caption_y(row_axes)
        for ax, panel in zip(row_axes, row):
            if captions == "below":
                fg.panel_caption(ax, panel.caption, last_caption_y, weight=CAPTION_WEIGHT)
            elif captions == "above":
                ax.set_title(panel.caption, size=CAPTION_PT, pad=3, weight=CAPTION_WEIGHT)
            else:  # inside — bare letter, descriptions belong in the main caption
                ax.text(0.035, 0.965, f"({next(letters)})", transform=ax.transAxes,
                        ha="left", va="top", size=CAPTION_PT, weight=CAPTION_WEIGHT, zorder=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                                  boxstyle="square,pad=0.25"))

    if draw_axes and axis_labels:
        fg.grid_labels(fig, [axes[row, 0] for row in range(nrows)],
                       ylabel=r"$y$-coordinate [km]")
        if captions == "below":
            positions = [ax.get_position() for ax in used_axes]
            x_mid = 0.5 * (min(p.x0 for p in positions) + max(p.x1 for p in positions))
            fig.text(x_mid, last_caption_y - (fg.CAPTION_LINE_IN + 0.06) / fig.get_figheight(),
                     r"$x$-coordinate [km]", ha="center", va="top")
        else:
            fg.grid_labels(fig, [axes[nrows - 1, col] for col in range(len(panels[-1]))],
                           xlabel=r"$x$-coordinate [km]")

    if draw_colorbar:
        fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
        bar = 0.13 / fig_w
        if colorbar_cell is None:
            first = axes[0, ncols - 1].get_position()
            last = axes[nrows - 1, ncols - 1].get_position()
            cax = fig.add_axes([last.x1 + 0.10 / fig_w, last.y0, bar, first.y1 - last.y0])
            label_room = fig_h - 2 * EDGE_IN
        else:
            cell = axes[colorbar_cell].get_position()
            block = (0.13 + 0.05 + YTICK_LABEL_IN + fg.AXIS_LABEL_IN) / fig_w
            height = 0.86 * cell.height
            cax = fig.add_axes([cell.x0 + 0.5 * (cell.width - block),
                                cell.y0 + 0.5 * (cell.height - height), bar, height])
            label_room = height * fig_h
        add_colorbar(fig, image, cax, normalization_mode, v_max,
                     label=density_label(label_room))

    if draw_legend:
        positions = [ax.get_position() for ax in used_axes]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_figure = fig.transFigure.inverted()
        floor = min([fg.figure_bbox(ax).y0 for ax in used_axes]
                    + [text.get_window_extent(renderer).transformed(to_figure).y0
                       for text in fig.texts])
        bottom = floor - 0.06 / fig.get_figheight()
        # Centred on the panels, then pulled back inside the canvas if that
        # would hang the legend off the edge of a narrow figure.
        x_mid = 0.5 * (min(p.x0 for p in positions) + max(p.x1 for p in positions))
        half = 0.5 * legend_width / fig.get_figwidth()
        edge = EDGE_IN / fig.get_figwidth()
        x_mid = min(max(x_mid, edge + half), 1 - edge - half)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(x_mid, bottom),
                   ncol=legend_ncol, frameon=False, handlelength=1.6,
                   columnspacing=1.2, borderpad=0.0)

    fg.save(fig, save_path)
    plt.close(fig)


def latex_snippet(save_path: Path, spec_rows: list[list[PanelSpec]], width: float,
                  captions: str = "below", draw_legend: bool = False) -> str:
    """The figure environment to paste, with phantom subcaptions for \\subref.

    The include width is the *canvas* width, not ``--width``: a figure with a
    legend is already padded out to the full text width around narrower panels,
    and scaling it in LaTeX on top of that would undo exactly what that buys.
    """
    stem = save_path.stem
    letters = iter(PANEL_LETTERS)
    phantoms = "\n".join(
        f"  \\begin{{subfigure}}{{0pt}}\\phantomsubcaption\\label{{fig:{stem}-{next(letters)}}}\\end{{subfigure}}%"
        for spec_row in spec_rows for _ in spec_row
    )
    canvas = canvas_width(width, draw_legend)
    include_width = "\\textwidth" if canvas >= 0.999 else f"{canvas:g}\\textwidth"

    # With the letters inside the panels the descriptions have nowhere to live,
    # so they belong in the main caption as a run-in list.
    caption = "..."
    if captions == "inside":
        letters = iter(PANEL_LETTERS)
        caption = "... " + "; ".join(
            f"(\\subref{{fig:{stem}-{next(letters)}}}) {spec.caption}"
            for spec_row in spec_rows for spec in spec_row) + "."

    return (
        "\\begin{figure}[tb]\n"
        "  \\centering\n"
        f"{phantoms}\n"
        f"  \\includegraphics[width={include_width}]{{{save_path.name}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{stem}}}\n"
        "\\end{figure}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path, help="Text file describing the panels (see module docstring)")
    parser.add_argument("--out", type=Path, required=False, help="Output PDF path, defaults to spec with .pdf extension")
    parser.add_argument("--width", type=float, default=fg.W_FULL,
                        help="Width of the panels and colorbar as a fraction of \\textwidth "
                             "(default: 1.0). With --legend the canvas stays full width and "
                             "the panel block is centred in it, so the legend keeps the whole "
                             "text width")
    parser.add_argument("--background_map_path", type=Path,
                        default=Path("./scripts/population_maps/europe_3035_1km.tif"))
    parser.add_argument("--normalization_percentile", type=float, default=99.9)
    parser.add_argument("--normalization_mode", type=str, default="log")
    parser.add_argument("--range_km", type=float, default=250,
                        help="Half-width of every panel window, in km, and the tick step (default: 250)")
    parser.add_argument("--fit_window", action="store_true",
                        help="Fit the window to the trajectories instead of --range_km")
    parser.add_argument("--no_axes", action="store_true",
                        help="Drop the axes entirely and put a scale bar in the first panel")
    parser.add_argument("--no_axis_labels", action="store_true",
                        help="Keep the tick labels but drop the x/y axis labels")
    parser.add_argument("--captions", choices=("below", "above", "inside"), default="above",
                        help="Where the (a), (b), … captions go (default: below, like subcaption)")
    parser.add_argument("--panel_gap", type=float, default=PANEL_GAP_IN,
                        help=f"Free space between panels in inches, same in x and y "
                             f"(default: {PANEL_GAP_IN})")
    parser.add_argument("--no_colorbar", action="store_true", help="Drop the shared density colorbar")
    parser.add_argument("--legend", action="store_true", help="Draw the shared legend below the panels")
    args = parser.parse_args()

    spec_rows = parse_spec(args.spec)
    print(f"{args.spec}: {sum(len(r) for r in spec_rows)} panels in {len(spec_rows)} row(s)")
    out_path = args.out if args.out else Path("plots/trajectories") / args.spec.with_suffix(".pdf").name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bs.init()
    panels = load_panels(spec_rows, make_transformer())
    plot_figure(panels, out_path, background_map=args.background_map_path, width=args.width,
                normalization_mode=args.normalization_mode,
                normalization_percentile=args.normalization_percentile,
                range_km=None if args.fit_window else args.range_km,
                draw_axes=not args.no_axes, axis_labels=not args.no_axis_labels,
                draw_colorbar=not args.no_colorbar, draw_legend=args.legend,
                captions=args.captions, panel_gap=args.panel_gap)

    print("\n" + latex_snippet(out_path, spec_rows, args.width, args.captions,
                               draw_legend=args.legend))
