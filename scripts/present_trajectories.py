"""Plot evaluation trajectories on the population-density map.

Two entry points:

* this script — one PDF per trajectory subdirectory, next to the CSV;
* :mod:`scripts.plot_trajectory_figure` — several of those panels composed into
  a single paper figure, which imports the drawing helpers below.

Everything that draws is written against an ``Axes`` so both can share it. The
figure geometry comes from :mod:`scripts.common.figures`, so the saved PDF is
exactly as wide as its LaTeX slot and never gets rescaled on the page.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import bluesky as bs
import numpy as np
import pandas as pd
import pyproj
from bluesky.tools.position import Position
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.colors import FuncNorm, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.plot import plotting_extent
from tqdm import tqdm

from bluesky_gym.envs.common import functions as fn
from bluesky_gym.maps.map_sources import MapSourceConfigType, TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler
from scripts.common import figures as fg
from scripts.common.run_paths import resolve_run, RunPaths, load_trajectory_details

from scripts.common.colors import *

# Successful-approach arc (the SINK polyline in BaseNavigationEnv._set_terminal_condition):
# crossing it terminates the episode as "success". Hard-coded to match all recent
# BaseNavigationEnv runs (config.yaml: faf_distance=0, iaf_angle=60, iaf_distance=37).
FAF_DISTANCE_KM = 0.0
IAF_ANGLE_DEG = 60.0
IAF_DISTANCE_KM = 37.0
ARC_NUM_POINTS = 36

PLOT_CRS = "epsg:3035"
BACKGROUND_PIXELS = 512
MARGIN_M = 25_000.0            # padding between the outermost trajectory and the frame
TICK_STEPS_KM = (10, 25, 50, 100, 200, 250, 500)
# Trajectories are drawn with a thin white halo: black lines vanish over the
# dark end of the density colormap, which is exactly where they matter. Set to
# 0 to go back to plain lines.
TRAJECTORY_HALO_LW = 1.8
# Terminal geometry. The wedge is filled so it stays distinguishable from the
# (also solid, also red) failed trajectories that cover it; the fill has to stay
# light enough not to shift the density colours underneath it.
GEOMETRY_LW = 1.5
GEOMETRY_HALO_LW = 1.4
GEOMETRY_FILL_ALPHA = 0.3


@dataclass
class TrajectoryPanel:
    """One map panel: a set of trajectories and the destination it is centred on.

    ``center`` is filled in by :func:`project_panel`, which also adds the
    projected ``x``/``y`` columns to ``trajectories``.
    """

    trajectories: pd.DataFrame
    destination: Position
    caption: str = ""
    center: tuple[float, float] = (0.0, 0.0)


def load_panel(traj_dir: Path, caption: str = "") -> TrajectoryPanel | None:
    """Read a trajectory subdirectory (trajectories.csv + details) into a panel."""
    csv_path = traj_dir / "trajectories.csv"
    details = load_trajectory_details(traj_dir)
    if not csv_path.exists() or details is None:
        return None

    runway = Position(name=details["runway"], reflat=0, reflon=0)
    if details.get("destination_latlon", None):
        runway.lat = details["destination_latlon"][0]
        runway.lon = details["destination_latlon"][1]

    return TrajectoryPanel(pd.read_csv(csv_path), runway, caption)


def make_transformer(destination_crs: str = PLOT_CRS) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs("WGS84", destination_crs, always_xy=True)


def project_panel(panel: TrajectoryPanel, transformer: pyproj.Transformer) -> TrajectoryPanel:
    """Project lat/lon to the plot CRS and record the destination as the centre."""
    df = panel.trajectories
    df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)
    panel.center = transformer.transform(panel.destination.lon, panel.destination.lat)
    return panel


def panel_half_width(panel: TrajectoryPanel, margin_m: float = MARGIN_M) -> float:
    """Half-width of the smallest square window around the destination that fits.

    Square and destination-centred on purpose: panels sharing one half-width are
    at the same km-per-inch, which is what makes two of them comparable side by
    side.
    """
    cx, cy = panel.center
    dx = (panel.trajectories["x"] - cx).abs().max()
    dy = (panel.trajectories["y"] - cy).abs().max()
    return float(max(dx, dy)) + margin_m


def build_norm(normalization_mode: str, v_max: float):
    if not np.isfinite(v_max) or v_max <= 0:
        v_max = 1.0  # empty/zero window (e.g. ocean or all-zero synthetic map)
    if normalization_mode == "log":
        return FuncNorm(functions=(np.log1p, np.expm1), vmin=0, vmax=v_max)
    if normalization_mode in ("min_max", "min-max"):
        return Normalize(vmin=0, vmax=v_max, clip=True)
    raise ValueError(f"Unknown normalization_mode: {normalization_mode!r}")


def density_cmap():
    cmap = plt.get_cmap(HEATMAP_COLORS).copy()
    cmap.set_bad(BACKGROUND_COLOR)  # NaN pixels (no-data / ocean) render grey instead of transparent
    return cmap


def draw_background(ax, raster_sampler: RasterSampler, center: tuple[float, float],
                    half_width_m: float, norm, cmap, pixels: int = BACKGROUND_PIXELS):
    """Draw the density raster for the square window around ``center``."""
    cx, cy = center
    bounds = (cx - half_width_m, cy - half_width_m, cx + half_width_m, cy + half_width_m)
    background = raster_sampler.get_background(*bounds, width=pixels, height=pixels)
    transform = raster_sampler.get_dst_transform_from_bounds(*bounds, width=pixels, height=pixels)
    extent = plotting_extent(background, transform)

    data = np.where(np.isfinite(background) & (background >= 0), background, np.nan)
    im = ax.imshow(data, extent=extent, origin="upper", cmap=cmap, norm=norm)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return im


def draw_trajectories(ax, panel: TrajectoryPanel) -> None:
    """Draw one line per start angle: black = success, red = failure, both solid."""
    df = panel.trajectories
    if "termination_reason" not in df.columns:
        print("No 'termination_reason' column found. Assuming 'success' everywhere")

    for _, group in df.groupby("start_angle"):
        reason = group["termination_reason"].iloc[0] if "termination_reason" in df.columns else "success"
        success = reason == "success"
        ax.plot(group["x"], group["y"],
                color=TRAJECTORY_COLOR if success else RESTRICT_COLOR,
                linewidth=0.9, zorder=4)


def draw_terminal_geometry(ax, destination: Position, transformer: pyproj.Transformer) -> None:
    """Draw the approach wedge: success arc and failed-approach boundary (SINK geometry).

    Same construction as ``BaseNavigationEnv._set_terminal_condition``.

    The wedge is filled, not just outlined. Failed trajectories are solid red
    too, so the geometry cannot be told apart from the data by colour or dash
    pattern alone — an area versus a stroke reads unambiguously even where the
    two overlap, which they do exactly where the wedge matters. The white
    casing keeps the outline visible under a pile of trajectories.
    """
    destination_xy = transformer.transform(destination.lon, destination.lat)
    back_bearing = fn.bound_angle_0_360(destination.refhdg + 180)
    faf_lat, faf_lon = fn.get_point_at_distance(destination.lat, destination.lon, FAF_DISTANCE_KM, back_bearing)
    arc_angles = np.linspace(back_bearing + IAF_ANGLE_DEG / 2, back_bearing - IAF_ANGLE_DEG / 2, ARC_NUM_POINTS)
    arc_lat, arc_lon = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE_KM, arc_angles)
    arc_x, arc_y = transformer.transform(arc_lon, arc_lat)

    casing = [patheffects.withStroke(linewidth=GEOMETRY_LW + GEOMETRY_HALO_LW,
                                     foreground="white", alpha=0.8)]
    ax.fill(np.concatenate(([destination_xy[0]], arc_x)),
            np.concatenate(([destination_xy[1]], arc_y)),
            facecolor=SINK_COLOR, alpha=GEOMETRY_FILL_ALPHA, edgecolor="none", zorder=3)
    ax.plot([arc_x[0], destination_xy[0], arc_x[-1]], [arc_y[0], destination_xy[1], arc_y[-1]],
            color=RESTRICT_COLOR, linewidth=GEOMETRY_LW, zorder=5, path_effects=casing)
    ax.plot(arc_x, arc_y, color=SINK_COLOR, linewidth=GEOMETRY_LW, zorder=5, path_effects=casing)
    ax.scatter(*destination_xy, marker=".", linewidths=3, color=TRAJECTORY_COLOR, zorder=6)


def legend_handles(include_failed_trajectory:bool=True) -> list:
    """Handles for a legend shared by every trajectory panel.

    The wedge entry is a patch so it matches how the geometry is drawn: the
    trajectory entries are the only lines in the legend, which is what keeps
    them apart from the geometry now that both are solid red.
    """
    if include_failed_trajectory:
        return [
        Line2D([], [], color=TRAJECTORY_COLOR, linewidth=0.9, label="Successful trajectory"),
        Line2D([], [], color=RESTRICT_COLOR, linewidth=0.9, label="Failed trajectory"),
        Line2D([], [], color=SINK_COLOR, linewidth=1.5 * GEOMETRY_LW, label="Success arc"),
        Line2D([], [], color=RESTRICT_COLOR, linewidth=1.5 * GEOMETRY_LW, label="Failure radial"),
    ]
    else:
        return [
            Line2D([], [], color=TRAJECTORY_COLOR, linewidth=0.9, label="Successful trajectory"),
            Line2D([], [], color=SINK_COLOR, linewidth=1.5 * GEOMETRY_LW, label="Success arc"),
            Line2D([], [], color=RESTRICT_COLOR, linewidth=1.5 * GEOMETRY_LW, label="Failure radial"),
        ]


def _nice_step_km(half_width_m: float, max_ticks_per_side: int = 2) -> float:
    half_km = half_width_m / 1000
    for step in TICK_STEPS_KM:
        if half_km / step <= max_ticks_per_side:
            return step
    return TICK_STEPS_KM[-1]


def km_ticks(ax, center: tuple[float, float], half_width_m: float,
             show_x: bool = True, show_y: bool = True,
             step_km: float | None = None) -> None:
    """Ticks every ``step_km`` km, labelled relative to the destination.

    ``step_km`` defaults to a nice round number for the window; pass the
    evaluation start distance instead to put the ticks on a distance that means
    something — the ring the aircraft spawn on.

    Ticks are drawn on every panel of a grid; only the outer ones get labels, so
    the panels still read as sharing one coordinate system.
    """
    step = step_km if step_km else _nice_step_km(half_width_m)
    cx, cy = center
    n = int(half_width_m / 1000 // step)
    offsets = np.arange(-n, n + 1) * step
    labels = [f"{offset:g}" for offset in offsets]

    ax.set_xticks(cx + offsets * 1000)
    ax.set_yticks(cy + offsets * 1000)
    ax.set_xticklabels(labels if show_x else [])
    ax.set_yticklabels(labels if show_y else [])


def scale_bar(ax, half_width_m: float, length_km: float | None = None,
              x0: float = 0.06, y0: float = 0.07) -> None:
    """Scale bar in the lower-left of ``ax``, for panels drawn without axes."""
    if length_km is None:
        length_km = _nice_step_km(half_width_m)
    fraction = (length_km * 1000) / (2 * half_width_m)
    halo = [patheffects.withStroke(linewidth=3.0, foreground="white", alpha=0.8)]
    ax.plot([x0, x0 + fraction], [y0, y0], transform=ax.transAxes, color="black",
            linewidth=1.5, solid_capstyle="butt", zorder=7, path_effects=halo)
    ax.text(x0 + fraction / 2, y0 + 0.02, f"{length_km:g} km", transform=ax.transAxes,
            ha="center", va="bottom", size=plt.rcParams["xtick.labelsize"], zorder=7,
            path_effects=halo)


DENSITY_LABEL = r"Population density [ppl/km$^2$]"


def density_label(available_in: float) -> str:
    """Longest density label that fits along a colorbar ``available_in`` tall.

    The label is rotated, so its length is bounded by the figure *height* — and
    a label that runs off the top is silently clipped rather than resized.
    """
    return DENSITY_LABEL


def add_colorbar(fig, im, cax, normalization_mode: str, v_max: float, label: str | None = None):
    """Density colorbar.

    Density is clipped at ``v_max`` (the normalization percentile of the map):
    everything above it saturates to the darkest colour. The 'max' extend arrow
    plus the labelled top tick make that clip explicit.
    """
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(label if label is not None else DENSITY_LABEL)
    if normalization_mode == "log":
        nice_ticks = np.array([0, 1, 10, 100, 1_000, 10_000, 100_000], dtype=float)
        ticks = [t for t in nice_ticks if t < v_max] + [v_max]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
    return cbar


def plot_trajectories(
        trajectories: pd.DataFrame,
        map_config: MapSourceConfigType,
        destination: Position,
        save_path: Path | None = None,
        normalization_mode: str = "min_max",
        normalization_percentile: float = 99.9,
        range_km: float | None = 250,
        width: float = 0.33,
):
    """Single-panel trajectory plot, sized for a ``0.33\\textwidth`` LaTeX slot.

    ``range_km`` fixes the half-width of the window (plus a margin) so that every
    run plotted with the same evaluation start distance comes out at the same
    scale; pass ``None`` to fit the window to the data instead.
    """
    map_source = map_config.build()
    raster_sampler = RasterSampler(map_source, resampling="cubic_spline", destination_crs=PLOT_CRS)
    v_max = map_source.get_normalization_value(normalization_percentile)
    transformer = make_transformer(raster_sampler.destination_crs)

    panel = project_panel(TrajectoryPanel(trajectories, destination), transformer)
    half_width_m = range_km * 1000 + MARGIN_M if range_km else panel_half_width(panel)
    norm = build_norm(normalization_mode, v_max)

    # right/top leave room for the outermost tick label, which sticks out past
    # the axes box by about half its width.
    fig, axes = fg.paper_grid(1, 1, width=width, panel_aspect=1.0,
                              left=0.58, right=0.18, bottom=0.46, top=0.10)
    ax = axes[0, 0]

    draw_background(ax, raster_sampler, panel.center, half_width_m, norm, density_cmap())
    draw_trajectories(ax, panel)
    draw_terminal_geometry(ax, destination, transformer)
    km_ticks(ax, panel.center, half_width_m, step_km=range_km)
    fg.grid_labels(fig, axes, xlabel=r"$x$-coordinate [km]", ylabel=r"$y$-coordinate [km]")

    if save_path is not None:
        fg.save(fig, save_path)
    plt.close(fig)


def plot_trajectory_subdir(traj_dir: Path, background_map: Path, normalization_percentile: float,
                           normalization_mode: str) -> None:
    """Plot a single trajectory subdirectory (contains trajectories.csv + details)."""
    panel = load_panel(traj_dir)
    if panel is None:
        print(f"Skipping {traj_dir} — missing trajectories.csv or details")
        return

    details = load_trajectory_details(traj_dir) or {}
    # Always use the real population map as the plot background, even for runs where
    # the agent flew without a population map in its observation.
    map_config = TiffMapSourceConfig(file_path=background_map)

    save_path = traj_dir / "plot.pdf"
    if save_path.exists():
        print(f"Overwriting existing plot: {save_path}")

    plot_trajectories(panel.trajectories, map_config, destination=panel.destination,
                      save_path=save_path, normalization_percentile=normalization_percentile,
                      normalization_mode=normalization_mode,
                      range_km=details.get("start_distance", 250))


def present_for_run(run_paths: RunPaths, background_map: Path, normalization_percentile: float,
                    normalization_mode: str) -> None:
    """Plot all trajectory subdirectories for a run (searches recursively)."""
    if not run_paths.trajectories_dir.exists():
        print(f"No trajectories found for {run_paths.run_id}")
        return

    # Find all directories containing a trajectories.csv + details (json or legacy pkl) pair
    for csv_path in sorted(run_paths.trajectories_dir.rglob("trajectories.csv")):
        traj_dir = csv_path.parent
        if (traj_dir / "details.json").exists() or (traj_dir / "details.pkl").exists():
            plot_trajectory_subdir(traj_dir, background_map=background_map,
                                   normalization_percentile=normalization_percentile,
                                   normalization_mode=normalization_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trajectories for trained run(s). PDFs are saved to the CSV directory.")
    parser.add_argument("run_refs", nargs="+", help="Run reference(s) or path to a trajectories.csv")
    parser.add_argument("--background_map_path", type=str, default="./scripts/population_maps/europe_3035_1km.tif",
                        help="Path to map to use as the background of the plots.")
    parser.add_argument("--normalization_percentile", type=float, default=99.9)
    parser.add_argument("--normalization_mode", type=str, default="log")
    args = parser.parse_args()

    bs.init()

    # Legacy: if a single arg is a CSV file, plot that directly
    if len(args.run_refs) == 1 and args.run_refs[0].endswith(".csv"):
        csv_path = Path(args.run_refs[0])
        plot_trajectory_subdir(csv_path.parent, Path(args.background_map_path),
                               normalization_percentile=args.normalization_percentile,
                               normalization_mode=args.normalization_mode)
    else:
        runs = [resolve_run(r) for r in args.run_refs]
        for run_path in tqdm(runs, desc="Runs"):
            print(f"\nPlotting trajectories for: {run_path.run_id}")
            present_for_run(run_path, Path(args.background_map_path), normalization_mode=args.normalization_mode,
                            normalization_percentile=args.normalization_percentile)
