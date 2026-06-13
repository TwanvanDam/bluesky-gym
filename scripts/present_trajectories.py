import argparse
from pathlib import Path

import bluesky as bs
import numpy as np
import pandas as pd
import pyproj
from bluesky.tools.position import Position
from matplotlib import pyplot as plt
from matplotlib.colors import FuncNorm, Normalize
from rasterio.plot import plotting_extent
from tqdm import tqdm

from bluesky_gym.envs.common import functions as fn
from bluesky_gym.maps.map_sources import MapSourceConfigType, TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler
from scripts.common.run_paths import resolve_run, RunPaths, load_trajectory_details

# Successful-approach arc (the SINK polyline in BaseNavigationEnv._set_terminal_condition):
# crossing it terminates the episode as "success". Hard-coded to match all recent
# BaseNavigationEnv runs (config.yaml: faf_distance=0, iaf_angle=60, iaf_distance=37).
FAF_DISTANCE_KM = 0.0
IAF_ANGLE_DEG = 60.0
IAF_DISTANCE_KM = 37.0
ARC_NUM_POINTS = 36


def plot_trajectories(
        trajectories: pd.DataFrame,
        map_config: MapSourceConfigType,
        destination: Position,
        save_path: Path | None = None,
        normalization_mode: str = "min_max",
        normalization_percentile: float = 99.9,
):
    map_source = map_config.build()
    raster_sampler = RasterSampler(map_source, resampling="cubic_spline", destination_crs="epsg:3035")
    v_max = map_source.get_normalization_value(normalization_percentile)
    coordinate_transformer = pyproj.Transformer.from_crs("WGS84", raster_sampler.destination_crs, always_xy=True)
    destination_xy = coordinate_transformer.transform(destination.lon, destination.lat)

    trajectories["x"], trajectories["y"] = coordinate_transformer.transform(trajectories["lon"].values,
                                                                            trajectories["lat"].values)

    x_min = - 25_000 + trajectories["x"].min()
    x_max = 25_000 + trajectories["x"].max()
    y_min = -25_000 + trajectories["y"].min()
    y_max = 25_000 + trajectories["y"].max()

    background = raster_sampler.get_background(x_min, y_min, x_max, y_max, width=512, height=512)
    background_transform = raster_sampler.get_dst_transform_from_bounds(x_min, y_min, x_max, y_max, width=512,
                                                                        height=512)
    extent = plotting_extent(background, background_transform)

    background_data = np.where(np.isfinite(background) & (background >= 0), background, np.nan)

    if not np.isfinite(v_max) or v_max <= 0:
        vmax = 1.0  # empty/zero window (e.g. ocean or all-zero synthetic map)

    if normalization_mode == "log":
        norm = FuncNorm(functions=(np.log1p, np.expm1), vmin=0, vmax=v_max)
    elif normalization_mode in ["min_max", "min-max"]:
        norm = Normalize(vmin=0, vmax=v_max, clip=True)
    else:
        raise ValueError(f"Unknown normalization_mode: {normalization_mode!r}")

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("grey")  # NaN pixels (no-data / ocean) render grey instead of transparent

    im = plt.imshow(
        background_data,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm
    )

    # Density is clipped at vmax (the normalization_percentile of the window): everything
    # above it saturates to the darkest colour. The 'max' extend arrow plus the labelled
    # top tick make that clip explicit.
    cbar = plt.colorbar(im, extend="max", fraction=0.046, pad=0.04)
    cbar.set_label("Population density (people/km²)")
    if normalization_mode == "log":
        nice_ticks = np.array([0, 1, 10, 100, 1_000, 10_000, 100_000], dtype=float)
        ticks = [t for t in nice_ticks if t < v_max] + [v_max]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.scatter(*destination_xy, marker=".", linewidths=5, color="green")


    for start_angle, group in trajectories.groupby("start_angle"):
        color = "black" if group["termination_reason"].iloc[0] == "success" else "red"
        plt.plot(group["x"], group["y"], color=color)
        plt.plot(group["x"].iloc[0], group["y"].iloc[0], marker="o", color="green", linewidth=1,
                 label="Start" if start_angle == trajectories["start_angle"].min() else "")

    # Successful-approach arc (SINK): same geometry as BaseNavigationEnv._set_terminal_condition.
    back_bearing = fn.bound_angle_0_360(destination.refhdg + 180)
    faf_lat, faf_lon = fn.get_point_at_distance(destination.lat, destination.lon, FAF_DISTANCE_KM, back_bearing)
    arc_angles = np.linspace(back_bearing + IAF_ANGLE_DEG / 2, back_bearing - IAF_ANGLE_DEG / 2, ARC_NUM_POINTS)
    arc_lat, arc_lon = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE_KM, arc_angles)
    arc_x, arc_y = coordinate_transformer.transform(arc_lon, arc_lat)
    arc_x = [destination_xy[0], * arc_x, destination_xy[0]]
    arc_y = [destination_xy[1], * arc_y, destination_xy[1]]
    plt.plot(arc_x, arc_y, color="green", linewidth=2, label="Success arc")

    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.close()


def plot_trajectory_subdir(traj_dir: Path, background_map: Path, normalization_percentile:float, normalization_mode: str) -> None:
    """Plot a single trajectory subdirectory (contains trajectories.csv + details.pkl)."""
    csv_path = traj_dir / "trajectories.csv"
    eval_details = load_trajectory_details(traj_dir)
    if not csv_path.exists() or eval_details is None:
        print(f"Skipping {traj_dir} — missing trajectories.csv or details")
        return

    df = pd.read_csv(csv_path)

    runway = Position(name=eval_details["runway"], reflat=0, reflon=0)
    if eval_details.get("destination_latlon", None):
        runway.lat = eval_details["destination_latlon"][0]
        runway.lon = eval_details["destination_latlon"][1]

    agent_used_map = eval_details["map_path"] is not None
    # Always use the real population map as the plot background, even for runs where
    # the agent flew without a population map in its observation.
    map_config = TiffMapSourceConfig(file_path=background_map)

    save_path = traj_dir / f"plot.png"

    if save_path.exists():
        print(f"Overwriting existing plot: {save_path}")

    plot_trajectories(df, map_config, destination=runway, save_path=save_path,
                      normalization_percentile=normalization_percentile, normalization_mode=normalization_mode)


def present_for_run(run_paths: RunPaths, background_map: Path, normalization_percentile:float, normalization_mode:str) -> None:
    """Plot all trajectory subdirectories for a run (searches recursively)."""
    if not run_paths.trajectories_dir.exists():
        print(f"No trajectories found for {run_paths.run_id}")
        return

    # Find all directories containing a trajectories.csv + details (json or legacy pkl) pair
    for csv_path in sorted(run_paths.trajectories_dir.rglob("trajectories.csv")):
        traj_dir = csv_path.parent
        if (traj_dir / "details.json").exists() or (traj_dir / "details.pkl").exists():
            plot_trajectory_subdir(traj_dir, background_map=background_map, normalization_percentile=normalization_percentile,
                                   normalization_mode=normalization_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trajectories for trained run(s). PNG's are saved to CSV directory.")
    parser.add_argument("run_refs", nargs="+", help="Run reference(s) or path to a trajectories.csv")
    parser.add_argument("--background_map_path", type=str, default="./scripts/population_maps/europe_3035_1km.tif",
                        help="Path to map to use as the background of the plots.")
    parser.add_argument("--normalization_percentile", type=float, default=99.9)
    parser.add_argument("--normalization_mode", type=str, default="log")
    args = parser.parse_args()

    # Legacy: if a single arg is a CSV file, plot that directly
    if len(args.run_refs) == 1 and args.run_refs[0].endswith(".csv"):
        csv_path = Path(args.run_refs[0])
        plot_trajectory_subdir(csv_path.parent, Path(args.background_map_path), args.normalization_mode, args.normalization_percentile)
    else:
        runs = [resolve_run(r) for r in args.run_refs]
        bs.init()
        for run_path in tqdm(runs, desc="Runs"):
            print(f"\nPlotting trajectories for: {run_path.run_id}")
            present_for_run(run_path, Path(args.background_map_path), normalization_mode=args.normalization_mode,
                            normalization_percentile=args.normalization_percentile)
