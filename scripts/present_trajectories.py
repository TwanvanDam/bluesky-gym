import argparse
import pickle
from pathlib import Path

import bluesky as bs
import numpy as np
import pandas as pd
import pyproj
from bluesky.tools.position import Position
from matplotlib import pyplot as plt
from rasterio.plot import plotting_extent
from tqdm import tqdm

from bluesky_gym.maps.map_sources import MapSourceConfigType, TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler
from scripts.common.run_paths import resolve_run, RunPaths


def plot_trajectories(
        trajectories: pd.DataFrame,
        map_config: MapSourceConfigType,
        runway: str = "EHAM/RW27",
        run_name: str = "",
        agent_used_map: bool = False,
        save_path: Path | None = None,
):
    map_source = map_config.build()
    raster_sampler = RasterSampler(map_source, resampling="cubic_spline", destination_crs="epsg:3035")
    destination = Position(name=runway, reflat=0, reflon=0)

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

    plt.imshow(background, extent=extent, origin="upper", cmap="Blues", vmin=0, vmax=np.nanpercentile(background, 99))
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.scatter(*destination_xy, marker=".", linewidths=5)

    for start_angle, group in trajectories.groupby("start_angle"):
        color = "black" if group["termination_reason"].iloc[0] == "success" else "red"
        plt.plot(group["x"], group["y"], color=color)
        plt.plot(group["x"].iloc[0], group["y"].iloc[0], marker="o", color="green",
                 label="Start" if start_angle == trajectories["start_angle"].min() else "")
    map_label = "with map" if agent_used_map else "no map"
    plt.title(f"{run_name} | runway: {runway} | {map_label}")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.close()


DEFAULT_BACKGROUND_MAP_PATH = Path(__file__).parent / "population_maps" / "ESTAT_OBS-VALUE-T_2021_V2.tiff"


def plot_trajectory_subdir(traj_dir: Path, run_name: str = "") -> None:
    """Plot a single trajectory subdirectory (contains trajectories.csv + details.pkl)."""
    csv_path = traj_dir / "trajectories.csv"
    details_path = traj_dir / "details.pkl"
    if not csv_path.exists() or not details_path.exists():
        print(f"Skipping {traj_dir} — missing trajectories.csv or details.pkl")
        return

    df = pd.read_csv(csv_path)
    with open(details_path, "rb") as f:
        eval_details = pickle.load(f)

    runway = eval_details["runway"]
    agent_used_map = eval_details["map_path"] is not None
    # Always use the real population map as the plot background, even for runs where
    # the agent flew without a population map in its observation.
    background_map_path = eval_details["map_path"] or DEFAULT_BACKGROUND_MAP_PATH
    map_config = TiffMapSourceConfig(file_path=background_map_path)

    save_path = traj_dir / f"plot.png"

    if save_path.exists():
        print(f"Plot already exists, skipping: {save_path}")
        return

    plot_trajectories(df, map_config, runway=runway, run_name=run_name, agent_used_map=agent_used_map, save_path=save_path)


def present_for_run(run_paths: RunPaths) -> None:
    """Plot all trajectory subdirectories for a run (searches recursively)."""
    if not run_paths.trajectories_dir.exists():
        print(f"No trajectories found for {run_paths.run_id}")
        return

    # Find all directories containing a trajectories.csv + details.pkl pair
    for csv_path in sorted(run_paths.trajectories_dir.rglob("trajectories.csv")):
        traj_dir = csv_path.parent
        if (traj_dir / "details.pkl").exists():
            plot_trajectory_subdir(traj_dir, run_name=run_paths.run_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trajectories for trained run(s).")
    parser.add_argument("run_refs", nargs="+", help="Run reference(s) or path to a trajectories.csv")
    args = parser.parse_args()

    # Legacy: if a single arg is a CSV file, plot that directly
    if len(args.run_refs) == 1 and args.run_refs[0].endswith(".csv"):
        csv_path = Path(args.run_refs[0])
        plot_trajectory_subdir(csv_path.parent)
    else:
        runs = [resolve_run(r) for r in args.run_refs]
        bs.init()
        for run_paths in tqdm(runs, desc="Runs"):
            print(f"\nPlotting trajectories for: {run_paths.run_id}")
            present_for_run(run_paths)
