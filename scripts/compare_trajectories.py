"""Compare fuel and noise metrics between two runs.

For each trajectory config present in both runs, generates a figure with:
  - Left panel:  both runs' trajectories overlaid on the population map
  - Top right:   absolute fuel / noise per start angle (grouped bars)
  - Bottom right: difference (run_b − run_a) per start angle (signed bars)

Usage:
    python -m scripts.compare_trajectories <run_a> <run_b>
    python -m scripts.compare_trajectories <run_a> <run_b> --out comparisons/my_comparison
"""

import argparse
import pickle
from pathlib import Path

import bluesky as bs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from bluesky.tools.position import Position
from rasterio.plot import plotting_extent

from bluesky_gym.maps.map_sources import TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler, MapObservationConfig
from bluesky_gym.metrics.fuel_model import FuelModel
from bluesky_gym.metrics.noise_model import NoiseConfig
from scripts.common.run_paths import resolve_run, RunPaths

MAP_PATH = "/home/twanvandam/Thesis/scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff"


def build_metric_fn():
    """Initialise models once and return a calculate_metrics(df) function."""
    raster_sampler = RasterSampler(
        map_source=TiffMapSourceConfig(file_path=MAP_PATH).build(),
        resampling="cubic_spline",
        destination_crs="epsg:3035",
    )
    noise_model = NoiseConfig().build()
    fuel_model = FuelModel("a320")

    def _fuel(altitude, tas, sim_dt, mass):
        return fuel_model.step_fuel_flow(mass=mass, tas=tas, altitude=altitude) * sim_dt

    def _noise(lat, lon, altitude, sim_dt):
        pos = Position(name=f"{lat},{lon}", reflat=0, reflon=0)
        k_m, k_px = noise_model.get_noise_power_kernel_shape_meters_and_pixels(altitude)

        noise_kernel_map_extract_config = MapObservationConfig(shape=k_px, range=k_m)
        pop = raster_sampler.get_observation_clipped(center_position=pos, orientation=0,
                                                     observation_config=noise_kernel_map_extract_config)
        return noise_model.step_total_noise(pop, altitude, sim_dt)

    def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
        alt_key = "altitude" if "altitude" in df.columns else "alt"
        df["calculated_fuel"] = df.apply(
            lambda r: _fuel(r[alt_key], r["tas"], r["sim_dt"], r["mass"]), axis=1
        )
        df["calculated_noise"] = df.apply(
            lambda r: _noise(r["lat"], r["lon"], r[alt_key], r["sim_dt"]), axis=1
        )
        return df

    return calculate_metrics


def _load(csv_path: Path, calculate_metrics) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["start_angle"] = ((df["start_angle"] / 10).round() * 10).astype(int)
    return calculate_metrics(df)


def _add_trajectory_overlay(
        ax: plt.Axes,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        label_a: str,
        label_b: str,
        details: dict,
) -> None:
    """Overlay trajectories from two runs on the population map."""
    map_path = details.get("map_path", MAP_PATH)
    runway = details.get("runway", "EHAM/RW27")

    map_source = TiffMapSourceConfig(file_path=map_path).build()
    rs = RasterSampler(map_source, resampling="cubic_spline", destination_crs="epsg:3035")
    transformer = pyproj.Transformer.from_crs("WGS84", rs.destination_crs, always_xy=True)

    destination = Position(name=runway, reflat=0, reflon=0)
    dest_xy = transformer.transform(destination.lon, destination.lat)

    for df in (df_a, df_b):
        df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)

    all_x = pd.concat([df_a["x"], df_b["x"]])
    all_y = pd.concat([df_a["y"], df_b["y"]])
    print(f"{label_a} starting positions (x, y): {df_a.groupby('start_angle').first()[['x', 'y']].values}")
    print(f"{label_b} starting positions (x, y): {df_b.groupby('start_angle').first()[['x', 'y']].values}")
    print(f"{label_a} end positions (x, y): {df_a.groupby('start_angle').last()[['x', 'y']].values}")
    print(f"{label_b} end positions (x, y): {df_b.groupby('start_angle').last()[['x', 'y']].values}")
    print(
        f"{label_a} destination start to end distances (m): {np.sqrt((df_a.groupby('start_angle').first()['x'] - dest_xy[0]) ** 2 + (df_a.groupby('start_angle').first()['y'] - dest_xy[1]) ** 2).values}")
    print(
        f"{label_b} destination start to end distances (m): {np.sqrt((df_b.groupby('start_angle').first()['x'] - dest_xy[0]) ** 2 + (df_b.groupby('start_angle').first()['y'] - dest_xy[1]) ** 2).values}")

    x_min = all_x.min() - 25_000
    x_max = all_x.max() + 25_000
    y_min = all_y.min() - 25_000
    y_max = all_y.max() + 25_000

    background = rs.get_background(x_min, y_min, x_max, y_max, width=512, height=512)
    bg_transform = rs.get_dst_transform_from_bounds(x_min, y_min, x_max, y_max, width=512, height=512)
    extent = plotting_extent(background, bg_transform)

    ax.imshow(background, extent=extent, origin="upper", cmap="Blues",
              vmin=0, vmax=np.nanpercentile(background, 99))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.scatter(*dest_xy, marker="x", s=60, color="black", zorder=5)

    for i, (df, color, label) in enumerate(
            [(df_a, "tab:orange", label_a), (df_b, "limegreen", label_b)]
    ):
        for j, (_, group) in enumerate(df.groupby("start_angle")):
            ax.plot(group["x"], group["y"], color=color, linewidth=2.0,
                    label=label if j == 0 else None)

    ax.legend(fontsize=7)
    ax.set_title(f"Trajectories — {runway}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")


def _draw_fuel_abs(ax, x, angles, fuel_a, fuel_b, label_a, label_b):
    w = 0.4
    ax.bar([i - w / 2 for i in x], fuel_a.values, width=w, label=label_a, color="tab:orange")
    ax.bar([i + w / 2 for i in x], fuel_b.values, width=w, label=label_b, color="limegreen")
    ax.set_xticks(list(x))
    ax.set_xticklabels(angles, rotation=45)
    ax.set_title("Fuel consumption (kg)")
    ax.set_xlabel("Start angle (deg)")
    ax.legend(fontsize=7)


def _draw_noise_abs(ax, x, angles, noise_a, noise_b, label_a, label_b):
    w = 0.4
    ax.bar([i - w / 2 for i in x], noise_a.values, width=w, label=label_a, color="tab:orange")
    ax.bar([i + w / 2 for i in x], noise_b.values, width=w, label=label_b, color="limegreen")
    ax.set_xticks(list(x))
    ax.set_xticklabels(angles, rotation=45)
    ax.set_title("Noise exposure")
    ax.set_xlabel("Start angle (deg)")
    ax.legend(fontsize=7)


def _draw_fuel_diff(ax, x, angles, fuel_diff, label_a, label_b):
    colors_fuel = ["tab:red" if v > 0 else "tab:green" for v in fuel_diff.values]
    ax.bar(list(x), fuel_diff.values, color=colors_fuel)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(angles, rotation=45)
    ax.set_title(f"Fuel diff  ({label_b} − {label_a})")
    ax.set_xlabel("Start angle (deg)")
    ax.set_ylabel("Δ kg")


def _draw_noise_diff(ax, x, angles, noise_diff, label_a, label_b):
    colors_noise = ["tab:red" if v > 0 else "tab:green" for v in noise_diff.values]
    ax.bar(list(x), noise_diff.values, color=colors_noise)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(angles, rotation=45)
    ax.set_title(f"Noise diff  ({label_b} − {label_a})")
    ax.set_xlabel("Start angle (deg)")


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_comparison(
        csv_a: Path,
        csv_b: Path,
        label_a: str,
        label_b: str,
        save_path: Path,
        calculate_metrics,
) -> None:
    df_a = _load(csv_a, calculate_metrics)
    df_b = _load(csv_b, calculate_metrics)

    fuel_a = df_a.groupby("start_angle")["calculated_fuel"].sum()
    fuel_b = df_b.groupby("start_angle")["calculated_fuel"].sum()
    noise_a = df_a.groupby("start_angle")["calculated_noise"].sum()
    noise_b = df_b.groupby("start_angle")["calculated_noise"].sum()

    angles = sorted(set(fuel_a.index) | set(fuel_b.index))
    fuel_a = fuel_a.reindex(angles, fill_value=0)
    fuel_b = fuel_b.reindex(angles, fill_value=0)
    noise_a = noise_a.reindex(angles, fill_value=0)
    noise_b = noise_b.reindex(angles, fill_value=0)

    fuel_diff = fuel_b - fuel_a
    noise_diff = noise_b - noise_a

    # Load details for map rendering (optional)
    details_path = csv_a.parent / "details.pkl"
    details = {}
    if details_path.exists():
        with open(details_path, "rb") as f:
            details = pickle.load(f)

    x = range(len(angles))
    stem = save_path.stem
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Combined figure ---
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f"{label_a}  vs  {label_b}", fontsize=9)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.4, 1, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_fuel_abs = fig.add_subplot(gs[0, 1])
    ax_noise_abs = fig.add_subplot(gs[0, 2])
    ax_fuel_diff = fig.add_subplot(gs[1, 1])
    ax_noise_diff = fig.add_subplot(gs[1, 2])

    _add_trajectory_overlay(ax_map, df_a, df_b, label_a, label_b, details)
    _draw_fuel_abs(ax_fuel_abs, x, angles, fuel_a, fuel_b, label_a, label_b)
    _draw_noise_abs(ax_noise_abs, x, angles, noise_a, noise_b, label_a, label_b)
    _draw_fuel_diff(ax_fuel_diff, x, angles, fuel_diff, label_a, label_b)
    _draw_noise_diff(ax_noise_diff, x, angles, noise_diff, label_a, label_b)

    fig.tight_layout()
    _save_fig(fig, save_path)

    # --- Individual panel figures ---
    fig_map, ax = plt.subplots(figsize=(8, 8))
    fig_map.suptitle(f"{label_a}  vs  {label_b}", fontsize=9)
    _add_trajectory_overlay(ax, df_a, df_b, label_a, label_b, details)
    fig_map.tight_layout()
    _save_fig(fig_map, save_path.parent / f"{stem}_map.png")

    fig_fa, ax = plt.subplots(figsize=(7, 5))
    fig_fa.suptitle(f"{label_a}  vs  {label_b}", fontsize=9)
    _draw_fuel_abs(ax, x, angles, fuel_a, fuel_b, label_a, label_b)
    fig_fa.tight_layout()
    _save_fig(fig_fa, save_path.parent / f"{stem}_fuel_abs.png")

    fig_na, ax = plt.subplots(figsize=(7, 5))
    fig_na.suptitle(f"{label_a}  vs  {label_b}", fontsize=9)
    _draw_noise_abs(ax, x, angles, noise_a, noise_b, label_a, label_b)
    fig_na.tight_layout()
    _save_fig(fig_na, save_path.parent / f"{stem}_noise_abs.png")

    fig_fd, ax = plt.subplots(figsize=(7, 5))
    fig_fd.suptitle(f"{label_a}  vs  {label_b}", fontsize=9)
    _draw_fuel_diff(ax, x, angles, fuel_diff, label_a, label_b)
    fig_fd.tight_layout()
    _save_fig(fig_fd, save_path.parent / f"{stem}_fuel_diff.png")

    fig_nd, ax = plt.subplots(figsize=(7, 5))
    fig_nd.suptitle(f"{label_a}  vs  {label_b}", fontsize=9)
    _draw_noise_diff(ax, x, angles, noise_diff, label_a, label_b)
    fig_nd.tight_layout()
    _save_fig(fig_nd, save_path.parent / f"{stem}_noise_diff.png")


def compare_runs(run_a: RunPaths, run_b: RunPaths, out_dir: Path, calculate_metrics) -> None:
    if not run_a.trajectories_dir.exists():
        print(f"No trajectories for {run_a.run_id}")
        return
    if not run_b.trajectories_dir.exists():
        print(f"No trajectories for {run_b.run_id}")
        return

    subdirs_a = {p.parent.name: p for p in run_a.trajectories_dir.rglob("trajectories.csv")}
    subdirs_b = {p.parent.name: p for p in run_b.trajectories_dir.rglob("trajectories.csv")}

    common = sorted(set(subdirs_a) & set(subdirs_b))
    if not common:
        print(f"No matching trajectory configs between {run_a.run_id} and {run_b.run_id}")
        return

    for name in common:
        save_path = out_dir / f"{name}.png"
        if save_path.exists():
            print(f"Skipping (already exists): {save_path}")
            continue
        print(f"Comparing: {name}")
        _plot_comparison(
            subdirs_a[name],
            subdirs_b[name],
            label_a=run_a.run_name,
            label_b=run_b.run_name,
            save_path=save_path,
            calculate_metrics=calculate_metrics,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fuel and noise metrics between two runs.")
    parser.add_argument("run_a", help="First run reference")
    parser.add_argument("run_b", help="Second run reference")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: comparisons/<run_a>_vs_<run_b>/)",
    )
    args = parser.parse_args()
    bs.init()
    rp_a = resolve_run(args.run_a)
    rp_b = resolve_run(args.run_b)

    out_dir = (
        Path(args.out)
        if args.out
        else Path("comparisons") / f"{rp_a.run_name}_vs_{rp_b.run_name}"
    )

    calculate_metrics = build_metric_fn()
    compare_runs(rp_a, rp_b, out_dir, calculate_metrics)
