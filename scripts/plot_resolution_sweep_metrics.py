import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator

import matplotlib.pyplot as plt
import pandas as pd
import bluesky as bs
from bluesky.tools.position import Position
from tqdm import tqdm

from bluesky_gym.maps.map_sources import TiffMapSourceConfig
from bluesky_gym.maps.raster_sampler import RasterSampler, MapObservationConfig
from bluesky_gym.metrics.fuel_model import FuelModel
from bluesky_gym.metrics.noise_model import NoiseConfig

RUN_PATTERN = re.compile(r"(forward|centered)_(\d+)_seed(\d+)")
SUCCESS_REASON = "success"

FORWARD_COLOR = "#2196F3"
CENTERED_COLOR = "#FF6B35"
BASELINE_COLOR = "#555555"
BOX_OFFSET = 0.2
BOX_WIDTH = 0.35


@dataclass
class Record:
    mode: str
    fuel: float
    noise: float
    success: bool
    resolution: float | None
    seed: int

def build_metric_fn(map_path: Path) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Initialise models once and return a calculate_metrics(df) function."""
    raster_sampler = RasterSampler(
        map_source=TiffMapSourceConfig(file_path=map_path).build(),
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

def find_run_dirs(run_pattern: None | list[str], runs_root: Path) -> Generator:
    for run_dir in sorted(runs_root.iterdir()):
        if not run_pattern:
            pass
        elif not any(pattern in run_dir.name for pattern in run_pattern):
            continue
        yield run_dir

def find_csv(run_dir: Path, runway: str) -> None | Path:
    csvs = list(run_dir.glob(f"trajectories/*{runway}_map*/trajectories.csv"))
    if not csvs:
        return None
    if len(csvs) == 1:
        return csvs[0]
    else:
        print("More than one csv found for " + runway)
        return csvs[0]

def collect_metrics(runs_root: Path, runway: str, run_pattern: None| list[str], calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    run_dirs = find_run_dirs(run_pattern, runs_root)
    records = []
    for run_dir in tqdm(list(run_dirs)):
        m = RUN_PATTERN.search(run_dir.name)
        if not m:
            print(f"  Skipping {run_dir.name}: cannot parse mode/resolution/seed")
            continue
        mode, resolution, seed = m.group(1), int(m.group(2)), int(m.group(3))

        csv = find_csv(run_dir, runway)
        print(csv)
        if not csv:
            continue
        df = pd.read_csv(csv)
        df = calculate_metrics(df)
        df_grouped = df.groupby("start_angle")
        fuel_summed = df_grouped["calculated_fuel"].sum()
        noise_summed = df_grouped["calculated_noise"].sum()
        success_per_episode = df_grouped["termination_reason"].last() == SUCCESS_REASON

        for start_angle in fuel_summed.index:
            records.append(
                Record(
                    mode=mode,
                    resolution=resolution,
                    fuel=fuel_summed[start_angle],
                    noise=noise_summed[start_angle],
                    success=success_per_episode[start_angle],
                    seed=seed
                )
            )
    return pd.DataFrame(records)

def collect_baseline_metrics(baseline_run: Path, runway: str, calculate_metrics: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    seed_match = re.search(r"seed(\d+)", baseline_run.name)
    seed = int(seed_match.group(1)) if seed_match else 0

    csv = find_csv(baseline_run, runway)
    if not csv:
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = calculate_metrics(df)
    if "termination_reason" not in df.columns:
        df["termination_reason"] = "success"
    df_grouped = df.groupby("start_angle")
    fuel_summed = df_grouped["calculated_fuel"].sum()
    noise_summed = df_grouped["calculated_noise"].sum()
    success_per_episode = df_grouped["termination_reason"].last() == SUCCESS_REASON
    records = []
    for start_angle in fuel_summed.index:
        records.append(
            Record(
                mode="no_map",
                resolution=None,
                fuel=fuel_summed[start_angle],
                noise=noise_summed[start_angle],
                success=success_per_episode[start_angle],
                seed=seed,
            )
        )
    return pd.DataFrame(records)

def _draw_boxplot(ax, data, position, color):
    ax.boxplot(
        data,
        positions=[position],
        widths=BOX_WIDTH,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor=color, alpha=0.6),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        flierprops=dict(marker="o", color=color, alpha=0.4, markersize=3),
    )


def plot_metric_boxplot(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    metric: str,
    ylabel: str,
    runway: str,
) -> None:
    resolutions = sorted(df["resolution"].dropna().unique())
    # baseline at 0, resolutions start at 1
    x_positions = {res: i + 1 for i, res in enumerate(resolutions)}

    fig, ax = plt.subplots(figsize=(8, 5))

    legend_handles = []

    if baseline_df is not None and not baseline_df.empty:
        _draw_boxplot(ax, baseline_df[metric].values, position=0, color=BASELINE_COLOR)
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR, alpha=0.6, label="No-map baseline"))
        q1, median, q3 = baseline_df[metric].quantile([0.25, 0.5, 0.75])
        for val, ls in [(median, "--"), (q1, ":"), (q3, ":")]:
            ax.axhline(val, color=BASELINE_COLOR, linestyle=ls, linewidth=0.8, alpha=0.6)

    mode_config = [
        ("centered", CENTERED_COLOR, -BOX_OFFSET),
        ("forward",  FORWARD_COLOR,  +BOX_OFFSET),
    ]

    for mode, color, offset in mode_config:
        mode_df = df[df["mode"] == mode]
        # mode_df = mode_df[mode_df["success"] == True]
        for res in resolutions:
            data = mode_df[mode_df["resolution"] == res][metric].values
            if len(data) == 0:
                continue
            _draw_boxplot(ax, data, position=x_positions[res] + offset, color=color)
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6, label=mode.capitalize()))

    ax.set_xticks([0] + list(range(1, len(resolutions) + 1)))
    ax.set_xticklabels(["No map"] + [f"{r} km/px" for r in resolutions])
    ax.set_xlabel("Observation resolution")
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend_handles, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()
    out_path = Path(f"./plots/{metric}_{runway}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description="Plot resolution sweep metrics")
    arg_parser.add_argument("runs_root", type=str, help="path to runs root for comparison")
    arg_parser.add_argument("--baseline_run", nargs="+", type=str, help="path to baseline runs root")
    arg_parser.add_argument("--runway", type=str, default="EHAM_RW27", help="runway to use for comparison")
    arg_parser.add_argument("--map-path", type=str, default="./scripts/population_maps/europe_3035_1km.tif", help="path to map source")
    arg_parser.add_argument("--cache", type=bool, default=False, help="whether to cache results")
    args = arg_parser.parse_args()

    bs.init()
    calculate_metrics = build_metric_fn(Path(args.map_path))

    cache_path = Path(args.runs_root) / f"cached_metrics_{args.runway}.csv"
    if cache_path.exists() and args.cache:
        print("Using the cached metrics...")
        run_metrics = pd.read_csv(Path(args.runs_root) / f"cached_metrics_{args.runway}.csv")
    else:
        run_metrics = collect_metrics(Path(args.runs_root), args.runway, ["forward", "centered"], calculate_metrics)
        if args.cache:
            print(f"Saving results to {cache_path} ...")
            run_metrics.to_csv(cache_path)

    if args.baseline_run:
        baseline_metrics = collect_baseline_metrics(Path(args.baseline_run[0]), args.runway,calculate_metrics)
    else:
        baseline_metrics = None

    plot_metric_boxplot(run_metrics, baseline_metrics, "fuel", "fuel [kg]", args.runway)
    plot_metric_boxplot(run_metrics, baseline_metrics, "noise", "noise", args.runway)
