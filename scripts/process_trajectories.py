import argparse
from pathlib import Path

import pandas as pd
from bluesky_gym.metrics.evaluation_metrics import build_metric_fn
from scripts.common.run_paths import resolve_run, RunPaths

def plot_noise_bar_chart(df: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt
    noise_by_angle = df.groupby("start_angle")["calculated_noise"].sum()
    noise_by_angle.plot(kind="bar")
    plt.title("Total Noise by Start Angle")
    plt.xlabel("Start Angle (degrees)")
    plt.ylabel("Total Noise")
    plt.xticks(rotation=45)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fuel_bar_chart(df: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt
    fuel_by_angle = df.groupby("start_angle")["calculated_fuel"].sum()
    fuel_by_angle.plot(kind="bar")
    plt.title("Total Fuel Consumption by Start Angle")
    plt.xlabel("Start Angle (degrees)")
    plt.ylabel("Total Fuel Consumption (kg)")
    plt.xticks(rotation=45)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def process_trajectory_csv(csv_path: Path, traj_dir: Path) -> None:
    fuel_path = traj_dir / "fuel_bar_chart.png"
    noise_path = traj_dir / "noise_bar_chart.png"
    if fuel_path.exists() and noise_path.exists():
        print(f"Plots already exist, skipping: {traj_dir}")
        return

    df = pd.read_csv(csv_path)
    df["start_angle"] = ((df["start_angle"] / 10).round() * 10).astype(int)
    df = calculate_metrics(df)

    for start_angle, group in df.groupby("start_angle"):
        print(f"Start Angle: {start_angle} {group['calculated_fuel'].sum()} kg, {group['sim_dt'].sum()} seconds")
        print(f"Start Angle: {start_angle} {group['calculated_noise'].sum()} noise, {group['sim_dt'].sum()} seconds")

    plot_fuel_bar_chart(df, fuel_path)
    plot_noise_bar_chart(df, noise_path)


def process_for_run(run_paths: RunPaths) -> None:
    if not run_paths.trajectories_dir.exists():
        print(f"No trajectories found for {run_paths.run_id}")
        return
    for csv_path in sorted(run_paths.trajectories_dir.rglob("trajectories.csv")):
        traj_dir = csv_path.parent
        print(f"\n--- {run_paths.run_id} / {traj_dir.name} ---")
        process_trajectory_csv(csv_path, traj_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process trajectories and compute metrics.")
    parser.add_argument("run_refs", nargs="+", help="Run reference(s) or path to a trajectories.csv")
    args = parser.parse_args()

    file_path= Path("./scripts/population_maps/europe_3035.tif")

    calculate_metrics = build_metric_fn(file_path)

    # Legacy: if a single arg is a CSV file, process that directly
    if len(args.run_refs) == 1 and args.run_refs[0].endswith(".csv"):
        csv_path = Path(args.run_refs[0])
        process_trajectory_csv(csv_path, csv_path.parent)
    else:
        for run_paths in [resolve_run(r) for r in args.run_refs]:
            process_for_run(run_paths)
