import bluesky
import numpy as np
from bluesky.tools.position import Position

from bluesky_gym.envs.common.environment_factory import load_env_and_model, normalize_run_name
from bluesky_gym.maps.map_datasets import MapSourceConfigType, TiffMapSourceConfig, RandomMapSourceConfig


def render_experiment(run_name: str, map_config: MapSourceConfigType | None = None, runway: str = "18R"):
    bluesky.init()
    destination = Position(name=runway, reflat=0, reflon=0)
    options = {
        "destination_lat": destination.lat,
        "destination_lon": destination.lon,
        "destination_hdg": destination.refhdg
    }

    env, model = load_env_and_model(run_name, render_mode="human", map_config=map_config)
    while True:
        obs, info = env.reset(options=options)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"Fuel: {info['total_episode_fuel_used']:.2f} kg, Reward:{info['total_episode_fuel_reward']:.2f}")
        print(f"Noise: {info['total_episode_noise']:.2f}, Reward:{info['total_episode_noise_reward']:.2f}")
        print(f"Episode Length: {info['episode_length_seconds']/60:.2f} minutes")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Render a trained experiment by run name, with optional map override.")
    parser.add_argument("run_name", type=str, help="Name of the run to render (e.g. 'PopulationWrapper-v0/2026-03-07_10_55_19.yaml'). Must match the config in results_backup.")
    parser.add_argument("--runway", type=str, default="EHAM/RW18R", help="Runway to set as destination for rendering (default: EHAM/RW18R)")
    parser.add_argument("--use_real_map", action="store_true",default=False, help="Whether to use the real population map for this example (overrides any map in the original config)")
    args = parser.parse_args()

    run_name = normalize_run_name(args.run_name)
    use_zero_map = False
    print(f"Rendering run: {run_name} with runway: {args.runway} and use_real_map: {args.use_real_map}")
    if args.use_real_map:
        validation_map = TiffMapSourceConfig(file_path="scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff", source_unit="people_per_pixel")
    elif use_zero_map:
        validation_map = RandomMapSourceConfig(type="zero", resolution_m=1000, source_unit="people_per_pixel")
    else:
        validation_map = None

    render_experiment(run_name, map_config=validation_map, runway=args.runway)
