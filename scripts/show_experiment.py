import bluesky
from bluesky.tools.position import Position

from bluesky_gym.envs.common.environment_factory import load_env_and_model
from bluesky_gym.maps.map_sources import MapSourceConfigType, TiffMapSourceConfig, RandomMapSourceConfig, \
    TransformedTiffMapSourceConfig
from bluesky_gym.maps.map_transforms import Clip


def render_experiment(run_name: str, map_config: MapSourceConfigType | None = None, runway: str | None = None):
    bluesky.init()

    options = {}
    if runway:
        destination = Position(name=runway, reflat=0, reflon=0)
        options.update({
            "destination_lat": destination.lat,
            "destination_lon": destination.lon,
            "destination_hdg": destination.refhdg
        })

    env, model = load_env_and_model(run_name, render_mode="human", map_config=map_config)

    while True:
        obs, info = env.reset(options=options)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(info["termination_reason"])
        print(f"Fuel: {info['total_episode_fuel_used']:.2f} kg, Reward:{info['total_episode_fuel_reward']:.2f}")
        if 'total_episode_noise' in info:
            print(f"Noise: {info['total_episode_noise']:.2f}, Reward:{info['total_episode_noise_reward']:.2f}")
        print(f"Episode Length: {info['episode_length_seconds']/60:.2f} minutes")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Render a trained experiment by run name, with optional map override.")
    parser.add_argument("run_name", type=str, help="Run reference (e.g. 'PopulationWrapper-v0/RealMap_base_2026-...')")
    parser.add_argument("--runway", type=str, help="Runway to set as destination for rendering (e.g.: EHAM/RW18R)")
    parser.add_argument("--map_type", type=str, default="real_clipped", help="Whether to use the real population map for this example (overrides any map in the original config)")
    parser.add_argument("--map_path", type=str, default="scripts/population_maps/europe_3035_1km.tif", help="map dataset location (default = scripts/population_maps/europe_3035_1km.tif)")
    args = parser.parse_args()

    run_ref = args.run_name
    map_type = args.map_type.lower()
    print(f"Rendering run: {run_ref} with runway: {args.runway} and use_real_map: {args.map_type}")
    match map_type:
        case "original":
            validation_map = None  # Use the map from the original config (if any)
        case "real":
            validation_map = TiffMapSourceConfig(file_path=args.map_path, source_unit="people_per_pixel")
        case "real_clipped":
            validation_map = TransformedTiffMapSourceConfig(file_path=args.map_path, source_unit="people_per_pixel", spatial_transforms=[], value_transforms=[Clip(percentile=99.9)], window_margin_m=0)
        case "zero":
            validation_map = RandomMapSourceConfig(type="zero", resolution_m=1000, source_unit="people_per_pixel")
        case "random":
            covariance_models = {"cov_1" : {"cov_model" : "Gaussian", 'var':0.625, 'len_scale':60.3},
                         "cov_2" : {"cov_model" : "Gaussian", 'var':0.815, 'len_scale':2.63e2},
                         "cov_3" : {"cov_model" : "Integral", 'var':1.83, 'len_scale':37.0, 'nu': 0.233}}

            validation_map = RandomMapSourceConfig(type="population_density", resolution_m=4000, source_unit="people_per_km2", kwargs=dict(covariance_models=covariance_models, target_mean=361.60))
        case _:
            raise ValueError(f"Unsupported map type: {map_type}. Choose from 'original', 'real', 'zero', or 'random'.")
    render_experiment(run_ref, map_config=validation_map, runway=args.runway)
