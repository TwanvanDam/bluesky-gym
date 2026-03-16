import bluesky
from bluesky.tools.position import Position

from bluesky_gym.envs.common.environment_factory import load_env_and_model
from bluesky_gym.maps.map_datasets import MapSourceConfigType, TiffMapSourceConfig


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
    run_name = "PopulationWrapper-v0/2026-03-07_10_55_19"
    validation_map = TiffMapSourceConfig(file_path="scripts/population_maps/ESTAT_OBS-VALUE-T_2021_V2.tiff")

    render_experiment(run_name, map_config=validation_map, runway="EHRD/RW24")