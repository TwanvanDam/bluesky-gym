import csv
import os
from dataclasses import fields

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam, TensorBoardOutputFormat

from bluesky_gym.envs.base_navigation_env import Airport, Position, TerminationReason
from bluesky_gym.envs.common import functions


class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_dir, file_name='training_log.csv', verbose=0):
        super(CSVLoggerCallback, self).__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, file_name)
        self.headers = ['timesteps', 'episodes']
        self.initialized = False
        self.episode_count = 0

    def _on_step(self) -> bool:
        if not self.initialized:
            # Initialize headers based on keys in the infos dictionary
            self.info_keys = self.locals['infos'][0].keys()
            self.headers.extend(self.info_keys)
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            self.initialized = True

        if self.locals['dones'][0]:
            self.episode_count += 1
            timesteps = self.num_timesteps
            info_dict = self.locals['infos'][0]
            info_values = [info_dict.get(key, None) for key in self.info_keys]
            row = [timesteps, self.episode_count] + list(info_values)
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        return True

def _flatten_config(obj, prefix="") -> dict:
    """Recursively flatten a dataclass/dict into a flat dict with string values suitable for TB hparams."""
    flat = {}

    # Check if it's a Pydantic model
    if hasattr(obj, "model_dump"):
        items = obj.model_dump()
    # Check if it's a dataclass
    elif hasattr(obj, "__dataclass_fields__"):
        items = {f.name: getattr(obj, f.name) for f in fields(obj)}
    elif isinstance(obj, dict):
        items = obj
    else:
        # For primitive types, return them directly
        if isinstance(obj, (int, float, bool, str)) or obj is None:
            return {prefix: obj} if prefix else {str(obj): obj}
        else:
            return {prefix: str(obj)} if prefix else {}

    for key, value in items.items():
        full_key = f"{prefix}/{key}" if prefix else key

        # Recursively flatten nested objects
        if hasattr(value, "model_dump") or (hasattr(value, "__dataclass_fields__")) or isinstance(value, dict):
            flat.update(_flatten_config(value, full_key))
        # Handle None values
        elif value is None:
            flat[full_key] = "None"
        # Handle primitive types
        elif isinstance(value, (int, float, bool, str)):
            flat[full_key] = value
        # Handle lists and tuples
        elif isinstance(value, (list, tuple)):
            flat[full_key] = str(value)
        # Convert everything else to string
        else:
            flat[full_key] = str(value)

    return flat


class TensorboardCallback(BaseCallback):
    def __init__(self, experiment_config=None, verbose=0, validation_env=None, plot_frequency=10000, save_frequency=50000, save_dir=None):
        super().__init__(verbose)
        self.experiment_config = experiment_config
        self.validation_env = validation_env
        self.plot_frequency = plot_frequency
        self.save_frequency = save_frequency
        self.save_dir = save_dir
        self.last_plot_step = 0
        self.last_save_step = 0

    def _on_training_start(self) -> None:
        if self.experiment_config is None:
            return

        # --- Log hparams to TensorBoard ---
        hparam_dict = _flatten_config(self.experiment_config)
        # TensorBoard hparams require at least one metric to correlate with
        metric_dict = {
            "rollout/ep_rew_mean": 0.0,
            "rollout/ep_len_mean": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def make_validation_plot(self):
        angles = np.arange(0, 360, 10)
        destination = Airport(Position(lat=52.31, lon=4.7), hdg=180)
        figure = plt.figure()
        for angle in list(angles):
            aircraft_lat, aircraft_lon = functions.get_point_at_distance(destination.position.lat, destination.position.lon,
                                       300, angle)
            done = False
            obs, info = self.validation_env.reset(options={
                "airport_lat": destination.position.lat,
                "airport_lon": destination.position.lon,
                "airport_hdg": destination.hdg,
                "aircraft_lat": aircraft_lat,
                "aircraft_lon": aircraft_lon,
            }, seed=42)
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.validation_env.step(action)
                done = terminated or truncated
            points = [(position.lon, position.lat) for position in
                      self.validation_env.unwrapped.aircraft_positions]
            xs, ys = zip(*points)
            plt.plot(xs, ys)
        plt.xlim(self.validation_env.unwrapped.lon_min, self.validation_env.unwrapped.lon_max)
        plt.ylim(self.validation_env.unwrapped.lat_min, self.validation_env.unwrapped.lat_max)
        plt.scatter(destination.position.lon, destination.position.lat, marker=".", linewidths=5)
        print("saving figure")
        figures_dir = "scripts/common/results/figures_backup"
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(f"{figures_dir}/{self.experiment_config.run_name}_{self.num_timesteps}.png")
        # Write directly to TensorBoard writer to avoid interfering with SB3's logger step tracking
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                fmt.writer.add_figure("validation/circle_trajectories", figure, global_step=self.num_timesteps)
                fmt.writer.flush()
                break
        plt.close(figure)

    def _on_training_end(self) -> None:
        if self.validation_env is not None:
            self.make_validation_plot()

    def _on_step(self) -> bool:
        scalar_types = (int, float, np.integer, np.floating)
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # Record episode-level metrics exactly when an episode terminates.
        for done, info in zip(dones, infos):
            if not done:
                continue

            for key, value in info.items():
                if key == "termination_reason":
                    for reason in TerminationReason:
                        if reason.value == value:
                            self.logger.record_mean(f"episode/termination_reason/{reason.name}", 1.0)
                        else:
                            self.logger.record_mean(f"episode/termination_reason/{reason.name}", 0.0)
                if isinstance(value, scalar_types):
                    self.logger.record_mean(f"episode/{key}", float(value))

        # Check if it's time to make a validation plot
        if self.validation_env and self.num_timesteps - self.last_plot_step >= self.plot_frequency:
            self.make_validation_plot()
            self.last_plot_step = self.num_timesteps

        # Periodically save the model checkpoint
        if self.save_dir and self.save_frequency and self.num_timesteps - self.last_save_step >= self.save_frequency:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}_steps")
            self.model.save(checkpoint_path)
            if self.verbose:
                print(f"Model checkpoint saved to {checkpoint_path}")
            self.last_save_step = self.num_timesteps

        return True