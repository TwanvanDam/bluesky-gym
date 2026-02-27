import csv
import os
from dataclasses import fields

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam, Figure

from bluesky_gym.envs.base_navigation_env import Airport, Position
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
    if hasattr(obj, "__dataclass_fields__"):
        items = {f.name: getattr(obj, f.name) for f in fields(obj)}
    elif isinstance(obj, dict):
        items = obj
    else:
        return {prefix: obj}

    for key, value in items.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if hasattr(value, "__dataclass_fields__") or isinstance(value, dict):
            flat.update(_flatten_config(value, full_key))
        elif value is None:
            flat[full_key] = "None"
        elif isinstance(value, (int, float, bool, str)):
            flat[full_key] = value
        else:
            flat[full_key] = str(value)

    return flat


class TensorboardCallback(BaseCallback):
    def __init__(self, experiment_config=None, verbose=0, validation_env = None):
        super().__init__(verbose)
        self.experiment_config = experiment_config
        self.validation_env = validation_env

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

    def _on_training_end(self) -> None:
        angles = np.arange(0, 360, 180)
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
            })
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.validation_env.step(action)
                done = terminated or truncated
            points = [(position.lon, position.lat) for position in
                      self.validation_env.aircraft_positions]
            xs, ys = zip(*points)
            plt.plot(xs, ys)
        print("saving figure")
        plt.savefig("figure.png")
        self.logger.record("validation/circle_trajectories", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        self.logger.dump()
        plt.close(figure)

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            # Only log your custom termination statistics
            if 'termination_stats' in info:
                for stat_name, count in info['termination_stats'].items():
                    self.logger.record(f"termination/{stat_name}", count)

        return True