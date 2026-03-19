import csv
import os
from dataclasses import fields

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from bluesky_gym.envs.base_navigation_env import TerminationReason


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
    def __init__(self, experiment_config=None, verbose=0, validation_env=None, save_frequency=50000, save_dir=None):
        super().__init__(verbose)
        self.experiment_config = experiment_config
        self.validation_env = validation_env
        self.save_frequency = save_frequency
        self.save_dir = save_dir
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

        # Periodically save the model checkpoint
        if self.save_dir and self.save_frequency and self.num_timesteps - self.last_save_step >= self.save_frequency:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}_steps")
            self.model.save(checkpoint_path)
            if self.verbose:
                print(f"Model checkpoint saved to {checkpoint_path}")
            self.last_save_step = self.num_timesteps

        for info in self.locals.get('infos', []):
            # Only log your custom termination statistics
            if 'termination_stats' in info:
                for stat_name, count in info['termination_stats'].items():
                    self.logger.record(f"termination/{stat_name}", count)
        return True