from typing import Literal

import gymnasium as gym
from gymnasium.wrappers import ClipReward
from stable_baselines3 import SAC

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from bluesky_gym.maps.map_sources import MapSourceConfigType
from bluesky_gym.wrappers.action_rescaler import ScaledHeadingAction
from bluesky_gym.wrappers.distance_normalizer import DistanceNormalization
from bluesky_gym.wrappers.map_observation_normalizer import MapObservationNormalizer
from bluesky_gym.wrappers.population import Population
from bluesky_gym.wrappers.sin_cos_normalizer import SinCosNormalization
from scripts.common.run_paths import RunPaths, resolve_run
from scripts.config import ExperimentConfig
from scripts.feature_extractors import CombinedExtractor


def _apply_wrappers(env: gym.Env, config: ExperimentConfig) -> tuple[gym.Env, str]:
    """Apply the full wrapper stack in order. Returns (wrapped_env, env_name)."""
    name = "BaseNavigationEnv-v0"
    if config.navigation_config.use_sin_cos_obs:
        env = SinCosNormalization(env)
    if config.navigation_config.normalize_distance_obs:
        env = DistanceNormalization(env, normalization_factor=config.navigation_config.normalize_distance_obs)
    env = ScaledHeadingAction(env=env, max_deg=config.navigation_config.max_action_degrees)
    if config.population_config:
        env = Population(env, config=config.population_config)
        name = "PopulationWrapper-v0"
        if config.population_config.observation_normalization:
            env = MapObservationNormalizer(
                env,
                mode=config.population_config.observation_normalization,
                clip=config.population_config.clip_noise_reward,
            )
    if config.navigation_config.clip_reward_min is not None or config.navigation_config.clip_reward_max is not None:
        env = ClipReward(env=env, min_reward=config.navigation_config.clip_reward_min, max_reward=config.navigation_config.clip_reward_max)
    return env, name


def build_env(config: ExperimentConfig, render_mode: str | None = None) -> tuple[gym.Env, str]:
    """Build and wrap the environment from an ExperimentConfig."""
    env = BaseNavigationEnv(config=config.navigation_config, render_mode=render_mode)
    return _apply_wrappers(env, config)


def _load_model(run_paths: RunPaths, env: gym.Env, config: ExperimentConfig, model_type: Literal["final", "best"]) -> SAC:
    """Load a SAC model from a run directory, falling back to the latest checkpoint."""
    if config.agent_config.algorithm != "SAC":
        raise NotImplementedError(f"Algorithm {config.agent_config.algorithm!r} is not supported.")
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": {"config": config.agent_config.feature_extractor},
        "net_arch": config.agent_config.network_arch,
    }
    custom_objects = {"policy_kwargs": policy_kwargs}
    candidates = {"best":  run_paths.best_model,
                  "final": run_paths.final_model,
                  "latest_checkpoint": run_paths.latest_checkpoint}

    if model_type in candidates:
        path = candidates[model_type]
    else:
        print(f"{model_type}_model.zip not found. Loading latest checkpoint: {candidates['latest_checkpoint']}")
        path = candidates['latest_checkpoint']

    if path and path.exists():
        return SAC.load(path, env=env, device='auto', custom_objects=custom_objects)
    else:
        raise FileNotFoundError(f"No model or checkpoints found for run {run_paths.run_id}")


def load_env_and_model(
    run: str | RunPaths,
    render_mode: str | None = "human",
    map_config: MapSourceConfigType | None = None,
    model_type: Literal["best", "final"] = "best",
) -> tuple[gym.Env, SAC]:
    """Load a trained model and its environment from a run directory.

    Args:
        run: Run path, run_id string, or RunPaths object.
        render_mode: Passed to the base environment.
        map_config: Optional map source override; replaces the config's map_source_config
                    without mutating the on-disk config.
    """
    run_paths = resolve_run(run) if isinstance(run, str) else run
    config = ExperimentConfig.load(run_paths.config)
    if map_config and config.population_config:
        pop_config = config.population_config.model_copy(update={"map_source_config": map_config})
        config = config.model_copy(update={"population_config": pop_config})
    env, _ = build_env(config, render_mode)
    model = _load_model(run_paths, env, config, model_type=model_type)
    return env, model
