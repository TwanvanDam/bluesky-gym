# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BlueSky-Gym is a Gymnasium-style Reinforcement Learning environment library for Air Traffic Management (ATM) research. It combines the BlueSky air traffic simulator with Stable-Baselines3 and supports configurable training via YAML experiments.

The repo mixes two tracks:
- **Legacy benchmark envs**: registered via `bluesky_gym.register_envs()` (7 envs like `MergeEnv-v0`)
- **Current configurable stack**: `BaseNavigationEnv` + wrappers, driven by YAML configs and trained via `scripts/run_experiment.py`

Use the legacy env files only when editing published benchmark environments. For everything else, use the configurable stack.

## Commands

**Install (Python 3.12.* required):**
```bash
uv sync   # preferred (uses uv.lock for reproducibility)
# or
pip install -e .
```

**Single experiment training:**
```bash
python -m scripts.run_experiment HPC/experiments/Test.yaml
```

**Replay a trained run:**
```bash
python -m scripts.show_experiment "PopulationWrapper-v0/RealMap_base_2026-03-27_14_41_59_984163"
```

**Validate config schema:**
```bash
python scripts/config.py <config.yaml>
```

**Generate trajectories (single or batch):**
```bash
python -m scripts.generate_trajectories "PopulationWrapper-v0/RealMap_base_2026-..."
python -m scripts.generate_trajectories --env PopulationWrapper-v0
python -m scripts.generate_trajectories --pattern "RealMap_small_*"
```

**Plot trajectories (single or batch):**
```bash
python -m scripts.present_trajectories "PopulationWrapper-v0/RealMap_base_2026-..."
python -m scripts.present_trajectories --env PopulationWrapper-v0
```

**HPC (SLURM + Apptainer):**
```bash
apptainer build HPC/rl_env.sif HPC/container.def
sbatch HPC/run_training.sbatch HPC/experiments/Test.yaml
```

**Migrate old results to unified layout:**
```bash
bash scripts/migrate_to_runs.sh              # dry-run
bash scripts/migrate_to_runs.sh --execute    # copy files
```

**Move SLURM logs into a run directory:**
```bash
bash scripts/move_slurm_logs.sh runs/PopulationWrapper-v0/RealMap_base_2026-... 12345678
```

There is no automated test suite. Validate changes with short smoke runs via the training and replay scripts above.

## Architecture

### Big-picture (read in this order)

1. `scripts/common/run_paths.py` — Central path resolution (`RunPaths`, `resolve_run`, `iter_runs`). All scripts import from here; no script constructs result paths on its own.
2. `scripts/config.py` — Pydantic config contract (`ExperimentConfig`, `NavigationConfig`, etc.). Uses `extra='forbid'`; unknown YAML keys fail fast.
3. `bluesky_gym/envs/base_navigation_env.py` — canonical navigation env: BlueSky sim lifecycle, observation/action/reward components, rendering.
4. `bluesky_gym/wrappers/population.py` — adds 2D population density map observation + noise reward; owns rendering when wrapped.
5. `bluesky_gym/maps/map_datasets.py`, `bluesky_gym/maps/random_map_generators.py` — map source boundary (real GeoTIFF vs. Gaussian random field generated maps).
6. `scripts/run_experiment.py` — env assembly and SAC training entrypoint. Creates unified run directory.
7. `scripts/show_experiment.py` — model replay and trajectory visualization.

### Runtime data flow

```
HPC/experiments/*.yaml
  → ExperimentConfig.load(...)
  → load_env_from_config(...)  [bluesky_gym/envs/common/environment_factory.py]
  → BaseNavigationEnv
  → SinCosNormalization (optional)
  → DistanceNormalization (optional)
  → RescaleAction(-1, 1)
  → Population wrapper (optional, map-based obs)
  → MapObservationNormalizer (optional)
  → SAC.learn()
```

- Reward composition is additive via `BaseNavigationEnv.add_reward_component(...)`. `Population` injects `_get_noise_reward` into the base env.
- `Population` sets `base_env._render_owned_by_wrapper = True`; always call `env.render()` on the outermost wrapper.

### Results directory structure

All artifacts for a single run live together in one directory under `runs/`:

```
runs/
└── {env_name}/
    └── {run_name}/
        ├── config.yaml           # Experiment config snapshot
        ├── model.zip             # Final trained model
        ├── metadata.json         # Run metadata (slurm_job_id, status, timestamps)
        ├── tensorboard/          # TensorBoard event files
        ├── checkpoints/          # Periodic model checkpoints
        ├── trajectories/         # Evaluation trajectory data (one subdir per config)
        └── slurm/                # Copied SLURM stdout/stderr logs
```

Path resolution is centralized in `scripts/common/run_paths.py`. All scripts accept flexible run references: full path, `env_name/run_name`, or bare `run_name`. Downstream scripts (`generate_trajectories`, `present_trajectories`, `process_trajectories`) support batch mode via `--env` and `--pattern` flags.

## Conventions

- Observations are `gymnasium.spaces.Dict` with NumPy arrays (mostly `np.float64`); keep shapes/dtypes stable for SB3 `MultiInputPolicy`.
- BlueSky sim is initialized once (`bs.init(mode='sim', detached=True)`), then each episode resets via `bs.traf.reset()`.
- Action semantics: base env outputs heading delta in degrees; `RescaleAction` maps policy output from `[-1, 1]`.
- Coordinate handling is CRS-aware (WGS84 → `pygame_crs` via `pyproj`); map reprojection uses `rasterio.warp.reproject`.
- `base_navigation_env.py` imports `NavigationConfig` from `scripts.config`; keep `scripts` on `PYTHONPATH` (set in `HPC/container.def`).

## Research Plan

The active experiment roadmap (step-by-step plan, decision gates, writing guide) is at:

```
HPC/experiments/experiment_roadmap.md
```

Current status (2026-05-14): validating stable training config (`centered_16_all` × 3 seeds). Next step is Block A resolution sweep rerun once that gate is passed.

## Safe Edit Checklist

- **Config schema changes**: edit `scripts/config.py` and update matching `HPC/experiments/*.yaml` examples together.
- **Observation key changes**: update all wrappers/extractors that reference those keys (`scripts/feature_extractors.py`, `MapObservationNormalizer`).
- **Rendering changes**: verify both unwrapped base env and wrapped population env behavior.
- **Feature extractor wiring**: `run_experiment.py` checks `feature_extractor_config` but config field is named `feature_extractor` — update both sides consistently.
- **Wrapper order changes**: `show_experiment.py` accesses wrapper internals (`env.env.background_map`, `env.unwrapped`); reordering can break replay utilities.
