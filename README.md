# Balancing Noise and Fuel: Spatially Aware Reinforcement Learning for Air Traffic Control
This repository is a fork of [BlueSky-Gym](#bluesky-gym)

## Getting started:
set up a virtual environment using `uv`
```shell
uv sync
```
Download a population density dataset 
This paper uses:
GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0_R{3,4,5}_C{18,19,20,21}.tif  
from ([dataset](https://human-settlement.emergency.copernicus.eu/download.php?ds=pop))

```shell
uv run scripts/merge_population_maps.py /path/to/downloads/GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0_R{3,4,5}_C{18,19,20,21}.tif \
    --output scripts/population_maps/europe_3035_1km.tif
```


## Model structure
The model consists of a navigation module in `bluesky-gym/envs/base_navigation_env.py` and a wrapper `bluesky-gym/wrappers/population.py` that add the population density maps and noise reward

## Runs
Extract the relevant zip-files in the `run` directory
Each run has the following structure:
```
runs/
└── {env_name}/
    └── {run_name}/
        ├── config.yaml           # Experiment configation as defined by config.py
        ├── best_model.zip        # Best trained model
        ├── metadata.json         # Run metadata (slurm_job_id, status, timestamps)
        ├── tensorboard/          # TensorBoard event files (only for the runs that use it to plot training rewards)
        └── trajectories/         # Trajectory data using the trained policy (one directory per runway) 
```

### Render environment
`uv run scripts/show_experiment runs/{env_name}/{run_name} --runway EHAM/RW27`

Any runway in the BlueSky database will work.
space pauses the simulation, r displays a radius of 250 km center at the airport, b displays the 10% borders

### Run experiment (Train a policy)
To run the training of experiment defined by `scripts/config.py` with seed `0`
```shell
uv run scripts/run_experiment.py config.yaml --seed 0
```
The scripts in `HPC` are used to train in parallel on a HPC.

### Generate trajectories
```shell
scripts/generate_all_trajectories.sh`
```

### Transform Parameters
```shell
uv run scripts/get_transform_parameters.py
```
### Generate density scaling sweep
```shell
scripts/generate_density_scaling.sh runs/scaling`
```
Performs the density scaling sweep for all the runs in the `runs/scaling` directory

Instructions on how to generate the tables and figures in the paper can be found in [Figures readme](plots/README.md) and [Tables readme](tables/README.md)

### Sweeps → paper sections

| Paper section        | zip-file                         | repository name              | notes                               |
| -------------------- |----------------------------------|------------------------------|-------------------------------------|
| Observation geometry | `observation-geometry`           | `runs/resolution_sweep_2/`   |                                     |
| Multi-scale          | `multi-scale`                    | `runs/multi-scale-sweep/`    |                                     |
| Domain randomization | `domain-randomization`           | `runs/transforms/`           |                                     |
| Baseline (no map)    | `baseline-no-map`                | `runs/BaseNavigationEnv-v0/` |                                     |
| Generalization       | `generalization-EHAM`            | `runs/generalization/`       | contains duplicates                 |
| Density frontier     | `generalization-density-scaling` | `runs/scaling/`              | contains duplicates                 |
| Appendix             | `appendix`                       | `runs/appendix/`             | contains tensorboard training logs  |
| Alignment            | `appendix-map-alignment`         | `runs/convergence/`          | contains duplicates                 |
| Groot et al. model | `groot-reference-model` | `runs/groot_legacy_model` | Used as a comparison at EHAM RW27   |

The Groot et al. model is represented by `groot-reference-model` (`runs/groot_legacy_model`) and is  used as a comparison at EHAM RW27

# BlueSky-Gym
A gymnasium style library for standardized Reinforcement Learning research in Air Traffic Management developed in Python.
Built on [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) and The Farama Foundation's [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

<p align="center">
    <img src="https://github.com/user-attachments/assets/6ae83579-78af-4cb7-8096-3a10af54a5c5" width=50% height=50%><br/>
    <em>An example trained agent attempting the merge environment available in BlueSky-Gym.</em>
</p>

For a complete list of the currently available environments click [here](bluesky_gym/envs/README.md)

## Installation

`pip install bluesky-gym`

Note that the pip package is `bluesky-gym`, for usage however, import as `bluesky_gym`.

## Usage
Using the environments follows the standard API from Gymnasium, an example of which is given below:

```python
import gymnasium as gym
import bluesky_gym
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = ... # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
```

Additionally you can directly use algorithms from standardized libraries such as [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) or [RLlib](https://docs.ray.io/en/latest/rllib/index.html) to train a model:

```python
import gymnasium as gym
import bluesky_gym
from stable_baselines3 import DDPG
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode=None)
model = DDPG("MultiInputPolicy",env)
model.learn(total_timesteps=2e6)
model.save()
```

For more info, please refer to the [workshop slides](https://docs.google.com/presentation/d/1Jpwdrx__OMdgHWtQ1yCVQyxsdDFk2ieX/edit?usp=drive_link&ouid=109800667545002770848&rtpof=true&sd=true) that provide additional information on BlueSky-Gym and how to use it for your own needs.

## Contributing and Assistance
If you would like to contribute to BlueSky-Gym or need assistance in setting up or creating your own environments, do not hesitate to open an issue or reach out to one of us via the BlueSky-Gym [Discord](https://discord.gg/s7CdxcSX).
Additionally you can have a look at the [roadmap](https://github.com/TUDelft-CNS-ATM/bluesky-gym/issues/24) for inspiration on where you can contribute and to get an idea of the direction BlueSky-Gym is going.


## Citing

If you use BlueSky-Gym in your work, please cite it using:
```bibtex
@misc{bluesky-gym,
  author = {Groot, DJ and Leto, G and Vlaskin, A and Moec, A and Ellerbroek, J},
  title = {BlueSky-Gym: Reinforcement Learning Environments for Air Traffic Applications},
  year = {2024},
  journal = {SESAR Innovation Days 2024},
}
```

List of publications & preprints using `BlueSky-Gym` (please open a pull request to add missing entries):
*   _missing entry_
