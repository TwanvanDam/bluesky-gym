### Map dataset coverage
```shell
uv run scripts/inspect_population_maps.py --exclusion 52.308 4.764 250
```

### Observation modes figure
```shell
uv run scripts/visualize_observation_modes.py 
```

## Plots
### plot sweep overviews
```shell
uv run scripts/plot_resolution_sweep.py runs/resolution_sweep_2 --scenario EDDF_RW25R --cache --baseline runs/BaseNavigationEnv-v0/sweep_2_no_map_seed0*
uv run scripts/plot_multi_scale_sweep.py runs/multi-scale-sweep --scenario EDDF_RW25R --cache --baseline runs/resolution_sweep_2/sweep_2_centered_4_seed0*
uv run scripts/plot_weird_comparison.py --scenario EDDF_RW25R
uv run scripts/plot_transform_sweep.py runs/transforms --scenario EDDF_RW25R --cache --baseline runs/resolution_sweep_2/sweep_2_centered_4_seed0*  
uv run scripts/plot_generalization_sweep.py runs/generalization --scenario EHAM_RW27 --cache
uv run scripts/plot_density_scaling_sweep.py runs/scaling --runway EDDF_RW25R --use-cache
uv run scripts/plot_resolution_sweep.py runs/appendix/resolution_sweep_1_backfill --scenario EDDF_RW25R --plots breakdown --baseline runs/BaseNavigationEnv-v0/no_map_seed0*
uv run scripts/plot_alignment_comparison.py --cache --runway EDDF_RW25R
```

### Plot trajectories
```shell
uv run scripts/plot_trajectory_figure.py runs/resolution_sweep_2/observation_geometry.txt --legend
uv run scripts/plot_trajectory_figure.py runs/generalization/generalization_trajectories.txt --legend
uv run scripts/plot_trajectory_figure.py runs/generalization/generalization_scaling.txt --legend
uv run scripts/plot_trajectory_figure.py runs/generalization/generalization_failures.txt --legend
uv run scripts/plot_trajectory_figure.py runs/appendix/initial_exploration.txt --width 0.75 --legend
uv run scripts/plot_trajectory_figure.py runs/appendix/appendix_trajectories.txt --width 0.75 --legend
```

## Training reward curves
```shell
uv run scripts/plot_run_rewards.py runs/appendix/centered_16
uv run scripts/plot_run_rewards.py runs/appendix/centered_16_learning_lower_correct
uv run scripts/plot_run_rewards.py runs/appendix/centered_16_ablation
```

## Figures without a dedicated script
The simulation boundary figure is a screenshot, made using `scripts/show_experiment.py`