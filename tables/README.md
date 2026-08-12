## Means tables
```shell
uv run scripts/create_means_table.py runs/resolution_sweep_2/cached_metrics_EDDF_RW25R.csv --baseline runs/resolution_sweep_2/cached_baseline_metrics_EDDF_RW25R.csv -o tables/observation_geometry_means.tex
uv run scripts/create_means_table.py runs/multi-scale-sweep/cached_metrics_EDDF_RW25R.csv --baseline runs/multi-scale-sweep/cached_baseline_metrics_EDDF_RW25R.csv -o tables/multi_scale_means.tex
uv run scripts/create_means_table.py runs/transforms/cached_metrics_EDDF_RW25R.csv --baseline runs/transforms/cached_baseline_metrics_EDDF_RW25R.csv -o tables/transform_means.tex
uv run scripts/create_means_table.py runs/generalization/cached_metrics_EHAM_RW27.csv --group-by config -o tables/generalization_means.tex
```
`uv run scripts/plot_density_scaling_sweep.py runs/scaling --runway EDDF_RW25R --use-cache` Produces a table with failure rates, next to the frontier plot.
### Row labels used in the paper's tables
| Paper label                     | Runs                                              |
|---------------------------------| ------------------------------------------------- |
| `No-map` (observation geometry) | `runs/BaseNavigationEnv-v0/sweep_2_no_map_seed0*` |
| `No-map` (appendix)             | `runs/BaseNavigationEnv-v0/no_map_seed0*` |
| `C4` / `C4-old`                 | `runs/resolution_sweep_2/sweep_2_centered_4_seed0*` |

`C4` is the reference row of the multi-scale and transform tables and is supplied via
`--baseline`. Note that `create_means_table.py` labels any `--baseline` row `No-map`,
so the first row of the transform table has to be relabelled `C4` by hand.
