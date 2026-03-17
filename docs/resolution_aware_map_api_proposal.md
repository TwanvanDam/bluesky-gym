# Resolution-Aware Map API Proposal

This document proposes API shapes for making map units consistent across GeoTIFF and synthetic map sources.

Status: **proposal only** (no implementation yet).

## Problem

`TiffMapSource` maps have an implicit physical resolution from GeoTIFF metadata, while random generators currently only return arrays.
That means map values can represent different units (or no explicit unit), which breaks comparability of reward terms and observations.

## Design goals

- Make map value semantics explicit.
- Make spatial resolution explicit for random maps.
- Normalize to a canonical internal unit before reward usage.
- Keep wrappers/reward code source-agnostic.
- Support faster map regeneration during training.

## Canonical internal units

Use **`people_per_m2`** internally.

Input semantics can be one of:

- `people_per_m2`
- `people_per_km2`
- `people_per_cell` (count in a raster cell)
- `relative_index` (dimensionless synthetic index)

All map sources should expose a conversion path to canonical density.

## Proposed API (example)

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Callable
import numpy as np
from affine import Affine

ValueSemantics = Literal[
    "people_per_m2",
    "people_per_km2",
    "people_per_cell",
    "relative_index",
]


@dataclass(frozen=True)
class MapMetadata:
    crs: str
    transform: Affine
    width: int
    height: int
    pixel_area_m2: float
    value_semantics: ValueSemantics
    nodata: Optional[float] = None


@dataclass(frozen=True)
class GeneratedMap:
    values: np.ndarray
    value_semantics: ValueSemantics
    # Optional for generators that know real-world calibration.
    calibration_scale: float = 1.0


class MapSource:
    @property
    def metadata(self) -> MapMetadata:
        ...

    def read_values(self) -> np.ndarray:
        ...

    def regenerate(self, rng: np.random.Generator | None = None) -> None:
        ...

    # Canonical read path consumed by wrappers/reward logic.
    def read_density_people_per_m2(self) -> np.ndarray:
        ...
```

## Proposed config model shape (example)

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any


class TiffMapSourceConfig(BaseModel):
    type: Literal["tiff"] = "tiff"
    file_path: str
    # Defaults can be inferred from dataset metadata if available.
    value_semantics: Literal[
        "people_per_m2", "people_per_km2", "people_per_cell"
    ] = "people_per_cell"


class RandomMapSourceConfig(BaseModel):
    type: Literal["cities", "polygon", "population_density"]
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    # Explicit spatial contract (pick one path).
    resolution_m: Optional[float] = None
    output_shape: Optional[tuple[int, int]] = None

    # Explicit value semantics for generated values.
    value_semantics: Literal[
        "people_per_m2", "people_per_km2", "people_per_cell", "relative_index"
    ] = "relative_index"

    # Optional calibration for relative synthetic maps.
    calibration_scale: float = 1.0
```

## Proposed generator contract (example)

Current generators return only `np.ndarray`. Proposed shape:

```python
from typing import Callable

GeneratorFn = Callable[[tuple[int, int], np.random.Generator | None], GeneratedMap]


def generate_cities(shape=(512, 512), rng=None, **kwargs) -> GeneratedMap:
    values = ...  # existing city generation logic
    return GeneratedMap(
        values=values.astype(np.float32),
        value_semantics="relative_index",
        calibration_scale=2000.0,  # example, tunable
    )


def generate_population_density(shape=(512, 512), rng=None, **kwargs) -> GeneratedMap:
    values = ...  # existing GRF logic
    return GeneratedMap(
        values=values.astype(np.float32),
        value_semantics="people_per_cell",  # or people_per_m2 if calibrated
    )


def generate_random_shapes_map(shape=(512, 512), rng=None, **kwargs) -> GeneratedMap:
    values = ...  # mask-like map
    return GeneratedMap(
        values=values.astype(np.float32),
        value_semantics="relative_index",
    )
```

## Conversion rules (example)

```python
def to_people_per_m2(values, semantics, pixel_area_m2, calibration_scale=1.0):
    if semantics == "people_per_m2":
        return values
    if semantics == "people_per_km2":
        return values / 1_000_000.0
    if semantics == "people_per_cell":
        return values / pixel_area_m2
    if semantics == "relative_index":
        return (values * calibration_scale) / pixel_area_m2
    raise ValueError(f"Unknown semantics: {semantics}")
```

## Example usage flow

1. `RandomMapSourceConfig.build_for_env(env)` computes map transform from env bounds.
2. If `resolution_m` is set, derive `(height, width)` from bounds; if `output_shape` is set, derive effective resolution.
3. Generator returns `GeneratedMap` with explicit semantics.
4. `MapSource` stores metadata (`pixel_area_m2`, semantics, nodata).
5. `Population` wrapper consumes only canonical `people_per_m2` arrays.

## End-to-end information flow to reward (current code + target behavior)

This is the concrete path from map data to reward computation.

### A) GeoTIFF source -> reward

1. **Config selection**  
   `PopulationConfig.map_source_config` is set to `TiffMapSourceConfig`.

2. **Map source construction**  
   `TiffMapSourceConfig.build_for_env(...)` calls `build()` and returns `TiffMapSource` in `bluesky_gym/maps/map_datasets.py`.  
   What happens: rasterio opens the file; CRS + transform + raster band are now available from the dataset.

3. **Sampler setup**  
   `Population.__init__` creates `RasterSampler(map_source=..., destination_crs=base_env.pygame_crs)` in `bluesky_gym/wrappers/population.py`.  
   What happens: sampler prepares a WGS84 -> destination CRS transformer and knows how to reproject from source map to requested windows.

4. **Per-step map extraction for noise**  
   In `Population._get_noise_reward`, the noise kernel size is requested from `NoiseModel.get_noise_power_kernel_shape_meters_and_pixels(...)`, then `RasterSampler.get_observation_clipped(...)` extracts that window from the map.

5. **Reprojection/resampling**  
   `RasterSampler._extract_view_from_map(...)` calls `rasterio.warp.reproject(...)` with source dataset CRS/transform and destination window transform.  
   What happens: values are sampled onto the noise-kernel pixel grid.

6. **Reward calculation**  
   `NoiseModel.step_normalized_noise(...)` computes weighted sum of `population_map_extract * noise_power_kernel`, normalizes by `mean_step_noise`, and `Population._get_noise_reward` converts this to a penalty.

7. **Unit note (important)**  
   Today, values are used as-is. With the proposed API, conversion to canonical `people_per_m2` happens in `MapSource` before step 4.

### B) Random generator source -> reward

1. **Config selection**  
   `PopulationConfig.map_source_config` is set to `RandomMapSourceConfig(type=...)`.

2. **Generator binding + spatial framing**  
   `RandomMapSourceConfig.build_for_env(...)` selects the generator (`cities`, `polygon`, or `population_density`) and calls `from_env_bounds(...)` in `bluesky_gym/maps/map_datasets.py`.  
   What happens: a map transform is created so generated pixels cover env bounds in `env.pygame_crs`.

3. **Initial generation**  
   `RandomMapSource.__init__` immediately calls `regenerate()`.  
   What happens: generator returns an array; array is written into an in-memory GeoTIFF (`rasterio.MemoryFile`) with CRS + transform.

4. **Episode regeneration**  
   On `Population.reset(...)`, `map_source.regenerate(rng=base_env.np_random)` is called.  
   What happens: a new synthetic raster is generated with episode-seeded RNG and replaces the in-memory dataset.

5. **Sampling + reward path**  
   From here, flow is identical to TIFF path: `RasterSampler.get_observation_clipped(...)` -> `Population._get_noise_reward(...)` -> `NoiseModel.step_normalized_noise(...)`.

6. **Unit note (important)**  
   Today, random generator values are unit-implicit. With the proposed API, generators return `GeneratedMap(value_semantics=...)`, and `MapSource` converts to canonical `people_per_m2` before sampling.

### What each stage produces

- **Map source stage:** raster + CRS + transform (+ semantics/metadata in proposed API).
- **Sampler stage:** extracted local population window in the noise kernel shape.
- **Noise model stage:** scalar step noise (weighted integral over window).
- **Reward stage:** normalized negative penalty added via `BaseNavigationEnv.add_reward_component(...)`.

## Example YAML sketch

```yaml
map_source:
  type: population_density
  resolution_m: 500
  value_semantics: people_per_cell
  kwargs:
    # generator-specific knobs
    len_scales: [1.71, 28.9, 80.2]
```

## Speeding up map generation: options

### Option 1: Multi-resolution generation (recommended first)

- Generate random maps at lower resolution (e.g. 128x128 or 256x256).
- Upsample once to env resolution with bilinear/cubic interpolation.
- Keep map transform physically correct.

Pros:
- Large speedups with minimal architecture changes.
- Good for smooth density fields (`cities`, GRF).

Cons:
- Can blur sharp boundaries (`polygon`) unless nearest-neighbor is used for masks.

---

### Option 2: Pre-generated map bank (recommended)

- Generate `N` maps at startup (or offline), store arrays in memory/mmap.
- On reset, pick one map by RNG index (optionally apply cheap perturbations).

Pros:
- Near-zero per-episode generation time.
- Deterministic and reproducible.

Cons:
- Less diversity unless `N` is large or perturbations are added.
- Startup time and memory footprint increase.

---

### Option 3: Async prefetch queue

- Background worker process/thread continuously generates upcoming maps.
- Training loop only dequeues prepared maps.

Pros:
- Hides generation latency behind policy learning.
- Preserves high diversity.

Cons:
- More complexity (IPC, shutdown behavior, RNG streams).
- Harder debugging if worker fails.

---

### Option 4: Cache static expensive components

- Precompute reusable kernels/FFT plans/model objects once.
- Reuse arrays and buffers in-place to avoid repeated allocations.

Pros:
- Improves both startup and per-episode costs.
- Minimal behavior change.

Cons:
- Requires careful refactor of generator internals.

---

### Option 5: JIT or vectorization improvements

- Move hotspot loops to `numba` or C-accelerated routines.
- Replace Python-level geometry loops with vectorized operations where possible.

Pros:
- Strong speedups for compute-heavy custom code.

Cons:
- Extra dependency/tooling complexity.
- Diminishing returns if bottleneck is already in C extensions.

---

### Option 6: Tune stochastic field complexity (`population_density`)

- Reduce output shape.
- Use fewer covariance components.
- Lower precision (`float32`) and avoid repeated model reconstruction.

Pros:
- Directly targets current slowest generator.

Cons:
- Can reduce map realism if over-tuned.

## Suggested rollout

1. Add explicit metadata contract (`MapMetadata`, `GeneratedMap`) in API only.
2. Convert one generator (`cities`) + one source path end-to-end.
3. Add conversion to `people_per_m2` in one centralized function.
4. Introduce Option 1 + Option 2 for immediate speedup.
5. Add Option 3 if generation still bottlenecks training.

## Open decisions

- Should synthetic `relative_index` maps be calibrated to realistic densities globally, or remain task-relative?
- Should `resolution_m` be required for random maps, with `output_shape` as a derived fallback, or vice versa?
- Should nodata handling be unified at map source level (`np.nan`) before wrappers consume values?


