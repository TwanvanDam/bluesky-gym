"""Backfill best_model.zip for completed runs that lack one.

For each run, reads rollout/ep_rew_mean from TensorBoard, computes the
mean over the interval between the previous checkpoint and each
checkpoint, and copies the checkpoint with the highest mean to
best_model.zip.

Usage:
    python -m scripts.backfill_best_model PopulationWrapper-v0/RealMap_base_2026-...
    python -m scripts.backfill_best_model --dry-run PopulationWrapper-v0/RealMap_base_2026-...
    python -m scripts.backfill_best_model --overwrite PopulationWrapper-v0/RealMap_base_2026-...
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from scripts.common.run_paths import RunPaths, resolve_run, update_metadata

REWARD_TAG = "rollout/ep_rew_mean"
_CKPT_RE = re.compile(r"checkpoint_(\d+)_steps\.zip")


def _load_reward_series(run: RunPaths) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (steps, values) for rollout/ep_rew_mean, or None if unavailable."""
    event_files = sorted(run.tensorboard_dir.glob("*"))
    if not event_files:
        return None
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()
    if REWARD_TAG not in ea.Tags().get("scalars", []):
        return None
    scalars = ea.Scalars(REWARD_TAG)
    steps = np.array([s.step for s in scalars])
    values = np.array([s.value for s in scalars])
    return steps, values


def _windowed_mean(steps: np.ndarray, values: np.ndarray,
                   prev_step: int, at_step: int) -> float | None:
    """Mean reward over the interval (prev_step, at_step]."""
    mask = (steps > prev_step) & (steps <= at_step)
    if mask.any():
        return float(values[mask].mean())
    # Fall back to the last value before this checkpoint
    before = steps <= at_step
    if before.any():
        return float(values[before][-1])
    return None


def _checkpoint_steps(run: RunPaths) -> dict[int, Path]:
    """Return {step: path} for all step-based checkpoints."""
    result = {}
    for p in run.checkpoints_dir.glob("checkpoint_*_steps.zip"):
        m = _CKPT_RE.match(p.name)
        if m:
            result[int(m.group(1))] = p
    return result


def process_run(run: RunPaths, dry_run: bool, overwrite: bool, verbose: bool) -> str:
    if run.best_model.exists() and not overwrite:
        return "skip (best_model.zip already exists)"

    reward_data = _load_reward_series(run)
    if reward_data is None:
        return "skip (no TensorBoard reward data)"

    steps_tb, values_tb = reward_data
    checkpoints = _checkpoint_steps(run)
    if not checkpoints:
        return "skip (no checkpoints)"

    scored = {}
    prev_step = 0
    for step in sorted(checkpoints):
        mean = _windowed_mean(steps_tb, values_tb, prev_step, step)
        if mean is not None:
            if verbose:
                print(step,mean)
            scored[step] = (mean, checkpoints[step])
        prev_step = step

    if not scored:
        return "skip (could not score any checkpoint)"

    best_step = max(scored, key=lambda s: scored[s][0])
    best_mean, best_path = scored[best_step]

    if dry_run:
        return f"would copy {best_path.name} (windowed_mean={best_mean:.4f})"

    shutil.copy2(best_path, run.best_model)
    update_metadata(run, best_checkpoint=best_path.name, windowed_mean=f"{best_mean:.4f}")
    return f"copied {best_path.name} (windowed_mean={best_mean:.4f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_refs", nargs="+",
                        help="Run reference(s) (e.g. 'PopulationWrapper-v0/RealMap_base_2026-...')")
    parser.add_argument("--dry-run", action="store_true", help="print actions without copying")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing best_model.zip")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    args = parser.parse_args()

    runs = [resolve_run(r) for r in args.run_refs]

    for run in runs:
        status = process_run(run, dry_run=args.dry_run, overwrite=args.overwrite, verbose=args.verbose)
        print(f"{run.run_id}: {status}")


if __name__ == "__main__":
    main()
