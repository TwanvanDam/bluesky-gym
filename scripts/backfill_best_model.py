"""Backfill best_model.zip for completed runs that lack one.

For each run, reads rollout/ep_rew_mean from TensorBoard, computes a
windowed mean over the 50 k steps preceding each checkpoint, and copies
the checkpoint with the highest windowed mean to best_model.zip.

Usage:
    python -m scripts.backfill_best_model --env PopulationWrapper-v0
    python -m scripts.backfill_best_model --env PopulationWrapper-v0 --dry-run
    python -m scripts.backfill_best_model --env PopulationWrapper-v0 --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from scripts.common.run_paths import RunPaths, iter_runs, update_metadata

WINDOW_STEPS = 500
REWARD_TAG = "rollout/ep_rew_mean"
_CKPT_RE = re.compile(r"checkpoint_(\d+)_steps\.zip")


def _load_reward_series(tb_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (steps, values) for rollout/ep_rew_mean, or None if unavailable."""
    event_files = sorted(tb_dir.glob("*"))
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


def _windowed_mean(steps: np.ndarray, values: np.ndarray, at_step: int) -> float | None:
    mask = (steps > at_step - WINDOW_STEPS) & (steps <= at_step)
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


def process_run(run: RunPaths, dry_run: bool, overwrite: bool) -> str:
    if run.best_model.exists() and not overwrite:
        return "skip (best_model.zip already exists)"

    reward_data = _load_reward_series(run.tensorboard_dir)
    if reward_data is None:
        return "skip (no TensorBoard reward data)"

    steps_tb, values_tb = reward_data
    checkpoints = _checkpoint_steps(run)
    if not checkpoints:
        return "skip (no checkpoints)"

    scored = {}
    for step, path in checkpoints.items():
        mean = _windowed_mean(steps_tb, values_tb, step)
        if mean is not None:
            scored[step] = (mean, path)

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
    parser.add_argument("--env", required=True, help="env name, e.g. PopulationWrapper-v0")
    parser.add_argument("--dry-run", action="store_true", help="print actions without copying")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing best_model.zip")
    args = parser.parse_args()

    try:
        runs = iter_runs(args.env)
    except FileNotFoundError as e:
        raise SystemExit(e)

    if not runs:
        raise SystemExit(f"No run directories found for env '{args.env}'")

    for run in runs:
        status = process_run(run, dry_run=args.dry_run, overwrite=args.overwrite)
        print(f"{run.run_name}: {status}")


if __name__ == "__main__":
    main()