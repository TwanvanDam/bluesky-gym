#!/usr/bin/env bash
# Post-training pipeline: backfill best_model.zip, generate trajectories, plot them.
#
# Usage:
#   bash scripts/postprocess_run.sh <run_ref> [<run_ref> ...]
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <run_ref> [<run_ref> ...]" >&2
    exit 1
fi

echo "== backfill_best_model =="
uv run scripts.backfill_best_model.py "$@"

echo "== generate_trajectories =="
uv run scripts.generate_trajectories.py "$@"

echo "== present_trajectories =="
uv run scripts.present_trajectories.py "$@"

echo "== done =="
