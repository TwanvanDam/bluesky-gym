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
uv run scripts/backfill_best_model.py "$@"

echo "== generate_trajectories =="
uv run scripts/generate_trajectories.py "$@"

echo "== present_trajectories =="
uv run scripts/present_trajectories.py "$@"

echo "== plot_run_rewards =="
mkdir -p plots
uv run scripts/plot_run_rewards.py "$@"

echo "== generate_obsidian_notes =="
mapfile -t RESOLVED_PATHS < <(uv run python -c "
import sys
sys.path.insert(0, 'scripts')
from common.run_paths import resolve_run
from pathlib import Path
for ref in sys.argv[1:]:
    print(Path(resolve_run(ref).root).resolve())
" "$@")
(cd ~/Obsidian/Thesis && python generate_notes.py "${RESOLVED_PATHS[@]}")

echo "== done =="
