#!/usr/bin/env bash
# Generate + plot trajectories for every run under the given root(s).
#
# Both scripts accept many run refs and call bs.init() once, so each is invoked
# a single time over the whole batch for efficiency. Existing trajectory subdirs
# are skipped by generate_trajectories, so this is safe to re-run/resume.
#
# Usage:
#   bash scripts/generate_all_trajectories.sh                 # default root: runs
#   bash scripts/generate_all_trajectories.sh runs/transforms runs/convergence
set -euo pipefail

ROOTS=("$@")
[ "${#ROOTS[@]}" -eq 0 ] && ROOTS=("runs")

# All run directories (a run is a dir containing config.yaml) under the roots.
mapfile -t RUNS < <(find "${ROOTS[@]}" -type f -name config.yaml -printf '%h\n' | sort -u)
if [ "${#RUNS[@]}" -eq 0 ]; then
    echo "No runs (config.yaml) found under: ${ROOTS[*]}" >&2
    exit 1
fi

echo "Found ${#RUNS[@]} runs."

echo "== generate_trajectories (${#RUNS[@]} runs) =="
uv run scripts/generate_trajectories.py --runway EHAM/RW27 --lat_lon 52.3322 4.75 "${RUNS[@]}"
uv run scripts/generate_trajectories.py --runway EDDF/RW25R  "${RUNS[@]}"

echo "== present_trajectories (${#RUNS[@]} runs) =="
uv run scripts/present_trajectories.py ${PRESENT_ARGS:-} "${RUNS[@]}"

echo "== done =="
