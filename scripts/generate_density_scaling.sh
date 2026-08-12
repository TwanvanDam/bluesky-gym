#!/usr/bin/env bash
# Generate density-scaled trajectories for the implicit fuel-weight sweep.
#
# For every run under <runs_root>, fly on an in-distribution runway with the OBSERVED
# population density multiplied by each alpha (generate_trajectories.py --scale_density).
# Fuel/noise are measured later against the TRUE (unscaled) density, so each alpha is one
# operating point on that config's fuel-noise frontier.
#
# Special cases, detected from the run-directory name:
#   *no_map*                       -> flown on a zeroed map (--no_map); scaling is a no-op,
#                                     so a single alpha=1 point is generated (fixed reference).
#   *E_3_256* / *groot* / *legacy* -> legacy benchmark: cannot be scaled and has no
#                                     in-distribution (EDDF) eval, so it is skipped entirely.
#   everything else (map configs)  -> full alpha sweep.
#
# Trajectories land in <run>/trajectories/{RUNWAY}_scale_{alpha}/ and generate_trajectories
# skips folders that already exist, so re-running is safe/idempotent.
#
# Usage:
#   bash scripts/generate_density_scaling.sh runs/generalization
#   RUNWAY=EDDF/RW25R ALPHAS="0.25 0.5 1 2 4" \
#       bash scripts/generate_density_scaling.sh runs/density_scaling
#
# Then plot with:
#   python -m scripts.plot_density_scaling_sweep runs/generalization --runway EDDF_RW25R
set -euo pipefail

RUNS_ROOT="${1:?Usage: $0 <runs_root>}"
RUNWAY="${RUNWAY:-EDDF/RW25R}"
ALPHAS="${ALPHAS:-0.25 0.5 1 2 4}"
START_DISTANCE="${START_DISTANCE:-250}"

if [ ! -d "$RUNS_ROOT" ]; then
    echo "Directory not found: $RUNS_ROOT" >&2
    exit 1
fi

gen() {  # gen <run_dir> <label> [extra generate_trajectories args...]
    local run_dir="$1"; local label="$2"; shift 2
    uv run scripts/generate_trajectories.py "$run_dir" \
        --runway "$RUNWAY" --start_distance "$START_DISTANCE" --label "$label" "$@"
}

for run_dir in "$RUNS_ROOT"/*/; do
    [ -d "$run_dir" ] || continue
    name="$(basename "$run_dir")"
    echo ""
    echo "=== $name ==="
    case "$name" in
        *no_map*)
            # No density input: a single zeroed-map run is the fixed reference point.
            gen "$run_dir" "scale_1" --no_map
            ;;
        *E_3_256*|*groot*|*legacy*)
            # Legacy benchmark: unscalable and no in-distribution eval -> not part of this sweep.
            echo "-- skipping legacy benchmark (not used in the density-scaling sweep)"
            ;;
        *)
            for a in $ALPHAS; do
                echo "-- alpha=$a"
                gen "$run_dir" "scale_$a" --scale_density "$a"
            done
            ;;
    esac
done

echo ""
echo "== done =="
echo "Plot with: python -m scripts.plot_density_scaling_sweep $RUNS_ROOT --runway ${RUNWAY//\//_}"
