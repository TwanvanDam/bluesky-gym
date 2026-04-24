#!/bin/bash
# Auto-copy SLURM logs to run directories based on metadata.json.
#
# For each run under runs/, reads metadata.json to find slurm_job_id and config_stem,
# then copies matching log files from HPC/logs/ into <run_dir>/slurm/.
#
# Skips runs that already have logs or lack a slurm_job_id in metadata.
# The main slurm script log (slurm-{job_id}) is copied to every run sharing that job_id.
#
# Usage:
#   bash scripts/auto_copy_slurm_logs.sh              # dry-run
#   bash scripts/auto_copy_slurm_logs.sh --execute    # actually copy

set -euo pipefail

EXECUTE=false
if [[ "${1:-}" == "--execute" ]]; then
    EXECUTE=true
fi

RUNS_DIR="runs"
LOG_OUT_DIR="HPC/logs/out"
LOG_ERR_DIR="HPC/logs/err"

n_copied=0
n_skipped=0
n_no_meta=0

copy_file() {
    local src="$1"
    local dest_dir="$2"
    local dest="$dest_dir/$(basename "$src")"
    if [[ "$EXECUTE" == true ]]; then
        cp "$src" "$dest"
        echo "  Copied: $src -> $dest"
    else
        echo "  [dry-run] Would copy: $src -> $dest"
    fi
    n_copied=$((n_copied + 1))
}

while IFS= read -r meta; do
    run_dir=$(dirname "$meta")
    slurm_dir="$run_dir/slurm"

    # Skip if slurm dir already has files
    if [[ -d "$slurm_dir" ]] && [[ -n "$(ls -A "$slurm_dir" 2>/dev/null)" ]]; then
        echo "Skipping (logs already present): $run_dir"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # Parse metadata
    slurm_job_id=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('slurm_job_id',''))" "$meta")
    config_stem=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('config_stem',''))" "$meta")

    if [[ -z "$slurm_job_id" ]]; then
        echo "Skipping (no slurm_job_id): $run_dir"
        n_no_meta=$((n_no_meta + 1))
        continue
    fi

    echo "Processing: $run_dir (job=$slurm_job_id, config=$config_stem)"

    [[ "$EXECUTE" == true ]] && mkdir -p "$slurm_dir"

    found=0

    # Config-specific logs: {config_stem}_{job_id}.{out,err}
    if [[ -n "$config_stem" ]]; then
        for f in \
            "$LOG_OUT_DIR/${config_stem}_${slurm_job_id}.out" \
            "$LOG_ERR_DIR/${config_stem}_${slurm_job_id}.err"; do
            if [[ -f "$f" ]]; then
                copy_file "$f" "$slurm_dir"
                found=$((found + 1))
            fi
        done
    fi

    # Main slurm script logs: slurm-{job_id}.{out,err} — shared across all runs in the job
    for f in \
        "$LOG_OUT_DIR/slurm-${slurm_job_id}.out" \
        "$LOG_ERR_DIR/slurm-${slurm_job_id}.err"; do
        if [[ -f "$f" ]]; then
            copy_file "$f" "$slurm_dir"
            found=$((found + 1))
        fi
    done

    if [[ $found -eq 0 ]]; then
        echo "  No log files found for job $slurm_job_id"
    fi

done < <(find "$RUNS_DIR" -name "metadata.json" | sort)

echo ""
echo "Done. Copied=$n_copied, Skipped(logs present)=$n_skipped, Skipped(no job id)=$n_no_meta"
if [[ "$EXECUTE" == false ]]; then
    echo "(dry-run — pass --execute to actually copy)"
fi
