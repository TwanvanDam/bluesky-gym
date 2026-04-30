#!/bin/bash
#SBATCH --job-name=bluesky-rl-parallel
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # 1 GPU shared by all background processes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7        # ~1 CPU per experiment + overhead
#SBATCH --mem=40G                # ~6G per experiment
#SBATCH --time=24:00:00
#SBATCH --output=HPC/logs/out/slurm-%j.out
#SBATCH --error=HPC/logs/err/slurm-%j.err

# ── Description ──────────────────────────────────────────────
# Runs up to 6 RL experiments in parallel on a single GPU using
# shell backgrounding. Arguments are config/seed pairs.
#
# Usage:
#   sbatch HPC/run_parallel.sbatch config1.yaml seed1 config2.yaml seed2 ...
#
# For sweeps, use make_sweep.py to batch and submit automatically:
#   python HPC/make_sweep.py --configs cfg1.yaml cfg2.yaml --seeds 0 1 2 --submit
# ─────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Args given   : $#"
echo "Start time   : $(date)"
echo "=========================================="

# ── Validation ───────────────────────────────────────────────
if [ "$#" -eq 0 ]; then
    echo "ERROR: No arguments provided."
    echo "Usage: sbatch $0 config1.yaml seed1 config2.yaml seed2 ..."
    exit 1
fi

if [ $(($# % 2)) -ne 0 ]; then
    echo "ERROR: Arguments must be config/seed pairs (got $# args, expected an even number)."
    exit 1
fi

NPAIRS=$(($# / 2))
if [ "$NPAIRS" -gt 6 ]; then
    echo "WARNING: More than 6 pairs provided ($NPAIRS). Only the first 6 will be run."
fi

PROJECT_DIR=$(pwd)
CONTAINER="${PROJECT_DIR}/HPC/rl_env.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    exit 1
fi

mkdir -p HPC/logs/out HPC/logs/err

# ── Run ──────────────────────────────────────────────────────
ARGS=("$@")
COUNT=0
i=0
while [ $i -lt ${#ARGS[@]} ] && [ $COUNT -lt 6 ]; do
    CONFIG="${ARGS[$i]}"
    SEED="${ARGS[$((i+1))]}"
    BASENAME=$(basename "$CONFIG" .yaml)
    RUNNAME="${BASENAME}_seed${SEED}"

    echo "Launching: $RUNNAME"
    echo "Logs -> HPC/logs/out/${RUNNAME}_${SLURM_JOB_ID}.out"

    apptainer exec --nv \
        --bind "${PROJECT_DIR}:/workspace" \
        --pwd /workspace \
        "$CONTAINER" \
        python -m scripts.run_experiment "$CONFIG" \
            --seed "$SEED" \
            --slurm-job-id "$SLURM_JOB_ID" \
            --slurm-log-out "HPC/logs/out/${RUNNAME}_${SLURM_JOB_ID}.out" \
            --slurm-log-err "HPC/logs/err/${RUNNAME}_${SLURM_JOB_ID}.err" \
        >"HPC/logs/out/${RUNNAME}_${SLURM_JOB_ID}.out" \
        2>"HPC/logs/err/${RUNNAME}_${SLURM_JOB_ID}.err" &

    sleep 2
    i=$((i+2))
    COUNT=$((COUNT+1))
done

wait

echo "=========================================="
echo "End time     : $(date)"
echo "=========================================="
