#!/bin/bash
#SBATCH --job-name=bluesky-rl-parallel
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # 1 GPU (any type) shared by all background processes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # Total CPUs shared by all experiments (2 cpus per experiment if running 4 in parallel)
#SBATCH --mem=24G                # Total memory (~6G per experiment)
#SBATCH --time=06:00:00
#SBATCH --output=HPC/logs/out/slurm-%j.out
#SBATCH --error=HPC/logs/err/slurm-%j.err

# ── Description ──────────────────────────────────────────────
# Runs 4 reinforcement learning experiments in parallel on a single
# GPU using shell backgrounding (no srun sub-steps).
#
# Usage:
#   sbatch HPC/run_parallel.sbatch HPC/experiments/exp1.yaml HPC/experiments/exp2.yaml
# ─────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Configs given: $#"
echo "Start time   : $(date)"
echo "=========================================="

# ── Validation ───────────────────────────────────────────────
if [ "$#" -eq 0 ]; then
    echo "ERROR: No configuration files provided."
    echo "Usage: sbatch $0 path/to/config1.yaml path/to/config2.yaml ..."
    exit 1
fi

if [ "$#" -gt 4 ]; then
    echo "WARNING: More than 2 configs provided ($#). Only the first 2 will be run."
fi

PROJECT_DIR=$(pwd)
CONTAINER="${PROJECT_DIR}/HPC/rl_env.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    exit 1
fi

# Ensure log directories exist so the job doesn't fail trying to write to them
mkdir -p HPC/logs/out HPC/logs/err

# ── Run ──────────────────────────────────────────────────────
# Loop over all arguments passed via the command line ("$@")
for CONFIG in "${@:1:4}"; do
    # Extract the filename without the path and extension for clean log names
    # e.g., "HPC/experiments/test1.yaml" becomes "test1"
    BASENAME=$(basename "$CONFIG" .yaml)

    echo "Launching experiment: $BASENAME"
    echo "Logs -> HPC/logs/out/${BASENAME}_${SLURM_JOB_ID}.out"

    # Run directly with shell backgrounding — no srun sub-steps needed
    apptainer exec --nv \
        --bind "${PROJECT_DIR}:/workspace" \
        --pwd /workspace \
        "$CONTAINER" \
        python -m scripts.run_experiment "$CONFIG" \
        >"HPC/logs/out/${BASENAME}_${SLURM_JOB_ID}.out" \
        2>"HPC/logs/err/${BASENAME}_${SLURM_JOB_ID}.err" &

    # Slight delay to prevent race conditions during container startup
    sleep 2
done

# Wait for all background tasks to finish before ending the Slurm job
wait

echo "=========================================="
echo "End time     : $(date)"
echo "=========================================="