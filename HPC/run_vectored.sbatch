#!/bin/bash
#SBATCH --job-name=bluesky-rl-parallel
#SBATCH --partition=lr-co,general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # 1 GPU shared by all background processes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # 3 seeds × (4 env workers + 1 main) + 1 spare
#SBATCH --mem=24G                 # ~8G per experiment × 3 seeds
#SBATCH --time=24:00:00
#SBATCH --output=HPC/logs/out/slurm-%j.out
#SBATCH --error=HPC/logs/err/slurm-%j.err

# ── Description ──────────────────────────────────────────────
# Runs one experiment config for seeds 0, 1, 2 in parallel on a single GPU.
# Designed for n_envs=4 configs (each seed uses 5 CPUs).
#
# Usage:
#   sbatch HPC/run_parallel.sbatch HPC/experiments/config.yaml
# ─────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "=========================================="

CONFIG="$1"

if [ -z "$CONFIG" ]; then
    echo "ERROR: No config provided."
    echo "Usage: sbatch $0 HPC/experiments/config.yaml"
    exit 1
fi

PROJECT_DIR=$(pwd)
CONTAINER="${PROJECT_DIR}/HPC/rl_env.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    exit 1
fi

BASENAME=$(basename "$CONFIG" .yaml)
echo "Config       : $CONFIG"
echo "Seeds        : 0 1 2"
echo "=========================================="

mkdir -p HPC/logs/out HPC/logs/err

# ── Launch 3 seeds in parallel ────────────────────────────────
for SEED in 0 1 2; do
    RUNNAME="${BASENAME}_seed${SEED}"
    echo "Launching: $RUNNAME"

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
done

wait

echo "=========================================="
echo "End time     : $(date)"
echo "=========================================="
