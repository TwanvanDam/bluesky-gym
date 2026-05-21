#!/bin/bash
#SBATCH --job-name=bluesky-rl-parallel
#SBATCH --partition=lr-co,general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # 1 GPU shared by all background processes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # 2 configs × 3 seeds × ~2 CPU/seed + spare
#SBATCH --mem=64G                # 2 configs × 3 seeds × ~8G/seed
#SBATCH --time=20:00:00
#SBATCH --output=HPC/logs/out/slurm-%j.out
#SBATCH --error=HPC/logs/err/slurm-%j.err

# ── Description ──────────────────────────────────────────────
# Runs two experiment configs for seeds 0, 1, 2 in parallel on a single GPU
# (6 runs total). Seeds are fixed; configs are positional arguments.
#
# Usage:
#   sbatch HPC/run_vectored.sbatch HPC/experiments/config1.yaml HPC/experiments/config2.yaml
# ─────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "=========================================="

CONFIG1="$1"
CONFIG2="$2"

if [ -z "$CONFIG1" ] || [ -z "$CONFIG2" ]; then
    echo "ERROR: Two config files required."
    echo "Usage: sbatch $0 HPC/experiments/config1.yaml HPC/experiments/config2.yaml"
    exit 1
fi

PROJECT_DIR=$(pwd)
CONTAINER="${PROJECT_DIR}/HPC/rl_env.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    exit 1
fi

echo "Config 1     : $CONFIG1"
echo "Config 2     : $CONFIG2"
echo "Seeds        : 0 1 2"
echo "Total runs   : 6"
echo "=========================================="

mkdir -p HPC/logs/out HPC/logs/err

# ── Launch 2 configs × 3 seeds in parallel (cap: 6) ──────────
COUNT=0
for CONFIG in "$CONFIG1" "$CONFIG2"; do
    BASENAME=$(basename "$CONFIG" .yaml)
    for SEED in 0 1 2; do
        [ $COUNT -ge 6 ] && break 2
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

        COUNT=$((COUNT+1))
    done
done

wait

echo "=========================================="
echo "End time     : $(date)"
echo "=========================================="
