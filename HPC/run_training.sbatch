#!/bin/bash
#SBATCH --job-name=bluesky-rl-training
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=HPC/logs/slurm-%j.out
#SBATCH --error=HPC/logs/slurm-%j.err

# ── Description ──────────────────────────────────────────────
# Slurm batch script for training on the DAIC (TU Delft).
# Uses the Apptainer/Singularity container built from container.def.
#
# Usage:
#   sbatch HPC/run_training.sbatch HPC/experiments/Test.yaml  # runs a single experiment
# ─────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname)"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "GPUs         : $SLURM_GPUS_ON_NODE"
echo "Start time   : $(date)"
echo "=========================================="

# ── Paths ────────────────────────────────────────────────────
PROJECT_DIR=$(pwd)
CONTAINER="${PROJECT_DIR}/HPC/rl_env.sif"

# Verify the container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    echo "Build it first with: Apptainer build HPC/rl_env.sif HPC/container.def"
    exit 1
fi

# ── Run ──────────────────────────────────────────────────────
# Bind the project directory into the container at /workspace
# Pass an experiment config path as argument

echo "Running single experiment: $1"
apptainer exec --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --pwd /workspace \
    "$CONTAINER" \
    python -m scripts.run_experiment "$1"

echo "=========================================="
echo "End time     : $(date)"
echo "=========================================="

