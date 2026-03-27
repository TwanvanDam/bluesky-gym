#!/bin/bash
#SBATCH --job-name=bluesky-test
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=HPC/logs/out/test-%j.out
#SBATCH --error=HPC/logs/err/test-%j.err

# Usage: sbatch HPC/run_test.sbatch HPC/experiments/exp1.yaml

PROJECT_DIR=$(pwd)
CONTAINER="${PROJECT_DIR}/HPC/rl_env.sif"
CONFIG="${1:?ERROR: No config file provided. Usage: sbatch $0 path/to/config.yaml}"
BASENAME=$(basename "$CONFIG" .yaml)

mkdir -p HPC/logs/out HPC/logs/err

echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Config: $BASENAME | Start: $(date)"

apptainer exec --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --pwd /workspace \
    "$CONTAINER" \
    python -m scripts.run_experiment "$CONFIG" \
    >"HPC/logs/out/${BASENAME}_${SLURM_JOB_ID}.out" \
    2>"HPC/logs/err/${BASENAME}_${SLURM_JOB_ID}.err"

echo "Done: $(date)"
