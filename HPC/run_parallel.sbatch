#!/bin/bash
#SBATCH --job-name=bluesky-rl-parallel
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu               # 1 Physical GPU shared by all tasks
#SBATCH --ntasks=4               # Max number of tasks running SIMULTANEOUSLY
#SBATCH --cpus-per-task=2        # CPUs per task (total CPUs = 4 * 2 = 8)
#SBATCH --mem=16G                # Total memory for the job (e.g., 4 tasks * 4GB)
#SBATCH --time=06:00:00
#SBATCH --output=HPC/logs/out/slurm-%j.out
#SBATCH --error=HPC/logs/err/slurm-%j.err

# ── Description ──────────────────────────────────────────────
# Runs multiple reinforcement learning experiments in parallel
# on a single GPU using background srun tasks.
#
# Usage:
#   1. Run specific files:
#      sbatch HPC/run_parallel.sbatch HPC/experiments/exp1.yaml HPC/experiments/exp2.yaml
#
#   2. Run ALL yaml files in a folder at once using a wildcard (*):
#      sbatch HPC/run_parallel.sbatch HPC/experiments/*.yaml
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
for CONFIG in "$@"; do
    # Extract the filename without the path and extension for clean log names
    # e.g., "HPC/experiments/test1.yaml" becomes "test1"
    BASENAME=$(basename "$CONFIG" .yaml)

    echo "Launching experiment: $BASENAME"
    echo "Logs -> HPC/logs/out/${BASENAME}_${SLURM_JOB_ID}.out"

    # --exclusive ensures each task gets its own dedicated 2 CPUs
    # --output and --error redirect this specific task's logs
    # The '&' sends it to the background
    srun --exclusive --ntasks=1 --cpus-per-task=2 --mem=4G \
        --output="HPC/logs/out/${BASENAME}_${SLURM_JOB_ID}.out" \
        --error="HPC/logs/err/${BASENAME}_${SLURM_JOB_ID}.err" \
        apptainer exec --nv \
        --bind "${PROJECT_DIR}:/workspace" \
        --pwd /workspace \
        "$CONTAINER" \
        python -m scripts.run_experiment "$CONFIG" &

    # Optional: slight delay to prevent race conditions during container startup
    sleep 2
done

# Wait for all background tasks to finish before ending the Slurm job
wait

echo "=========================================="
echo "End time     : $(date)"
echo "=========================================="