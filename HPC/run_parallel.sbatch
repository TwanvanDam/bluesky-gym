#!/bin/bash
#SBATCH --job-name=bluesky-rl-parallel
#SBATCH --partition=lr-co,general
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # 1 GPU shared by all background processes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # ~1 CPU per environment + overhead
#SBATCH --mem=40G                # ~6G per experiment
#SBATCH --time=00:01:00
#SBATCH --output=HPC/logs/out/slurm-%j.out
#SBATCH --error=HPC/logs/err/slurm-%j.err

# ── Description ──────────────────────────────────────────────
# Runs RL experiments in parallel on a single GPU (up to 6 at a time).
# All seeds are run for every config (cartesian product).
#
# Usage:
#   sbatch HPC/run_parallel.sbatch --seeds 0,1,2 --configs config1.yaml config2.yaml
#   sbatch HPC/run_parallel.sbatch --seeds=0,1,2 --configs config1.yaml config2.yaml
# ─────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "=========================================="

# ── Parse arguments ───────────────────────────────────────────
SEEDS=()
CONFIGS=()

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --seeds=*)
            val="${arg#--seeds=}"
            val="${val//[\[\]]/}"  # strip optional [ ] brackets
            IFS=',' read -ra SEEDS <<< "$val"
            i=$((i+1))
            ;;
        --seeds)
            i=$((i+1))
            val="${!i}"
            val="${val//[\[\]]/}"
            IFS=',' read -ra SEEDS <<< "$val"
            i=$((i+1))
            ;;
        --configs)
            i=$((i+1))
            while [ $i -le $# ] && [[ "${!i}" != --* ]]; do
                CONFIGS+=("${!i}")
                i=$((i+1))
            done
            ;;
        *)
            echo "WARNING: Unknown argument: $arg"
            i=$((i+1))
            ;;
    esac
done

if [ ${#SEEDS[@]} -eq 0 ]; then
    echo "ERROR: No seeds provided."
    echo "Usage: sbatch $0 --seeds 0,1,2 --configs config1.yaml config2.yaml"
    exit 1
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "ERROR: No configs provided."
    echo "Usage: sbatch $0 --seeds 0,1,2 --configs config1.yaml config2.yaml"
    exit 1
fi

# ── Build pairs (all configs × all seeds) ────────────────────
PAIR_CONFIGS=()
PAIR_SEEDS=()
for CONFIG in "${CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        PAIR_CONFIGS+=("$CONFIG")
        PAIR_SEEDS+=("$SEED")
    done
done

NPAIRS=${#PAIR_CONFIGS[@]}
echo "Configs      : ${CONFIGS[*]}"
echo "Seeds        : ${SEEDS[*]}"
echo "Total runs   : $NPAIRS"
echo "=========================================="

if [ "$NPAIRS" -gt 6 ]; then
    echo "WARNING: More than 6 pairs ($NPAIRS). Only the first 6 will run in this job."
fi

PROJECT_DIR=$(pwd)
CONTAINER="${PROJECT_DIR}/HPC/rl_env.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    exit 1
fi

mkdir -p HPC/logs/out HPC/logs/err

# ── Launch experiments ────────────────────────────────────────
COUNT=0
for idx in "${!PAIR_CONFIGS[@]}"; do
    [ $COUNT -ge 6 ] && break
    CONFIG="${PAIR_CONFIGS[$idx]}"
    SEED="${PAIR_SEEDS[$idx]}"
    BASENAME=$(basename "$CONFIG" .yaml)
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

    sleep 2
    COUNT=$((COUNT+1))
done

wait

echo "=========================================="
echo "End time     : $(date)"
echo "=========================================="
