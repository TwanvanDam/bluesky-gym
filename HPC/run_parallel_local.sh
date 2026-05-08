#!/usr/bin/env bash

set -uo pipefail

# Runs local experiments concurrently and queues the rest.
#
# Usage:
#   bash HPC/run_parallel_local.sh --max-parallel 2 --configs HPC/experiments/centered_1.yaml HPC/experiments/forward_1.yaml
#   bash HPC/run_parallel_local.sh -j 3 --seeds 0,1,2 --configs HPC/experiments/centered_1.yaml HPC/experiments/forward_1.yaml
#   PYTHON_BIN=.venv/bin/python bash HPC/run_parallel_local.sh -j 2 --seeds 0,1 --configs HPC/experiments/*.yaml

usage() {
    echo "Usage: $0 [--max-parallel N|-j N] [--seeds 0,1,2] --configs config1.yaml [config2.yaml ...]" >&2
    echo "  Note: --configs must be the last flag; it collects all remaining non-flag arguments." >&2
}

SEEDS=()
CONFIGS=()
MAX_PARALLEL=2
PYTHON_BIN="${PYTHON_BIN:-python}"

i=1
while [ "$i" -le "$#" ]; do
    arg="${!i}"
    case "$arg" in
        --max-parallel=*)
            MAX_PARALLEL="${arg#--max-parallel=}"
            i=$((i + 1))
            ;;
        --max-parallel|-j)
            i=$((i + 1))
            if [ "$i" -gt "$#" ]; then
                echo "ERROR: $arg requires a positive integer." >&2
                usage
                exit 1
            fi
            MAX_PARALLEL="${!i}"
            i=$((i + 1))
            ;;
        --seeds=*)
            val="${arg#--seeds=}"
            val="${val//[\[\]]/}"
            IFS=',' read -ra SEEDS <<< "$val"
            i=$((i + 1))
            ;;
        --seeds)
            i=$((i + 1))
            if [ "$i" -gt "$#" ]; then
                echo "ERROR: --seeds requires a comma-separated value." >&2
                usage
                exit 1
            fi
            val="${!i}"
            val="${val//[\[\]]/}"
            IFS=',' read -ra SEEDS <<< "$val"
            i=$((i + 1))
            ;;
        --configs)
            i=$((i + 1))
            while [ "$i" -le "$#" ] && [[ "${!i}" != --* ]]; do
                CONFIGS+=("${!i}")
                i=$((i + 1))
            done
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $arg" >&2
            usage
            exit 1
            ;;
    esac
done

trim_array() {
    local -n arr=$1
    local value
    local trimmed=()

    for value in "${arr[@]}"; do
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        if [ -n "$value" ]; then
            trimmed+=("$value")
        fi
    done

    arr=("${trimmed[@]}")
}

trim_array SEEDS
trim_array CONFIGS

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-parallel must be a positive integer, got: $MAX_PARALLEL" >&2
    exit 1
fi

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "ERROR: No configs provided." >&2
    usage
    exit 1
fi

for config in "${CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo "ERROR: Config not found: $config" >&2
        exit 1
    fi
done

PAIR_CONFIGS=()
PAIR_SEEDS=()
if [ "${#SEEDS[@]}" -eq 0 ]; then
    for config in "${CONFIGS[@]}"; do
        PAIR_CONFIGS+=("$config")
        PAIR_SEEDS+=("")
    done
else
    for config in "${CONFIGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            PAIR_CONFIGS+=("$config")
            PAIR_SEEDS+=("$seed")
        done
    done
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="HPC/logs/local/${RUN_ID}"
mkdir -p "$LOG_DIR"

RUNNING=0
FAILURES=0
declare -a CHILD_PIDS=()

stop_children() {
    local pid

    echo
    echo "Stopping running experiments..."
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}

trap stop_children INT TERM

wait_for_one() {
    local status

    wait -n
    status=$?

    RUNNING=$((RUNNING - 1))
    if [ "$status" -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
}

run_with_log() {
    local run_name="$1" log_out="$2" log_err="$3"
    shift 3
    "$@" >"$log_out" 2>"$log_err"
    local status=$?
    if [ "$status" -ne 0 ]; then
        echo "WARNING: $run_name failed (exit $status)." >&2
    fi
    return "$status"
}

echo "=========================================="
echo "Local run ID : $RUN_ID"
echo "Host         : $(hostname)"
echo "Python       : $PYTHON_BIN"
echo "Max parallel : $MAX_PARALLEL"
echo "Configs      : ${CONFIGS[*]}"
if [ "${#SEEDS[@]}" -gt 0 ]; then
    echo "Seeds        : ${SEEDS[*]}"
else
    echo "Seeds        : none"
fi
echo "Total runs   : ${#PAIR_CONFIGS[@]}"
echo "Logs         : $LOG_DIR"
echo "Start time   : $(date)"
echo "=========================================="

for idx in "${!PAIR_CONFIGS[@]}"; do
    config="${PAIR_CONFIGS[$idx]}"
    seed="${PAIR_SEEDS[$idx]}"
    run_label="${config##*/}"
    run_label="${run_label%.yaml}"

    if [ -n "$seed" ]; then
        run_name="${run_label}_seed${seed}"
    else
        run_name="$run_label"
    fi

    log_out="${LOG_DIR}/${run_name}.out"
    log_err="${LOG_DIR}/${run_name}.err"

    if [ -n "$seed" ]; then
        command=("$PYTHON_BIN" -m scripts.run_experiment "$config" --seed "$seed" --slurm-log-out "$log_out" --slurm-log-err "$log_err")
    else
        command=("$PYTHON_BIN" -m scripts.run_experiment "$config" --slurm-log-out "$log_out" --slurm-log-err "$log_err")
    fi


    while [ "$RUNNING" -ge "$MAX_PARALLEL" ]; do
        wait_for_one
    done

    echo "Launching: $run_name -> $log_out"
    run_with_log "$run_name" "$log_out" "$log_err" "${command[@]}" &
    CHILD_PIDS+=("$!")
    RUNNING=$((RUNNING + 1))
done

while [ "$RUNNING" -gt 0 ]; do
    wait_for_one
done

trap - INT TERM

echo "=========================================="
echo "End time     : $(date)"
echo "Failures     : $FAILURES"
echo "Logs         : $LOG_DIR"
echo "=========================================="

if [ "$FAILURES" -ne 0 ]; then
    exit 1
fi
