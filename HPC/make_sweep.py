"""Batch and submit a hyperparameter sweep via run_parallel.sbatch.

Generates all (config, seed) pairs, splits them into batches of --batch-size,
and submits one run_parallel.sbatch job per batch.

Usage:
    python HPC/make_sweep.py \\
        --configs HPC/experiments/cfg1.yaml HPC/experiments/cfg2.yaml \\
        --seeds 0 1 2 \\
        [--batch-size 6] \\
        [--submit] \\
        [--test-only]
"""
import argparse
import subprocess
import sys
from itertools import islice
from pathlib import Path


def _chunked(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch and submit a hyperparameter sweep.")
    parser.add_argument("--configs", nargs="+", required=True, metavar="CONFIG",
                        help="Experiment YAML config paths.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, metavar="SEED",
                        help="Seeds to run for each config.")
    parser.add_argument("--batch-size", type=int, default=6, metavar="N",
                        help="Experiments per job (default: 6).")
    parser.add_argument("--submit", action="store_true",
                        help="Actually call sbatch. Without this flag, just print the commands.")
    parser.add_argument("--test-only", action="store_true",
                        help="Pass --test-only to sbatch (validates with SLURM without queuing). Implies --submit.")
    args = parser.parse_args()

    configs = [Path(c) for c in args.configs]
    for c in configs:
        if not c.exists():
            print(f"ERROR: Config not found: {c}", file=sys.stderr)
            sys.exit(1)

    pairs = [(str(c), s) for c in configs for s in args.seeds]
    batches = list(_chunked(pairs, args.batch_size))

    print(f"{len(pairs)} runs → {len(batches)} job(s) of up to {args.batch_size}\n")

    sbatch_script = "HPC/run_parallel.sbatch"
    for i, batch in enumerate(batches):
        flat = [x for config, seed in batch for x in (config, str(seed))]
        cmd = ["sbatch"] + (["--test-only"] if args.test_only else []) + [sbatch_script] + flat
        label = ", ".join(f"{Path(c).stem}:s{s}" for c, s in batch)
        print(f"Job {i + 1}/{len(batches)}: {label}")
        print(f"  {' '.join(cmd)}")
        if args.submit or args.test_only:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  sbatch failed:\n{result.stderr}", file=sys.stderr)
                sys.exit(result.returncode)
            print(f"  {result.stdout.strip()}")
        print()

    if not args.submit and not args.test_only:
        print("Dry run — pass --submit to actually submit.")


if __name__ == "__main__":
    main()
