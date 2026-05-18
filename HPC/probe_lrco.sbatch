#!/bin/bash
#SBATCH --job-name=lrco-probe
#SBATCH --partition=lr-co
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Probe whether lr-co compute nodes can see the umbrella project storage.
# Output goes to the submission directory via relative %x-%j paths; run this
# from $HOME so the logs land on a filesystem lr-co is guaranteed to mount.

UMBRELLA=/tudelft.net/staff-umbrella/phdJan/Twan/Thesis

echo "Host         : $(hostname)"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "Submit dir   : $SLURM_SUBMIT_DIR"
echo "--- umbrella visible? ---"
ls -ld "$UMBRELLA" 2>&1
echo "--- can read repo contents? ---"
ls "$UMBRELLA" 2>&1 | head
echo "--- umbrella mounts ---"
mount 2>/dev/null | grep -i umbrella || echo "(no umbrella mount found)"
echo "DONE"
