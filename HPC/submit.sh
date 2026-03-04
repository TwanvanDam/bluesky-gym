#!/bin/bash
# submit.sh - usage: ./submit.sh data.csv

FILE=$1
BASENAME=$(basename "$FILE" .yaml)   # strips path and extension

sbatch --job-name="training-${BASENAME}" myjob.sh "$FILE"