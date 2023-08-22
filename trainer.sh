#!/bin/bash
# run with `nohup bash datagenerator.sh &`

JOB1="scripts/trainer.sbatch"

mkdir -p logs/training

# Submit the first job and capture the job ID
JOB1_ID=$(sbatch $JOB1 | awk '{print $4}')
# while squeue -j $JOB1_ID | grep -q $JOB1_ID; do
#   sleep 1
# done
