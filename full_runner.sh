#!/bin/bash

# run with `nohup bash full_runner.sh &` to run in background

JOB1="datagen.sbatch"
JOB2="combiner.sbatch"
JOB3="estimator.sbatch"

# # Submit the first job and capture the job ID
JOB1_ID=$(sbatch $JOB1 | awk '{print $4}')

# # # Wait for the first job to complete
while squeue -j $JOB1_ID | grep -q $JOB1_ID; do
  sleep 1
done

# # Submit and wait for the second job
JOB2_ID=$(sbatch $JOB2 | awk '{print $4}')
while squeue -j $JOB2_ID | grep -q $JOB2_ID; do
  sleep 1
done

# Submit the final job
sbatch $JOB3
