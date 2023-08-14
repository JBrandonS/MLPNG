#!/bin/bash
# run with `nohup bash datagenerator.sh &`

set -e

SETTINGSFILE="settings/ul_nn_128.json"

JOB1="datagen/datagen.sbatch"
JOB2="datagen/combiner.sbatch"
JOB3="datagen/estimator.sbatch"

# Submit the first job and capture the job ID
JOB1_ID=$(sbatch $JOB1 $SETTINGSFILE | awk '{print $4}')
mkdir -p logs/datagen/$JOB1_ID
while squeue -j $JOB1_ID | grep -q $JOB1_ID; do
  sleep 1
done

# # # # Submit and wait for the second job
JOB2_ID=$(sbatch $JOB2 $SETTINGSFILE | awk '{print $4}')
mkdir -p logs/combiner/$JOB2_ID
while squeue -j $JOB2_ID | grep -q $JOB2_ID; do
  sleep 1
done

# # Submit the final job and let it run
sbatch $JOB3 $SETTINGSFILE
mkdir -p logs/estimator/$JOB3_ID
