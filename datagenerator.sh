#!/bin/bash

# run with `nohup bash datagenerator.sh &`

# exit on error
set -e

echo "Script PID: $$"

SETTINGS_DIR="settings/"

# SETTINGS=(
#   "ul_nn_128.json" "ul_nn_256.json" "ul_nn_512.json" "ul_nn_1024.json" "ul_nn_2048.json"
#   "l_nn_128.json"  "l_nn_256.json"  "l_nn_512.json"  "l_nn_1024.json"  "l_nn_2048.json"
#   "l_128.json"     "l_256.json"     "l_512.json"     "l_1024.json"     "l_2048.json"
# )

SETTINGS=(
  # "ul_nn_128_large.json" 
  # "l_128_large.json" 
  #"ul_nn_512_large.json" 
  # "l_512_large.json"
  "l_2048.json"
)

JOB1="sbatch/datagen.sbatch"
JOB2="sbatch/combiner.sbatch"
JOB3="sbatch/estimator.sbatch"

# Uncomment to run the heidelberg estimator test
sbatch "sbatch/heidelberg_est.sbatch"
# exit 0

for x in "${SETTINGS[@]}"
do
    SETTINGSFILE="$SETTINGS_DIR$x"

    # Submit the first job and capture the job ID
    JOB1_ID=$(sbatch $JOB1 "$SETTINGSFILE" | awk '{print $4}')
    mkdir -p logs/datagen/"$JOB1_ID"
    echo "Submitted Datagen of $x with ID $JOB1_ID"
    while squeue -j "$JOB1_ID" | grep -q "$JOB1_ID"; do
      sleep 1
    done

    JOB2_ID=$(sbatch $JOB2 "$SETTINGSFILE" | awk '{print $4}')
    echo "Submitted Combiner of $x with ID $JOB2_ID"
    while squeue -j "$JOB2_ID" | grep -q "$JOB2_ID"; do
      sleep 1
    done

    JOB3_ID=$(sbatch $JOB3 "$SETTINGSFILE" | awk '{print $4}')
    echo "Submitted Estimator of $x with ID $JOB3_ID"
done

echo "All jobs submitted."