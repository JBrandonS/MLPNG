#!/bin/bash

set -e

JOB1="sbatch/trainer.sbatch"

SETTINGS=(
  "ul_nn_128.json" #"ul_nn_256.json"  "ul_nn_512.json"  "ul_nn_1024.json"
  # "l_nn_128.json"   "l_nn_256.json"   "l_nn_512.json"   "l_nn_1024.json"
  # "l_128.json"      "l_256.json"      "l_512.json"      "l_1024.json"
)

SETTINGS_DIR="settings/"

mkdir -p logs/training

for x in "${SETTINGS[@]}"
do
    SETTINGSFILE="$SETTINGS_DIR$x"
    
    # Submit the first job and capture the job ID
    # JOB1_ID=$(sbatch $JOB1 "$SETTINGSFILE" | awk '{print $4}')
    JOB2_ID=$(sbatch $JOB1 "scripts/dcnn.py" "$SETTINGSFILE" | awk '{print $4}')
    JOB3_ID=$(sbatch $JOB1 "scripts/alm.py" "$SETTINGSFILE" | awk '{print $4}')
done
