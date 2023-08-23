#!/bin/bash
# run with `nohup bash datagenerator.sh &`

set -e

JOB1="scripts/trainer.sbatch"

SETTINGS=("ul_nn_256.json" "ul_nn_512.json" "ul_nn_1024.json" "ul_nn_2048.json")
SETTINGS_DIR="settings/"

mkdir -p logs/training

for x in "${SETTINGS[@]}"
do
    SETTINGSFILE="$SETTINGS_DIR$x"
    
    # Submit the first job and capture the job ID
    JOB1_ID=$(sbatch $JOB1 $SETTINGSFILE | awk '{print $4}')
    # while squeue -j $JOB1_ID | grep -q $JOB1_ID; do
    #   sleep 1
    # done
done
