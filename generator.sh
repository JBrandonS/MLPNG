#!/bin/bash
#
# This script is used to submit a series of jobs to the slurm scheduler to automate the datageneration along the settings files
# This can take days to run, so it's best to run it in the background and log out of the ssh session
#
# run with `nohup bash datagenerator.sh &` so you can close the terminal and log out of ssh without killing the process
#

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

# see config.py for the meaning of these settings
ARGS=("--nsmis 1000")

# the sbatch files to be used
JOB0="sbatch/almgen.sbatch"
JOB1="sbatch/datagen.sbatch"
# JOB1="sbatch/datagen-large.sbatch"
JOB2="sbatch/combiner.sbatch"
JOB3="sbatch/estimator.sbatch"

# Uncomment to run the heidelberg estimator test
# JOBH_ID=$(sbatch "sbatch/heidelberg.sbatch" | awk '{print $4}')
# echo "Submitted Heidelberg Estimator with ID $JOBH_ID"
# exit 0

for x in "${SETTINGS[@]}"
do
    SETTINGSFILE="$SETTINGS_DIR$x"

    # each block submits a slurm job based on the jobs files 
    # and settings files, possibly setups a logging dir if needed
    # then waits for the job to finish before moving on to the next one

    JOB0_ID=$(sbatch $JOB0 "$SETTINGSFILE" $ARGS | awk '{print $4}')
    mkdir -p logs/almgen/"$JOB1_ID"
    echo "Submitted Almgen of $x with ID $JOB0_ID"
    while squeue -j "$JOB0_ID" | grep -q "$JOB0_ID"; do
      sleep 1
    done

    # JOB1_ID=$(sbatch $JOB1 "$SETTINGSFILE" $ARGS | awk '{print $4}')
    # mkdir -p logs/datagen/"$JOB1_ID"
    # echo "Submitted Datagen of $x with ID $JOB1_ID"
    # while squeue -j "$JOB1_ID" | grep -q "$JOB1_ID"; do
    #   sleep 1
    # done

    JOB2_ID=$(sbatch $JOB2 "$SETTINGSFILE" $ARGS | awk '{print $4}')
    echo "Submitted Combiner of $x with ID $JOB2_ID"
    while squeue -j "$JOB2_ID" | grep -q "$JOB2_ID"; do
      sleep 1
    done

    # JOB3_ID=$(sbatch $JOB3 "$SETTINGSFILE" $ARGS | awk '{print $4}')
    # echo "Submitted Estimator of $x with ID $JOB3_ID"
    # # # you can be 'nice' and wait for the job to finish before moving on but it's not necessary
    # # while squeue -j "$JOB3_ID" | grep -q "$JOB3_ID"; do
    # #   sleep 1
    # # done
done

echo "All jobs submitted."