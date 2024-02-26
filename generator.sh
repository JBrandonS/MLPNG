#!/bin/bash

#
# This script is used to submit a series of jobs to the slurm scheduler to automate the generation pipeline along the settings files
# This can take days to run, so it's best to run it in the background and log out of the ssh session
#

# direcotry holding all the settings files
SETTINGS_DIR="settings/"

# list of the settings file to be used, will be ran in order
SETTINGS=(
  "l_128.json" 
  # "l_256.json"
  "l_512.json"
  # "l_1024.json"
  "l_2048.json"
)

# override sim settings, see config.py for the meaning of these settings, and others
ARGS=(
  "--nsims" "200"
  "--disable_lensing" # --lensing
  "--disable_noise" # --noise
  # "--fnl_range" "-100" "100"
  "--narray" "500" # change slurm args array to match this
  # "--force_alm_gen"
  # "--base_dir" "data-shared"
)

# override some slurm settings, only used for narray in this case
SLURM_ARGS=(
  "--array" "1-500"
)

echo "Running script $$"
echo "Submitting jobs with Settings overrides:" "${ARGS[@]}"
echo "                        Slurm overrides:" "${SLURM_ARGS[@]}"

# ensure some directories exist
mkdir -p data/alms #data/patches 
mkdir -p data/logs/combiner logs/estimator

for x in "${SETTINGS[@]}"
do
    SETTINGSFILE="$SETTINGS_DIR$x"

    # each block submits a slurm job based on the jobs files 
    # and settings files, possibly setups a logging dir if needed
    # then waits for the job to finish before moving on to the next one
    # the awk command is used to extract the job id from the sbatch output
    
    JOB0_ID=$(sbatch "${SLURM_ARGS[@]}" "sbatch/almgen.sbatch" "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    mkdir -p data/logs/almgen/"$JOB0_ID"
    echo "Submitted Almgen of $x with ID $JOB0_ID"

    # JOB1_ID=$(sbatch --dependency=afterok:$JOB0_ID "${SLURM_ARGS[@]}" "sbatch/patchgen.sbatch" "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    # mkdir -p data/logs/patchgen/"$JOB1_ID"
    # echo "Submitted patchgen of $x with ID $JOB1_ID"

    # If JOB1 exists, JOB2 depends on JOB1. Otherwise, it depends on JOB0
    PREV_JOB_ID="${JOB1_ID:-$JOB0_ID}"

    JOB2_ID=$(sbatch --dependency=afterok:"$PREV_JOB_ID" "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    echo "Submitted Combiner of $x with ID $JOB2_ID"

    JOB3_ID=$(sbatch --dependency=afterok:"$JOB2_ID" "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    echo "Submitted Estimator of $x with ID $JOB3_ID"
done

# Uncomment to run the heidelberg estimator test
# JOBH_ID=$(sbatch "sbatch/heidelberg.sbatch" | awk '{print $4}')
# echo "Submitted Heidelberg Estimator with ID $JOBH_ID"
# # exit 0

echo "All jobs submitted."