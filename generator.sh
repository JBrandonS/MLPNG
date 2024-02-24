#!/bin/bash

#
# This script is used to submit a series of jobs to the slurm scheduler to automate the generation pipeline along the settings files
# This can take days to run, so it's best to run it in the background and log out of the ssh session
#
# run with `nohup bash generator.sh &` so you can close the terminal and log out of ssh without killing the process
#

# direcotry holding all the settings files
SETTINGS_DIR="settings/"

# list of the settings file to be used, will be ran in order
SETTINGS=(
  "l_128.json" 
  "l_256.json"
  "l_512.json"
  "l_1024.json"
  "l_2048.json"
)

# override the json / default settings with command line arguments
# see config.py for the meaning of these settings
ARGS=(
  "--nsims" "1000"
    # "--lensing"
    # "--disable_lensing"
    # "--noise"
    # "--disable_noise"
    # "--force_alm_gen"
    # "--fnl_range" "-100" "100"
    # "--narray" "10" # change alm and patch runners from `sbatch $JOB0` to `sbatch --array=1-100 $JOB0`
)

# override the slurm settings with command line arguments
# only used for alm / patch generation
SLURM_ARGS=(
  # "--array" "1-10"
)


echo "Submitting jobs with overrides:" "${ARGS[@]}" "\n" "Slurm overrides:" "${SLURM_ARGS[@]}"

# the sbatch files to be used
JOB0="sbatch/almgen.sbatch"
JOB1="sbatch/patchgen.sbatch"
JOB2="sbatch/combiner.sbatch"
JOB3="sbatch/estimator.sbatch"

# Uncomment to run the heidelberg estimator test
# JOBH_ID=$(sbatch "sbatch/heidelberg.sbatch" | awk '{print $4}')
# echo "Submitted Heidelberg Estimator with ID $JOBH_ID"
# # exit 0

# each block submits a slurm job based on the jobs files 
# and settings files, possibly setups a logging dir if needed
# then waits for the job to finish before moving on to the next one
# the awk command is used to extract the job id from the sbatch output
for x in "${SETTINGS[@]}"
do
    SETTINGSFILE="$SETTINGS_DIR$x"

    JOB0_ID=$(sbatch "${SLURM_ARGS[@]}" $JOB0 "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    mkdir -p logs/almgen/"$JOB1_ID"
    echo "Submitted Almgen of $x with ID $JOB0_ID"
    while squeue -j "$JOB0_ID" | grep -q "$JOB0_ID"; do
      sleep 1
    done

    # JOB1_ID=$(sbatch "${SLURM_ARGS[@]}" $JOB1 "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    # mkdir -p logs/patchgen/"$JOB1_ID"
    # echo "Submitted patchgen of $x with ID $JOB1_ID"
    # while squeue -j "$JOB1_ID" | grep -q "$JOB1_ID"; do
    #   sleep 1
    # done

    JOB2_ID=$(sbatch $JOB2 "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    echo "Submitted Combiner of $x with ID $JOB2_ID"
    while squeue -j "$JOB2_ID" | grep -q "$JOB2_ID"; do
      sleep 1
    done

    JOB3_ID=$(sbatch $JOB3 "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    echo "Submitted Estimator of $x with ID $JOB3_ID"
    # # you can be 'nice' and wait for the job to finish before moving on but it's not necessary
    # while squeue -j "$JOB3_ID" | grep -q "$JOB3_ID"; do
    #   sleep 1
    # done
done

echo "All jobs submitted."