#!/bin/bash

#
# This script is used to submit a series of jobs to the slurm scheduler to automate the generation pipeline along the settings files
# This can take days to run, but it will run in the background and you can check the status of the jobs with squeue
#
# Uncomment to run the heidelberg estimator test
# JOBH_ID=$(sbatch "sbatch/heidelberg.sbatch" | awk '{print $4}')
# echo "Submitted Heidelberg Estimator with ID $JOBH_ID"
#

# directory holding all the settings files
SETTINGS_DIR="settings/"

# list of the settings file to be used, will be ran in order
SETTINGS=(
  # "l500_n128.json" 
  # "l750_n256.json"
  # "l1000_n512.json"
  # "l2000_n1024.json"
  # "l2000_n2048.json"
  "planck.json"
)

# override sim settings. These settings will take priority, see config.py for the meaning of these settings, and others
ARGS=(
  # "--nsims" "200"
  "--no-lensing"
  "--no-noise" 
  # "--fnl_range" "-100" "100"
  "--narray" "500" # change slurm args array to match this
  # "--force_alm_gen"
  # "--base_dir" "data-shared"
)

# override some slurm settings, only used for narray
SLURM_ARGS=(
  "--array" "1-500"
)

echo "Submitting jobs with Settings overrides:" "${ARGS[@]}"
echo "                        Slurm overrides:" "${SLURM_ARGS[@]}"

# loops over all settings files
# each block submits a slurm job based on the settings files
# the awk command is used to extract the job id from the sbatch output
# this uses slurms dependency system to ensure the jobs run in order and only after previous jobs have completed
for x in "${SETTINGS[@]}"; do
    SETTINGS_FILE="$SETTINGS_DIR$x"
    
    # generate the alms
    JOB0_ID=$(sbatch "${SLURM_ARGS[@]}" "sbatch/almgen.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    ### Not used, but generates the patches
    # JOB1_ID=$(sbatch --dependency=afterok:$JOB0_ID "${SLURM_ARGS[@]}" "sbatch/patchgen.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    # If we are generating patches, JOB1, then we need to use JOB1_ID, otherwise we use JOB0_ID
    PREV_JOB_ID="${JOB1_ID:-$JOB0_ID}"

    # combines the data into a single file
    JOB2_ID=$(sbatch --dependency=afterok:"$PREV_JOB_ID" "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')
    # JOB2_ID=$(sbatch "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    # runs the estimator on the combined data
    sbatch --dependency=afterok:"$JOB2_ID" "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" > /dev/null 2>&1
    # sbatch "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" > /dev/null 2>&1
done

echo "All jobs submitted."