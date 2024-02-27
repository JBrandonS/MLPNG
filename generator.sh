#!/bin/bash

#
# This script is used to submit a series of jobs to the slurm scheduler to automate the generation pipeline along the settings files
# This can take days to run, but it will run in the background and you can check the status of the jobs with squeue
#

# directory holding all the settings files
SETTINGS_DIR="settings/"

# list of the settings file to be used, will be ran in order
SETTINGS=(
  "l_128.json" 
  # "l_256.json"
  "l_512.json"
  # "l_1024.json"
  "l_2048.json"
)

# override sim settings. These settings will take priority, see config.py for the meaning of these settings, and others
ARGS=(
  "--nsims" "200"
  "--disable_lensing" 
  # --lensing
  "--disable_noise" 
  # --noise
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

for x in "${SETTINGS[@]}"
do
    SETTINGSFILE="$SETTINGS_DIR$x"

    # each block submits a slurm job based on the settings files
    # the awk command is used to extract the job id from the sbatch output
    # this uses slurms dependency system to ensure the jobs run in order and only after previous jobs have completed
    
    # generate the alms
    JOB0_ID=$(sbatch "${SLURM_ARGS[@]}" "sbatch/almgen.sbatch" "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    echo "Submitted Almgen of $x with ID $JOB0_ID"

    # Not used, but generates the patches
    # JOB1_ID=$(sbatch --dependency=afterok:$JOB0_ID "${SLURM_ARGS[@]}" "sbatch/patchgen.sbatch" "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    # echo "Submitted patchgen of $x with ID $JOB1_ID"

    # If we are generating patches, JOB1, then we need to use JOB1_ID, otherwise we use JOB0_ID
    PREV_JOB_ID="${JOB1_ID:-$JOB0_ID}"

    # combines the data into a single file, will do alms or alms and patches
    JOB2_ID=$(sbatch --dependency=afterok:"$PREV_JOB_ID" "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    echo "Submitted Combiner of $x with ID $JOB2_ID"

    # runs the estimator on the combined data
    JOB3_ID=$(sbatch --dependency=afterok:"$JOB2_ID" "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
    echo "Submitted Estimator of $x with ID $JOB3_ID"
done

# Uncomment to run the heidelberg estimator test
# JOBH_ID=$(sbatch "sbatch/heidelberg.sbatch" | awk '{print $4}')
# echo "Submitted Heidelberg Estimator with ID $JOBH_ID"

echo "All jobs submitted."