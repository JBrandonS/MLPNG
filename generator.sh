#!/bin/bash

#
# This script is used to submit a series of jobs to the slurm scheduler to automate the generation pipeline along the settings files
# This can take days to run, but it will run in the background and you can check the status of the jobs with squeue
#

#### Uncomment to run the heidelberg estimator test
# JOBH_ID=$(sbatch "sbatch/heidelberg.sbatch" | awk '{print $4}')
# echo "Submitted Heidelberg Estimator with ID $JOBH_ID"
#

# list of the settings file to be used, will be ran in order
SETTINGS=(
  "planck"
)

# override sim settings. These settings will take priority, see core.py for the meaning of these settings, and others
ARGS=(
  # "--nsims" "200"
  # "--lensing"
  # "--noise" 
  # "--narray" "50" # change slurm args array to match this
  # "--base_name" "planck_fnl50"
  # "--fnl_range" "-50" "50"
)

# override some slurm settings, only used for narray
SLURM_ARGS=(
  # "--array" "1-50"
)

# Just log the overrides to the console
if [ "${#ARGS[@]}" -ne 0 ]; then
  echo "Submitting jobs with Settings overrides:" "${ARGS[@]}"
  if [ "${#SLURM_ARGS[@]}" -ne 0 ]; then
    echo "                        Slurm overrides:" "${SLURM_ARGS[@]}"
  fi
elif [ "${#SLURM_ARGS[@]}" -ne 0 ]; then
  echo "Submitting jobs with Slurm overrides:" "${SLURM_ARGS[@]}"
fi

for x in "${SETTINGS[@]}"; do
    # loops over all settings files
    # each block submits a slurm job based on the settings files
    # the awk command is used to extract the job id from the sbatch output
    # this uses slurms dependency system to ensure the jobs run in order and only after previous jobs have completed
    # comment and uncomment the JOBX_ID lines to disable the dependency system
    SETTINGS_FILE="settings/$x.json"
    
    # generate the alms
    JOB0_ID=$(sbatch "${SLURM_ARGS[@]}" "sbatch/almgen.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    ### generates the patches
    JOB1_ID=$(sbatch --dependency=afterok:"$JOB0_ID" "${SLURM_ARGS[@]}" "sbatch/patchgen.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')
    # JOB1_ID=$(sbatch "${SLURM_ARGS[@]}" "sbatch/patchgen.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    # If we are generating patches wait for JOB1_ID, otherwise we use JOB0_ID
    PREV_JOB_ID="${JOB1_ID:-$JOB0_ID}"

    # combines the data into a single file
    JOB2_ID=$(sbatch --dependency=afterok:"$PREV_JOB_ID" "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')
    # JOB2_ID=$(sbatch "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    # runs the estimator on the combined data
    sbatch --dependency=afterok:"$JOB2_ID" "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" > /dev/null 2>&1
    # sbatch "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" > /dev/null 2>&1
done
echo "All jobs submitted."