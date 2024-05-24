#!/usr/bin/env bash

#
# This script is used to submit a series of jobs to the slurm scheduler to automate the generation pipeline along the settings files
# This can take days to run, but it will run in the background and you can check the status of the jobs with squeue
#

#### Uncomment to run the heidelberg estimator test
# JOBH_ID=$(sbatch "sbatch/heidelberg.sbatch" | awk '{print $4}')
# echo "Submitted Heidelberg Estimator with ID $JOBH_ID"


# list of the settings file to be used, will be ran in order
# these must be in settings/ and have the .json extension
SETTINGS=(
  "l256_n64"
  # "l500_n128"
  "heidelberg"
  # "planck"
  # "l2000_n2048"
)

# override sim settings. These settings will take priority, see core.py for the meaning of these settings, and others
ARGS=(
  "--nsims" "100"
  # "--lensing"
  # "--noise" 
  "--narray" "10" # change slurm args array to match this
  # "--base_name" "planck_fnl50"
  # "--fnl_range" "-50" "50"
  "--force_generation"
  "--polarizations" "TE"
)

# override some slurm settings, only used for narray
SLURM_ARGS=(
  "--array" "1-10"
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

# loops over all settings files
# each block submits a slurm job based on the settings files
for x in "${SETTINGS[@]}"; do
    SETTINGS_FILE="settings/$x.json"

    # generate the alms
    # job_id=$(sbatch "${SLURM_ARGS[@]}" "sbatch/almgen.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    ### generates the patches
    # job_id=$(sbatch --dependency=afterok:"$job_id" "${SLURM_ARGS[@]}" "sbatch/patchgen.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')
    job_id=$(sbatch "${SLURM_ARGS[@]}" "sbatch/patchgen.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    # combines the data into a single file
    job_id=$(sbatch --dependency=afterok:"$job_id" "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')
    # job_id=$(sbatch "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    # runs the estimator on the combined data
    sbatch --dependency=afterok:"$job_id" "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" > /dev/null 2>&1
    # sbatch "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" > /dev/null 2>&1
done