#!/usr/bin/env bash

#
# This script is used to submit a series of jobs to the slurm scheduler to automate the generation pipeline along the settings files
# This can take days to run, but it will run in the background and you can check the status of the jobs with squeue
#


# list of the settings file to be used, will be ran in order
# these must be in settings/ and have the .json extension
SETTINGS=(
  # "n32"
  "n64"
  # "n128"
  # "n256"
  # "n512"
  # "n1024"

  # these need to use high memory nodes, change sbatch/generator.sbatch to sbatch/generator-hm.sbatch in the main loop below 
  # "n2048"
  # "n4096"
  
  # "elsner"
  # "planck"
)

# override sim settings. These settings will take priority, see core.py for the meaning of these settings, and others
ARGS=(
  # "--nsims" "100"
  # "--lensing"
  # "--no-noise" 
  "--pols"
  "--fnl_range" "-100" "100"
  "--force_generation"
  # "--no-force_ksw"
  "--narray" "1"
)

# override some slurm settings, only used for narray
SLURM_ARGS=(
  "--array" "1-${ARGS[@]: -1}"
)

#### Uncomment to run the elsner estimator test
JOBH_ID=$(sbatch "sbatch/elsner.sbatch" "${ARGS[@]}" | awk '{print $4}')
echo "Submitted Elsner estimator test with ID $JOBH_ID"

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

    job_id=$(sbatch "${SLURM_ARGS[@]}" "sbatch/generator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    # combines the data into a single file
    job_id=$(sbatch --dependency=afterok:"$job_id" "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')
    # job_id=$(sbatch "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')

    # runs the estimator on the combined data
    sbatch --dependency=afterok:"$job_id" "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" > /dev/null 2>&1
    # sbatch "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" > /dev/null 2>&1
done