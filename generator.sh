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
  
  # these can only be run with nsims < 1000, due to time or memory
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
  "--nsims" "1000"
  "--pols" "T"
  "--shapes" "local"
  "--force_generation"
  "--phi_scale" "666"
  "--narray" "1"
)

# override the slurm array settings for narray, only used in data generation
ARR_ARGS=(
  "--array" "1-${ARGS[@]: -1}"
)

# general slurm args for all
SLURM_ARGS=(
  # "--partition" "dev"
  # "--time" "02:00:00"
  # "--ntasks" "1"
  # "--cpus-per-task" "10"
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

submit_job() {
  # define our submit function which will pull in the settings from global variables
  # this will check for a job_id and if found use that as a dependency to wait for the previous job to finish

  local job_id=$1
  local sbatch=$2
  local settings=$3
  local use_arr_args=${4:-"false"} # only want this for generator, as it uses the narray setting

  if [[ "$use_arr_args" == "true" ]]; then
    arr_args=("${ARR_ARGS[@]}")
  else
    arr_args=()
  fi

  if [ -z "$job_id" ]; then # no job_id found, just run the job
    job_id=$(sbatch "${arr_args[@]}" "${SLURM_ARGS[@]}" "$sbatch" "$settings" "${ARGS[@]}" | awk '{print $4}')
  else # job_id found, run the job with a dependency on the previous job
    job_id=$(sbatch "${arr_args[@]}" "${SLURM_ARGS[@]}" --dependency=afterok:"$job_id" "$sbatch" "$settings" "${ARGS[@]}" | awk '{print $4}')
  fi

  # acts as our return value
  echo "$job_id"
}

# loops over all settings files
# each block submits a slurm job based on the settings files
job_id=""
for x in "${SETTINGS[@]}"; do
  settings="settings/$x.json"

  # comment out if you want to run all settings files as dependent on the previous
  job_id=""

  # Submit the generator job
  job_id=$(submit_job "$job_id" "sbatch/generator.sbatch" "$settings" "true")

  # Combine the data into a single file
  job_id=$(submit_job "$job_id" "sbatch/combiner.sbatch" "$settings")

  # Run the estimator on the combined data
  # job_id=$(submit_job "$job_id" "sbatch/estimator.sbatch" "$settings")
done
