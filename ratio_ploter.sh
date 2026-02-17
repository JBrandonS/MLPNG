#!/usr/bin/env bash

#
# This script submits a ratio_ploter job to compute R_GL = (S/N)_GL / (S/N)_0 vs phi_scale.
# It loops over settings files and submits a single Slurm job per settings file.
#

# list of the settings files to be used
# these must be in settings/ and have the .json extension
SETTINGS=(
  "n256"
  # "n128"
  # "n512"
)

# override sim settings passed through to python -m mlpng.ratio_ploter
ARGS=(
  "--pols" "T"
  "--shapes" "local"
  "--force_generation"
  "--lensing"
)

# general slurm args for all
SLURM_ARGS=(
  # "--partition" "dev"
  # "--time" "02:00:00"
  # "--cpus-per-task" "16"
)

# Just log the overrides to the console
if [ "${#ARGS[@]}" -ne 0 ]; then
  echo "Submitting ratio_ploter jobs with Settings overrides:" "${ARGS[@]}"
  if [ "${#SLURM_ARGS[@]}" -ne 0 ]; then
    echo "                                  Slurm overrides:" "${SLURM_ARGS[@]}"
  fi
elif [ "${#SLURM_ARGS[@]}" -ne 0 ]; then
  echo "Submitting ratio_ploter jobs with Slurm overrides:" "${SLURM_ARGS[@]}"
fi

submit_job() {
  local job_id=$1
  local sbatch=$2
  local settings=$3

  if [ -z "$job_id" ]; then
    job_id=$(sbatch "${SLURM_ARGS[@]}" "$sbatch" "$settings" "${ARGS[@]}" | awk '{print $4}')
  else
    job_id=$(sbatch "${SLURM_ARGS[@]}" --dependency=afterok:"$job_id" "$sbatch" "$settings" "${ARGS[@]}" | awk '{print $4}')
  fi

  echo "$job_id"
}

# loops over all settings files
job_id=""
for x in "${SETTINGS[@]}"; do
  settings="settings/$x.json"

  # reset dependency chain per settings file
  job_id=""

  job_id=$(submit_job "$job_id" "sbatch/ratio_ploter.sbatch" "$settings")
  echo "Submitted ratio_ploter for $x: job $job_id"
done
