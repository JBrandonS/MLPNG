#!/usr/bin/env bash

#
# This script submits phi_scale_analysis jobs to train ML models at each phi_scale
# and compare R_GL^ML with analytical R_GL from ratio_ploter.
# Runs on the superpod with 4 GPUs per job.
#

# Settings files to use (must be in settings/ with .json extension)
SETTINGS=(
  "n64"
  # "n256"
  # "n128"
  # "n512"
)

# Override args passed through to python -m mlpng.phi_scale_analysis
ARGS=(
  "--nsims" "10000"
  "--pols" "T"
  "--shapes" "local"
  "--lensing"
)

# Slurm overrides (uncomment as needed)
SLURM_ARGS=(
  # "--partition" "dev"
  # "--time" "02:00:00"
)

# Log the overrides
if [ "${#ARGS[@]}" -ne 0 ]; then
  echo "Submitting phi_scale_analysis jobs with Settings overrides:" "${ARGS[@]}"
  if [ "${#SLURM_ARGS[@]}" -ne 0 ]; then
    echo "                                     Slurm overrides:" "${SLURM_ARGS[@]}"
  fi
elif [ "${#SLURM_ARGS[@]}" -ne 0 ]; then
  echo "Submitting phi_scale_analysis jobs with Slurm overrides:" "${SLURM_ARGS[@]}"
fi

for x in "${SETTINGS[@]}"; do
  settings="settings/$x.json"

  if [ ! -f "$settings" ]; then
    echo "Settings file $settings not found, skipping"
    continue
  fi

  JOB_ID=$(sbatch "${SLURM_ARGS[@]}" sbatch/phi_scale.sbatch "$settings" "${ARGS[@]}" | awk '{print $4}')
  echo "Submitted phi_scale_analysis for $x: job $JOB_ID"
done
