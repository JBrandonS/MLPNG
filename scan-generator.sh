#!/bin/bash

SETTINGS_DIR="settings/"
SETTINGS=(
  "planck.json"
)
ARGS_BASE=(
  "--lensing"
  "--noise" 
)

for i in {1..10}; do
  for x in "${SETTINGS[@]}"; do
      SETTINGS_FILE="$SETTINGS_DIR$x"

      nval=$((500 / i))
      bval=$((10 / i))
      ARGS=("${ARGS_BASE[@]}" "--noise_scale_tt" "$nval" "--beam_width" "$bval" "--base_name" "hei_planck_test_$i")

      sbatch "sbatch/heidelberg.sbatch" "${ARGS[@]}" > /dev/null 2>&1
      # JOB1_ID=$(sbatch "${SLURM_ARGS[@]}" "sbatch/almgen.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')
      # JOB2_ID=$(sbatch --dependency=afterok:"$JOB1_ID" "sbatch/combiner.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" | awk '{print $4}')
      # sbatch --dependency=afterok:"$JOB2_ID" "sbatch/estimator.sbatch" "${ARGS[@]}" "$SETTINGS_FILE" > /dev/null 2>&1
  done
done
echo "All jobs submitted."