#!/usr/bin/env bash

# This script will generate a set of training jobs based off the SETTINGS and MODELS arrays.

# The settings files to use for the sims. These should be found in settings/*.json
SETTINGS=(  
    "l500_n128"
    "heidelberg"
    "planck"
)

# CLI args to change the command ran, helps prevent needing to change the settings file for small / test changes
# make sure these align with the data generation or you will get errors
# see scripts/core.parse_args for more info and available arguments, or to add more
ARGS=("--nsims" "100")

# The AI models to train on the data, see scripts/trainer.py and scripts/models/ for more info
# these models should be registered with @register_model
MODELS=(
    "alm"
    "dcnn"
    "isensee"
    "resnet"
    "nagarajappa"
)

# loops through all the SETTINGS files and for each submits a slurm job for each MODEL
for x in "${SETTINGS[@]}"; do
    file="settings/$x.json"
    
    for model in "${MODELS[@]}"; do
        JOB_ID=$(sbatch "sbatch/trainer.sbatch" "--model" "$model" "${ARGS[@]}" "$file" | awk '{print $4}')
        echo "Submitted $model, $x, with ID $JOB_ID"
    done
done
