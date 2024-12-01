#!/usr/bin/env bash

# This script will generate a set of training jobs based off the SETTINGS and MODELS arrays.

# The settings files to use for the sims. These should be found in settings/*.json
SETTINGS=(
    # "n64"
    "n128"
    # "elsner"
    # "planck"
    # "n2048"
)

# CLI args to change the command ran, helps prevent needing to change the settings file for small / test changes
# make sure these align with the data generation or you will get errors
# see scripts/core.parse_args for more info and available arguments, or to add more
# some additional args are allowed for the trainer see scripts/models/modelcore.py
ARGS=(
    # "--nsims" "100"
    # "--lensing"
    # "--no-noise"
    # "--pols"
    # "--no-summary"
    "--tb"
    # "--wandb"
    "--narray" "1000"
)

# The AI models to train on the data, see scripts/trainer.py and scripts/models/ for more info
# these models should be registered with @register_model
MODELS=(
    "alm"
    # "isensee"
    "isensee_v2"
    # "dcnn"
    # "resnet"
    # "nagarajappa"
)

# loops through all the SETTINGS files and for each submits a slurm job for each MODEL
for x in "${SETTINGS[@]}"; do
    file="settings/$x.json"
    echo "Settings $x:"
    for model in "${MODELS[@]}"; do
        JOB_ID=$(sbatch "sbatch/trainer.sbatch" "--model" "$model" "${ARGS[@]}" "$file" | awk '{print $4}')
        printf "\tSubmitted %s, with ID %s\n" "$model" "$JOB_ID"
    done
done
