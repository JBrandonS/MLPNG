#!/usr/bin/env bash

SETTINGS=(
    "l500_n128"
    "heidelberg"
    "planck"
)
ARGS=("--nsims" "100")
MODELS=("alm")

for x in "${SETTINGS[@]}"; do
    for model in "${MODELS[@]}"; do
        sbatch sbatch/tuner.sbatch "--model" "$model" "${ARGS[@]}" "settings/$x.json"
    done
done
