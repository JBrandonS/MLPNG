#!/bin/bash

SETTINGS=(
    "l500_1000.json"      
    # "l750_n256.json"      
    # "l1000_n512.json"      
    # "l2000_n2048.json"
)
ARGS=(
    "--nsims" "1000"
    "--narray" "100"
    "--disable_noise" 
)

for x in "${SETTINGS[@]}"; do
    sbatch sbatch/tuner.sbatch "${ARGS[@]}" "settings/$x"
done
