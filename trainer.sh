#!/bin/bash

JOB="sbatch/trainer.sbatch"
SETTINGS_DIR="settings/"
SETTINGS=(
    "l500_n128.json"      
    # "l750_n256.json"      
    # "l1000_n512.json"      
    # "l_1024.json"
)
ARGS=(
    "--nsims" "200"
    "--disable_lensing" 
    "--disable_noise" 
    "--narray" "500"
)

for x in "${SETTINGS[@]}"
do
    SETTINGSFILE="$SETTINGS_DIR$x"
    
    # Submit the first job and capture the job ID
    # JOB1_ID=$(sbatch $JOB "scripts/isensee_attn.py" "$SETTINGSFILE" | awk '{print $4}')
    # JOB2_ID=$(sbatch $JOB "scripts/dcnn.py" "$SETTINGSFILE" | awk '{print $4}')
    # JOB3_ID=$(sbatch $JOB "scripts/alm.py" "$SETTINGSFILE" | awk '{print $4}')
    JOB3_ID=$(sbatch $JOB "scripts/attn_alm.py" "${ARGS[@]}" "$SETTINGSFILE" | awk '{print $4}')
done
