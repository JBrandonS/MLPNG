#!/bin/bash

FULL_CLEAN=0
CLEAN_INCOMPLETE=0

# Check for --full flag
if [[ "$1" == "--full" ]]; then
    echo "Running removing finalized data, are you sure?"
    echo "Press ctrl+c to cancel..."
    sleep 3
    FULL_CLEAN=1
fi

if [[ "$1" == "--clean-incomplete" ]]; then
    echo "Removing incomplete data files..."
    CLEAN_INCOMPLETE=1
fi

if [[ ! $(squeue --me -h -n VSC-jupyter) && -f .VSC-jupyter.out ]]; then
    rm .VSC-jupyter.out
fi

rm -rvf logs/datagen/*
rm -rvf logs/*/*.log
rm -vf nohup.out 
rm -vf .vscj.out

cd "data" || ( echo "cannot find data folder" && exit )

echo "Currently in directory: $(pwd)"

rm -rvf plots/*
rm -rvf models/*
rm -rvf tb/*
rm -rvf tensorboard/*
rm -rvf wandb/*

if [[ "$CLEAN_INCOMPLETE" -eq 1 ]] || [[ "$FULL_CLEAN" -eq 1 ]]; then
    directories=('alm_cache' 'lensed' 'unlensed')
    for dir in "${directories[@]}"; do
        cd "$dir" 2>/dev/null || continue
        echo "Currently in directory: $(pwd)"

        if [[ "$CLEAN_INCOMPLETE" -eq 1 ]] || [[ "$FULL_CLEAN" -eq 1 ]]; then
            rm -rvf ./*.nc
        fi

        # If the --full flag is passed, remove all data files in the directory
        if [[ "$FULL_CLEAN" -eq 1 ]]; then
            rm -rvf ./*
        fi

        cd ..
    done
fi

echo "Cleanup completed successfully."

