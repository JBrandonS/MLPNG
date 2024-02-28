#!/bin/bash

# a fairly destructive cleaner

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

if [[ ! $(squeue --me -h -n jupyter) && -f .vscj.out ]]; then
    rm .vscj.out
fi

rm -rvf logs

cd "data" || ( echo "cannot find data folder" && exit )
echo "Currently in directory: $(pwd)"

rm -rvf plots
rm -rvf models
rm -rvf tensorboard
rm -rvf wandb

if [[ "$CLEAN_INCOMPLETE" -eq 1 ]] || [[ "$FULL_CLEAN" -eq 1 ]]; then
    directories=('alms' 'patches')
    for dir in "${directories[@]}"; do

        if [[ "$CLEAN_INCOMPLETE" -eq 1 ]] || [[ "$FULL_CLEAN" -eq 1 ]]; then
            rm -rvf "${dir:?}"/*.nc
        fi

        # If the --full flag is passed, remove all data files in the directory
        if [[ "$FULL_CLEAN" -eq 1 ]]; then
            rm -rvf "${dir:?}"
        fi
    done
fi

echo "Cleanup completed successfully."

