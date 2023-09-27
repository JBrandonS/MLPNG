#!/bin/bash

FULL_CLEAN=0

# Check for --full flag
if [[ "$1" == "--full" ]]; then
    echo "Running removing finalized data, are you sure?"
    echo "Press ctrl+c to cancel..."
    sleep 3
    FULL_CLEAN=1
fi

cd "data" || ( echo "cannot find data folder" && exit )

echo "Currently in directory: $(pwd)"

rm -rvf plots/*
rm -rvf models/*
rm -rvf tensorboard/*

directories=('alm_cache' 'lensed' 'unlensed')

for dir in "${directories[@]}"; do
    cd "$dir" 2>/dev/null || continue
    echo "Currently in directory: $(pwd)"

    rm -rvf *.nc

    # If the --full flag is passed, remove all data files in the directory
    if [[ "$FULL_CLEAN" -eq 1 ]]; then
        rm -rvf ./*
    fi

    cd ..
done

cd ..

rm -rvf logs/datagen/*
rm -rvf logs/*/*.log
rm -vf nohup.out

echo "Cleanup completed successfully."

