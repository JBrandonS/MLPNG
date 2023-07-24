#!/bin/bash

cd "data"

echo "Currently in directory: $(pwd)"

rm -v alm_cache/*.nc
rm -rvf plots/*

directories=('lensed' 'unlensed')

for dir in "${directories[@]}"; do
    cd "$dir" 2>/dev/null || continue
    echo "Currently in directory: $(pwd)"

    rm -rvf *.nc
    rm -rvf models/*
    rm -rvf tensorboard/*

    cd ..
done

rm -rvf tmp/*

cd ..
rm -rvf logs/*

rm nohup.out

echo "Cleanup completed successfully."
