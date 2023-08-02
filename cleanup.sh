#!/bin/bash

cd "data"

echo "Currently in directory: $(pwd)"

rm -rvf plots/*
rm -rvf models/*
rm -rvf tensorboard/*

directories=('alm_cache' 'lensed' 'unlensed')

for dir in "${directories[@]}"; do
    cd "$dir" 2>/dev/null || continue
    echo "Currently in directory: $(pwd)"

    rm -rvf *.nc

    cd ..
done

cd ..
echo "Currently in directory: $(pwd)"

rm -rvf logs/*
rm -vf nohup.out

echo "Cleanup completed successfully."
