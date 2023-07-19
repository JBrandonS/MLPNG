#!/bin/bash

cd "data"

echo "Currently in directory: $(pwd)"

rm -iv alm_cache/*.nc 2>/dev/null

directories=('lensed' 'unlensed')

for dir in "${directories[@]}"; do
    cd "$dir" 2>/dev/null || continue
    echo "Currently in directory: $(pwd)"

    rm -rivf *.nc
    rm -rivf models/*
    rm -rivf plots/*
    rm -rivf tensorboard/*

    cd ..
done

rm -rivf tmp/* 2>/dev/null

cd ..
rm -rivf logs/* 2>/dev/null

rm nohup.out 2>/dev/null

echo "Cleanup completed successfully."
