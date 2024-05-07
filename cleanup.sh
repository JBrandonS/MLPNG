#!/bin/bash

# a fairly destructive cleaner

# remove the jupyter server file if it is not running exists
if [[ ! $(squeue --me -h -n jupyter) && -f .vscj.out ]]; then
    rm .vscj.out
fi

rm -rvf logs/almgen/*/*.log
rmdir logs/almgen/*/
rm -rvf logs/patchgen/*/*.log
rmdir logs/patchgen/*/
rm -rvf logs/combiner/*.log
rm -rvf logs/estimator/*.log
rm -rvf logs/heidelberg/*.log
rm -rvf logs/training/*.log
rm -rvf logs/tuning/*.log

# rm -rvf data/plots
rm -rvf data/models
rm -rvf data/tensorboard
rm -rvf data/wandb
rm -rvf data/tuning