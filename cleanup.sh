#!/usr/bin/env bash

# a fairly destructive cleaner which will remove the logs and some data files. It does not remove any sim data.

# remove the jupyter server file if it is not running exists
if [[ ! $(squeue --me -h -n jupyter) && -f .vscj.out ]]; then
    rm .vscj.out
fi

# logs
rm -rvf logs/combiner/*.log
rm -rvf logs/estimator/*.log
rm -rvf logs/heidelberg/*.log
rm -rvf logs/training/*.log
rm -rvf logs/tuning/*.log
rm -rvf logs/generator/*.log
rm -rvf logs/generator/*/

# some data
rm -rvf data/models
rm -rvf data/tensorboard
rm -rvf data/wandb
rm -rvf data/tuning

# jupyter logs
rm -rvf .jupyter*.out

# important data
# rm -rvf data/data
# rm -rvf data/kswmc
# rm -rvf data/plots