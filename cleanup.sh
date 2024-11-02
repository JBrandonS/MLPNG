#!/usr/bin/env bash

remove_data=false
remove_plots=false
remove_tf=false
remove_models=false
remove_ksw=false

print_help () {
    echo "Usage: $0 [options]"
    echo ""
    echo "A possibly destructive cleaner which will remove the logs and optionally data files."
    echo ""
    echo "Options:"
    echo "  -m, --models    Remove the model data"
    echo "  -d, --data      Remove the data files"
    echo "  -k, --ksw       Remove the ksw mc files"
    echo "  -p, --plots     Remove all plots"
    echo "  -t, --tf        Remove all misc tensorflow logs, such as tuning, tensorboard, and wandb logs"
    echo "  --all           Remove everything"
    echo "  -h, --help      Print this help message"
    echo ""
}


# Parse the CLI arguments
if ! valid_args=$(getopt \
    -o mdptkh \
    --long models,data,plots,ksw,tf,all,help \
    -- "$@"
); then
    # invalid arguments found, print usage and exit
    print_help
    exit 1
fi

# sets up the positional arguments from valid_args for use in the while loop
eval set -- "$valid_args"

# Here we actually process the CLI arguments
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help)
        print_help
        exit 0
        ;;
    -m | --models)
        remove_models=true
        shift
        ;;
    -d | --data)
        remove_data=true
        shift
        ;;
    -k | --ksw)
        remove_ksw=true
        shift
        ;;
    -p | --plots)
        remove_plots=true
        shift
        ;;
    -t | --tf)
        remove_tf=true
        shift
        ;;
    --all)
        remove_data=true
        remove_ksw=true
        remove_models=true
        remove_plots=true
        remove_tf=true
        shift
        ;;
    --)
        shift
        break
        ;;
  esac
done

# remove the jupyter server file if it is not running and exists
if [[ ! $(squeue --me -h -n jupyter) && -f .jupyter.out ]]; then
    rm .jupyter.out
fi

if [[ ! $(squeue --me -h -n jupyter-dev) && -f .jupyter-dev.out ]]; then
    rm .jupyter-dev.out
fi

# logs
rm -rvf data/logs/*

if [[ "$remove_plots" == true ]]; then
    rm -rvf data/plots/*
fi

if [[ "$remove_data" == true ]]; then
    rm -rvf data/data/*
fi

if [[ "$remove_ksw" == true ]]; then
    rm -rvf data/kswmc/*
fi

if [[ "$remove_models" == true ]]; then
    rm -rvf data/models/*
fi

if [[ "$remove_tf" == true ]]; then
    rm -rvf data/tensorboard
    rm -rvf data/wandb
    rm -rvf data/tuning
fi