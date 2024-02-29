#!/bin/bash

### Helper script to run the sbatch/jupyter.sbatch script
### and wait for the jupyter server to start, before printing the URL
### This script will reuse the running server if it exists.

print_help () {
    echo "Usage: start-jupyter.sh [options]"
    echo "Options:"
    echo "  -t, --time <time>       Time to run the job"
    echo "  -c, --cpus <cpus>       Number of CPUs to use"
    echo "  -g, --gpus <gpus>       Number of GPUs to use"
    echo "  -m, --mem <mem>         Memory to use"
    echo "  -h, --help              Print this help message"
    echo ""
    echo "All options are passed into sbatch and support the same format options as sbatch"
}

# Parse the CLI arguments
CLI_ARGS=()
VALID_ARGS=$(getopt -o t:c:g:m:h --long time:,cpus:,gpus:,mem:,help -- "$@")

# If arguments are invalid, print the help message and exit
# shellcheck disable=SC2181
if [[ $? -ne 0 ]]; then
    print_help
    exit 1;
fi

eval set -- "$VALID_ARGS"
# shellcheck disable=SC2078
while [ : ]; do
  case "$1" in
    -h | --help)
        print_help
        exit 0
        ;;
    -t | --time)
        CLI_ARGS+=("--time=$2")
        shift 2
        ;;
    -c | --cpus)
        CLI_ARGS+=("--cpus-per-task=$2")
        shift 2
        ;;
    -g | --gpus)
        CLI_ARGS+=("--gpus-per-task=$2")
        shift 2
        ;;
    -m | --mem)
        CLI_ARGS+=("--mem=$2")
        shift 2
        ;;
    --) 
        shift; 
        break 
        ;;
  esac
done

# Start the service
if [[ $(squeue --me -h -n jupyter) && -f .vscj.out ]]; then
    echo "Found Running server, reusing:"
    grep -m 1 -o 'http.*' .vscj.out
    exit 0
fi

# Remove the old log file if it exists
if [[ -f .vscj.out ]]; then
    rm .vscj.out
fi

# need to run diffrent sbatch script based on the hostname
if [[ $(hostname) == slogin* ]]; then
    RUN_SCRIPT="sbatch/jupyter-mp.sbatch"
else
    if [[ $(hostname) != m3login* ]]; then
        echo "Running in unknown login node, $(hostname), defaulting to m3"
    fi
    RUN_SCRIPT="sbatch/jupyter-m3.sbatch"
fi
sbatch -D "$PWD" "${CLI_ARGS[@]}" "$RUN_SCRIPT"

# Wait for the SLURM job to run
i=0
until [ -f .vscj.out ]
do
    i=$((i+1))
    echo -n "."
    # sleeps for 1 second for 10 seconds and then moves to 2... to keep from spamming
    sleep $((i/10 + 1))
done


# Wait for the jupyter server to start once the file is created and print the URL
line=""
while [[ -z "$line" ]]; do
    sleep 1
    line=$(grep -m 1 -o 'http.*' .vscj.out)
done

echo ""
echo "$line"
