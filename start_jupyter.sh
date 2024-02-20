#!/bin/bash

### Helper script to run the sbatch/jupyter.sbatch script
### and wait for the jupyter server to start, before printing the URL
### This script will reuse the running server if it exists.

# Print the CLI arguemnts
print_help () {
    echo "Usage: start-superpod-jupyter-server [options]"
    echo "Options:"
    echo "  -t, --time <time>       Time to run the job (default: 8:00:00)"
    echo "  -c, --cpus <cpus>       Number of CPUs to use (default: 16)"
    echo "  -g, --gpus <gpus>       Number of GPUs to use (default: 1)"
    echo "  -m, --mem <mem>         Memory to use (default: 64G)"
    echo "  -h, --help              Print this help message"
    exit 0
}

# Parse the CLI arguments
CLI_ARGS=""
VALID_ARGS=$(getopt -o t:c:g:m:h --long time:,cpus:,gpus:,mem:,help -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -h | --help)
        print_help
        exit
        ;;
    -t | --time)
        CLI_ARGS="$CLI_ARGS --time=$2"
        shift 2
        ;;
    -c | --cpus)
        CLI_ARGS="$CLI_ARGS --cpus-per-task=$2"
        shift 2
        ;;
    -g | --gpus)
        CLI_ARGS="$CLI_ARGS --gpus-per-task=$2"
        shift 2
        ;;
    -m | --mem)
        CLI_ARGS="$CLI_ARGS --mem=$2"
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
    echo "Found Running server, reusing"
else
    if [[ -f .vscj.out ]]; then
        rm .vscj.out
    fi

    # need to run diffrent sbatch script based on the hostname
    if [[ $(hostname) == slogin* ]]; then
        RUN_SCRIPT="sbatch/jupyter-mp.sbatch"
    else
        RUN_SCRIPT="sbatch/jupyter-m3.sbatch"
    fi
    sbatch -D $PWD $CLI_ARGS $RUN_SCRIPT
fi

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
echo $line
