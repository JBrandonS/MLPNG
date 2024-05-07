#!/usr/bin/env bash

### Helper script to run the jupyter.sbatch script
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
    echo "Found running jupyter server, reusing"
else
    # Remove the old log file if it exists
    if [[ -f .vscj.out ]]; then
        rm .vscj.out
    fi

    # need to run diffrent sbatch script based on the hostname
    if [[ $(hostname) == slogin* ]]; then
        RUN_SCRIPT="sbatch/jupyter-mp.sbatch"
    else
        RUN_SCRIPT="sbatch/jupyter-m3.sbatch"
    fi

    # kick off the slurm job
    id=$(sbatch -D "$PWD" "${CLI_ARGS[@]}" "$RUN_SCRIPT" | awk '{print $4}')
    echo "Submitted jupyter job $id"

    # setup spinner and remove cursor
    # add trap to ensure the cursor is added back
    spinner="/|\\-/|\\-"
    tput civis
    trap "tput cnorm" EXIT

    # Wait for the SLURM job to run
    i=0
    until [ -f .vscj.out ]
    do
        printf "\r%s" "${spinner:((i++ % 8)):1}"
        sleep 1
    done
    printf "\rJupyter server started!\n"
fi

# once the file is create, get and print the URL
url_command="grep -m 1 -o 'http.*' .vscj.out"
line=$(eval "$url_command")
if [[ -z "$line" ]]; then
    printf "Waiting for server to start..."

    while [[ -z "$line" ]]; do
        sleep 1
        line=$(eval "$url_command")
    done
    printf "\r" # remove the waiting line
fi
echo "Server URL: $line"