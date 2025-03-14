#!/usr/bin/env bash
#SBATCH --job-name=jupyter
#SBATCH --output=.jupyter.out
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

conda_env="ds25"

print_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -e, --env <env>         Which conda env to use (default: mlpng)"
    echo "  -h, --help              Print this help message"
    echo ""
}

# Parse the CLI arguments
if ! valid_args=$(
    getopt \
        -o e:h \
        --long env,help \
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
    -e | --env)
        conda_env="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    esac
done

module purge
module load spack conda gcc/13.2.0 cuda/12.4.1-vz7djzz hpc-x/2.17.1 openblas/0.3.26-sauswx6 amdlibflame/4.2-5lac6gu nccl/2.21.5-1-nnpmvmq cudnn/8.9.7.29-12-lo4uzx3
module list

echo "Loading conda env: $conda_env"
conda activate "$conda_env"

nvidia-smi
nvcc --version

time srun jupyter notebook --ip=0.0.0.0
