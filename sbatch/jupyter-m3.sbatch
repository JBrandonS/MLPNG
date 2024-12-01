#!/usr/bin/env bash
#SBATCH --job-name=jupyter
#SBATCH --output=.jupyter.out
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

conda_env="mlpng"

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
module load conda gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate "$conda_env"

echo "Starting Jupyter Notebook..."
echo "  - Conda env: $conda_env"
echo "  - $(which python)"
echo "  - $(which jupyter)"
echo "  - Module list:"
module list
echo ""

time srun jupyter notebook --ip=0.0.0.0
