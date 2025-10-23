#! /bin/bash
#SBATCH --job-name=initializor
#SBATCH --partition=standard-s
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --output=data/logs/initializor/%j.log
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

# create the logging dir
mkdir -p data/logs/initializor

echo "Running initializor on $(hostname) with cli $*"

module purge
module load conda
module load gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza
module list
eval "$(conda shell.bash hook)"
conda activate mlpng

export TF_CPP_MIN_LOG_LEVEL=3

time srun --mpi=pmi2 python -m mlpng.initializor "$@"
