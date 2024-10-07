#! /bin/bash
#SBATCH --job-name=estimator
#SBATCH --partition=standard-s
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=data/logs/estimator/%j.log
#SBATCH --time=1-00:00:00

# create the logging dir
mkdir -p data/logs/estimator

module purge
module load conda 
module load gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python -m mlpng.estimator "$@"