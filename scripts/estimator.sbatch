#! /bin/bash
#SBATCH --job-name=estimator
#SBATCH --partition=highmem
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/estimator/%j.log
#SBATCH --error=logs/estimator/%j.log
#SBATCH --time=48:00:00

module purge
module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy openmpi/4.1.4-7al4h2x fftw/3.3.10-gz7qiki

eval "$(conda shell.bash hook)"
conda activate mlpng

if [ -n "$1" ] 
then
    time srun python scripts/estimator.py "$1"
else
    time srun python scripts/estimator.py
fi