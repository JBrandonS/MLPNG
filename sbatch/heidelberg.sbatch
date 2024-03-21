#! /bin/bash
#SBATCH --job-name=heidelberg
#SBATCH --partition=standard-s
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/heidelberg/%j.log
#SBATCH --time=24:00:00

# create the logging dir
mkdir -p logs/heidelberg

module purge
module load spack conda 
module load gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python scripts/heidelberg.py