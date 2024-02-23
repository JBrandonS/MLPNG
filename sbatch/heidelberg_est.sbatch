#! /bin/bash
#SBATCH --job-name=hei-est-test
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=40G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/estimator/%j.log
#SBATCH --time=2:00:00

module purge
module load spack conda 
module load gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python scripts/heidelberg-estimator.py