#! /bin/bash
#SBATCH --job-name=combiner
#SBATCH --partition=highmem,standard-l,standard-s
#SBATCH --ntasks=1
#SBATCH --mem=500G  # per node
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl gcc-11.2.0/openmpi nvidia/23.5 nvhpc/23.5 gcc-11.2.0/ucx fftw

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python combiner.py
