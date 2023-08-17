#! /bin/bash
#SBATCH --job-name=estimator
#SBATCH --partition=standard-s
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=40G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/estimator/%j.log
#SBATCH --error=logs/estimator/%j.log

module purge
module load spack conda gcc-11.2.0/intel-oneapi-mkl gcc-11.2.0/openmpi gcc-11.2.0/ucx

eval "$(conda shell.bash hook)"
conda activate mlpng

if [ -n "$1" ]
then
    time srun python datagen/estimator.py $1
else
    time srun python datagen/estimator.py
fi