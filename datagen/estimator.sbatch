#! /bin/bash
#SBATCH --job-name=estimator
#SBATCH --partition=highmem,standard-l,standard-s
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

module purge
module load spack conda gcc-11.2.0/intel-oneapi-mkl gcc-11.2.0/openmpi nvidia/23.5 nvhpc/23.5 gcc-11.2.0/ucx

eval "$(conda shell.bash hook)"
conda activate mlpng

# time mpirun -n $SLURM_NTASKS python estimator.py
if [ -n "$1" ]
then
    time srun python datagen/estimator.py $1
else
    time srun python datagen/estimator.py
fi