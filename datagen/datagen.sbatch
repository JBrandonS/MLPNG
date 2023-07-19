#! /bin/bash
#SBATCH --job-name=1-datagen
#SBATCH --partition=highmem,standard-l
#SBATCH --ntasks=1
#SBATCH --mem=1000G  # per node
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --array=1-1

module purge
module load spack conda gcc-11.2.0/intel-oneapi-mkl gcc-11.2.0/openmpi nvidia/23.5 nvhpc/23.5 gcc-11.2.0/ucx

eval "$(conda shell.bash hook)"
conda activate mlpng

if [ -n "$1" ]
then
    time srun python datagen/datagen.py $1
else
    time srun python datagen/datagen.py
fi
