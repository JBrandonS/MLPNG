#! /bin/bash
#SBATCH --job-name=datagen
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
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
