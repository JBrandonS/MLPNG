#! /bin/bash
#SBATCH --job-name=datagen
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/datagen/%A/%j.log
#SBATCH --error=logs/datagen/%A/%j.log
#SBATCH --array=1-1000
#SBATCH --time=04:00:00

module purge
module load spack conda gcc-11.2.0/intel-oneapi-mkl gcc-11.2.0/openmpi gcc-11.2.0/ucx

eval "$(conda shell.bash hook)"
conda activate mlpng

if [ -n "$1" ]
then
    time srun python scripts/datagen.py $1
else
    time srun python scripts/datagen.py
fi
