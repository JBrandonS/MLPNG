#! /bin/bash
#SBATCH --job-name=datagen
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/datagen/%A/%j.log
#SBATCH --error=logs/datagen/%A/%j.log
#SBATCH --array=1-10
#SBATCH --time=01:30:00

module purge
module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy openmpi/4.1.4-7al4h2x fftw/3.3.10-gz7qiki

eval "$(conda shell.bash hook)"
conda activate mlpng

if [ -n "$1" ]
then
    time srun python scripts/datagen.py "$1"
else
    time srun python scripts/datagen.py
fi
