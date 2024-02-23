#! /bin/bash
#SBATCH --job-name=datagen
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/datagen/%A/%j.log
#SBATCH --array=1-10
#SBATCH --time=12:00:00

module purge
module load spack conda
module load gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python scripts/datagen.py "$@"

