#! /bin/bash
#SBATCH --job-name=patchgen
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/patchgen/%A/%j.log
#SBATCH --array=1-100
#SBATCH --time=02:00:00

# create the logging dir
mkdir -p logs/patchgen/"$SLURM_ARRAY_JOB_ID"/

module purge
module load conda
module load gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python -m scripts.patchgen "$@"

