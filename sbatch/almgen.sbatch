#! /bin/bash
#SBATCH --job-name=almgen
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/almgen/%A/%j.log
#SBATCH --array=1-100
#SBATCH --time=24:00:00

# create our logging dir, if needed
mkdir -p logs/almgen/"$SLURM_ARRAY_JOB_ID"/

module purge
module load conda
module load gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python -m scripts.almgen "$@"

