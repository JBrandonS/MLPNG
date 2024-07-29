#! /bin/bash
#SBATCH --job-name=hm-generator
#SBATCH --partition=highmem
#SBATCH --ntasks=1
#SBATCH --mem=1500G
#SBATCH --cpus-per-task=100
#SBATCH --output=logs/generator/%A/%j.log
#SBATCH --array=1-100
#SBATCH --time=24:00:00

# create our logging dir, if needed
mkdir -p logs/generator/"$SLURM_ARRAY_JOB_ID"/

module purge
module load conda
module load gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python -m scripts.generator "$@"

