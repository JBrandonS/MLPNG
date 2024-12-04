#! /bin/bash
#SBATCH --job-name=generator
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=64
#SBATCH --output=data/logs/generator/%A/%j.log
#SBATCH --array=1-100
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

# create our logging dir, if needed
mkdir -p data/logs/generator/"$SLURM_ARRAY_JOB_ID"/

module purge
module load conda
module load gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

# mkdir -p data/profiles/generator
# time srun python -m cProfile -o data/profiles/generator/$SLURM_JOB_ID.prof -m mlpng.generator "$@"
time srun python -m mlpng.generator "$@"
