#! /bin/bash
#SBATCH --job-name=generator
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --output=data/logs/generator/%A/%j.log
#SBATCH --array=1-1
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

# create our logging dir, if needed
mkdir -p data/logs/generator/"$SLURM_ARRAY_JOB_ID"/

echo "Running generator on $(hostname) with cli $*"

module purge
module load conda gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza
module list

conda activate mlpng

time srun python -m mlpng.generator "$@"
# time srun python -m mlpng.scale_test "$@"
