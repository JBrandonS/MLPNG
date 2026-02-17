#! /bin/bash
#SBATCH --job-name=ratio_ploter
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=data/logs/ratio_ploter/%A/%j.log
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

# create our logging dir, if needed
mkdir -p data/logs/ratio_ploter/"$SLURM_JOB_ID"/

echo "Running ratio_ploter on $(hostname) with cli $*"

module purge
module load conda gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza
module list

conda activate mlpng

time srun python -m mlpng.ratio_ploter "$@"
