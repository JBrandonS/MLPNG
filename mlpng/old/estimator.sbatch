#! /bin/bash
#SBATCH --job-name=estimator
#SBATCH --partition=standard-s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=data/logs/estimator/%j.log
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

# create the logging dir
mkdir -p data/logs/estimator

echo "Running estimator on $(hostname) with cli $*"

module purge
module load conda
module load gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza
module list
eval "$(conda shell.bash hook)"
conda activate mlpng

# mkdir -p data/profiles/estimator
# time srun python -m cProfile -o data/profiles/estimator/$SLURM_JOB_ID.prof -m mlpng.estimator "$@"
time srun --mpi=pmi2 python -m mlpng.estimator "$@"
