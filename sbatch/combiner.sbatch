#! /bin/bash
#SBATCH --job-name=combiner
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=1
#SBATCH --output=data/logs/combiner/%j.log
#SBATCH --time=8:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

# create the logging dir
mkdir -p data/logs/combiner

module purge
module load conda
module load gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python -m mlpng.combiner "$@"
