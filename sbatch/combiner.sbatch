#! /bin/bash
#SBATCH --job-name=combiner
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=500G  # per node
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/combiner/%j.log
#SBATCH --error=logs/combiner/%j.log
#SBATCH --time=12:00:00

module purge
module load spack conda
module load gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python mlpng/combiner.py "$1"
