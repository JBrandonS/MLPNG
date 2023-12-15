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
# We dont need all of these for this script, but makes sure everything is built on the same setup
module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy openmpi/4.1.4-7al4h2x fftw/3.3.10-gz7qiki

eval "$(conda shell.bash hook)"
conda activate mlpng

if [ -n "$1" ]
then
    time srun python scripts/combiner.py "$1"
else
    time srun python scripts/combiner.py
fi
