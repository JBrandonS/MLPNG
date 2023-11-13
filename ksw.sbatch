#! /bin/bash
#SBATCH --job-name=ksw-lowmem
#SBATCH --partition=highmem
#SBATCH --partition=standard-s
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=40G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ksw-%j.log
#SBATCH --error=logs/ksw-%j.log
#SBATCH --time=24:00:00

module purge
module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy openmpi/4.1.4-7al4h2x fftw/3.3.10-gz7qiki

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun python ksw_hes.py