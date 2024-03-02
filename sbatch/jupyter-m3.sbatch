#! /bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=.vscj.out
#SBATCH --partition=standard-s
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00


module purge
module load spack conda
module load gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

eval "$(conda shell.bash hook)"
conda activate mlpng

time srun jupyter notebook --ip=0.0.0.0
