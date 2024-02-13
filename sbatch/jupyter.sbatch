#! /bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=.vscj.out
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=500G  # per node
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00

module purge
module load spack conda

# m3
module load gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza

# superpod
# module load gcc/11.2.0 cuda/11.8.0-vbvgppx cudnn/8.7.0.84-11.8-fhn3dpf

eval "$(conda shell.bash hook)"
conda activate mlpng
# conda activate trainer

time srun jupyter notebook --ip=0.0.0.0
