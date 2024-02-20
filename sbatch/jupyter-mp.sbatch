#! /bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=.vscj.out
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=1024
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --time=8:00:00


module purge
module load spack conda
module load nvidia/nvhpc

eval "$(conda shell.bash hook)"
conda activate mlpng0gpu

time srun jupyter notebook --ip=0.0.0.0
