#! /bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=.vscj.out
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --time=8:00:00


module purge
module load conda nvidia/nvhpc

eval "$(conda shell.bash hook)"
conda activate mlpng-gpu

time srun jupyter notebook --ip=0.0.0.0
