#!/usr/bin/env bash
#SBATCH --job-name=jupyter
#SBATCH --output=.vscj.out
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=8:00:00


module purge
module load conda nvidia/nvhpc

eval "$(conda shell.bash hook)"
conda activate mlpng-gpu

time srun jupyter notebook --ip=0.0.0.0
