#! /bin/bash
#SBATCH --job-name=training
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=1024G  # per node
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=2
#SBATCH --output=logs/training/%j.log
#SBATCH --error=logs/training/%j.log
#SBATCH --time=2-00:00:00

module purge
module load conda nvidia/nvhpc

eval "$(conda shell.bash hook)"
conda activate mlpng-gpu 

time srun python "$1" "$2"
