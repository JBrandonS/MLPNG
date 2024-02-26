#! /bin/bash
#SBATCH --job-name=training
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --output=data/logs/training/%j.log
#SBATCH --time=2-00:00:00

module purge
module load conda nvidia/nvhpc

eval "$(conda shell.bash hook)"
conda activate mlpng-gpu 

time srun python "$1" "$2"
