#! /bin/bash
#SBATCH --job-name=training
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/training/%j.log
#SBATCH --time=06:00:00

# create the logging dir
mkdir -p logs/training

module purge
module load conda nvidia/nvhpc

eval "$(conda shell.bash hook)"
conda activate mlpng-gpu 

time srun python "$@"
