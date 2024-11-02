#!/usr/bin/env bash
#SBATCH --job-name=training
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --output=data/logs/training/%j.log
#SBATCH --time=24:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001 

# create the logging dir
mkdir -p data/logs/training

module purge
module load conda nvidia/nvhpc

eval "$(conda shell.bash hook)"
conda activate mlpng-gpu 

time srun python -m mlpng.trainer "$@"
