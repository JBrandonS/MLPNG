#!/usr/bin/env bash
#SBATCH --job-name=unet-optuna
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=data/logs/unet_optuna/%j.log
#SBATCH --time=0-8:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

echo "Running unet_optuna hyperparameter optimization on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Arguments: $*"

# Create logging directory
mkdir -p data/logs/unet_optuna

# Load required modules
module purge
module load miniforge gcc openblas openmpi
mamba activate ds25

# Create optuna storage directory
mkdir -p data/tuner

# Run the optimization
time srun python -m mlpng.full_trainer "$@"
echo "Optimization completed at $(date)"
