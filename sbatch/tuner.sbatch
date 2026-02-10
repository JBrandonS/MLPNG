#!/usr/bin/env bash
#SBATCH --job-name=tuner
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=data/logs/tuner/%j.log
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

module="mlpng.${1:-optuna_trainer}"
echo "Running tuner with module $module on $(hostname) (args: $*)"
shift # remove the first argument from $@

mkdir -p data/logs/tuner

module purge
module load miniforge gcc openblas openmpi
mamba activate ds25

time srun python -m "$module" "$@"
