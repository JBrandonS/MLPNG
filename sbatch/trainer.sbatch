#!/usr/bin/env bash
#SBATCH --job-name=trainer
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=4
#SBATCH --output=data/logs/trainer/%j.log
#SBATCH --time=2-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

module="mlpng.${1:-scn-jorik}"
echo "Running trainer with module $module on $(hostname)"
shift # remove the first argument from $@

mkdir -p data/logs/trainer

module purge
module load miniforge gcc openblas openmpi
mamba activate ds25

time srun python -m "$module" "$@"
