#!/usr/bin/env bash
#SBATCH --job-name=phi_scale
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=4
#SBATCH --output=data/logs/phi_scale/%j.log
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

mkdir -p data/logs/phi_scale

echo "Running phi_scale_analysis on $(hostname) with cli $*"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null | wc -l)"

module purge
module load miniforge gcc openblas openmpi
mamba activate ds25

time srun python -m mlpng.phi_scale_analysis "$@"
