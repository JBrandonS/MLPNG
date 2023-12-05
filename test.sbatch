#! /bin/bash
#SBATCH --job-name=test
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=500G  # per node
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/training/test-%j.log
#SBATCH --error=logs/training/test-%j.log
#SBATCH --time=2-00:00:00

module load spack conda
module load gcc/11.2.0  cuda/11.8.0-vbvgppx cudnn/8.7.0.84-11.8-fhn3dpf

eval "$(conda shell.bash hook)"

conda activate trainer

time srun python test.py