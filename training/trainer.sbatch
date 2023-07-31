#! /bin/bash
#SBATCH --job-name=trainer
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=500G  # per node
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

module load dev spack conda
module load cuda-11.4.4-gcc-10.3.0-ctldo35 cudnn-8.2.4.15-11.4-gcc-10.3.0-eluwegp

eval "$(conda shell.bash hook)"

conda activate trainer

time srun python training/trainer.py
