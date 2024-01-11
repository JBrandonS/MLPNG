#! /bin/bash
#SBATCH --job-name=trainer
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=1024G  # per node
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=8
#SBATCH --output=logs/training/%j.log
#SBATCH --error=logs/training/%j.log
#SBATCH --time=2-00:00:00

module load conda gcc/11.2.0
module load cuda/11.8.0-vbvgppx 
module load cudnn/8.7.0.84-11.8-fhn3dpf

eval "$(conda shell.bash hook)"

conda activate trainer

if [ -n "$1" ]
then
    time srun python scripts/trainer.py "$1"
else
    time srun python scripts/trainer.py
fi
