#! /bin/bash
#SBATCH --job-name=converter
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=1024G  # per node
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=0
#SBATCH --output=convert-%j.log
#SBATCH --error=convert-%j.log
#SBATCH --time=04:00:00

module load conda gcc/11.2.0
module load cuda/11.8.0-vbvgppx cudnn/8.7.0.84-11.8-fhn3dpf

eval "$(conda shell.bash hook)"

conda activate trainer

time srun python scripts/hdf5_to_tfds.py
