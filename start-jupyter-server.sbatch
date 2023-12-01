#! /bin/bash
#SBATCH --job-name=VSC-jupyter
#SBATCH --output=.VSC-jupyter.out
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=500G  # per node
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00

#module load conda gcc/11.2.0
#module load cuda/11.8.0-vbvgppx cudnn/8.7.0.84-11.8-fhn3dpf

module load spack conda
module load gcc/11.2.0  cuda/11.8.0-vbvgppx cudnn/8.7.0.84-11.8-fhn3dpf

eval "$(conda shell.bash hook)"

conda activate trainer

time srun jupyter notebook --ip=0.0.0.0
