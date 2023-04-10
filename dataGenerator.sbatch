#! /bin/bash
#SBATCH --job-name=mlpng-dataGenerator
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=128G  # per node
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=0

module load spack conda
module load cuda-11.4.4-gcc-10.3.0-ctldo35 cudnn-8.2.4.15-11.4-gcc-10.3.0-eluwegp
eval "$(conda shell.bash hook)"

conda activate mlpng

# time srun python -m cProfile -o dataGenerator.prof dataGenerator.py
time srun python dataGenerator.py
