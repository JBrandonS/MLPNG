#! /bin/bash
#SBATCH --job-name=dataGenerator
##SBATCH --partition=batch
#SBATCH --partition=standard-mem-s,standard-mem-m,standard-mem-l
#SBATCH --ntasks=1
#SBATCH --mem=250G  # per node
#SBATCH --cpus-per-task=36
#SBATCH --gpus-per-task=0

# module load spack conda
# module load cuda-11.4.4-gcc-10.3.0-ctldo35 cudnn-8.2.4.15-11.4-gcc-10.3.0-eluwegp
module load gcc-9.2 gcc intel python
eval "$(conda shell.bash hook)"

conda activate mlpng

# time srun python -m cProfile -o dataGenerator.prof dataGenerator.py
time srun python dataGenerator.py
