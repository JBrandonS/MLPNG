#! /bin/bash
#SBATCH --job-name=dataGen
#SBATCH --partition=standard-mem-s,standard-mem-m,standard-mem-l
#SBATCH --ntasks=1
#SBATCH --mem=250G  # per node
#SBATCH --cpus-per-task=36
#SBATCH --output=logs/dg-output_%j.log
#SBATCH --error=logs/dg-error_%j.log

module purge
module load spack intel fftw

eval "$(conda shell.bash hook)"

conda activate mlpng

time srun python _ksw_datagen.py > logs/dg-$SLURM_JOB_ID.log 2>&1
