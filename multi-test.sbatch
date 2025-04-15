#!/usr/bin/env bash
#SBATCH --job-name=multitest
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --output=%j.log
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

echo "Running trainer with module $module on $(hostname)"

module purge
module load spack conda gcc/13.2.0 cuda/12.4.1-vz7djzz hpc-x/2.17.1 openblas/0.3.26-sauswx6 amdlibflame/4.2-5lac6gu nccl/2.21.5-1-nnpmvmq cudnn/8.9.7.29-12-lo4uzx3
module list

conda activate ds25

nvidia-smi
nvcc --version

time srun python multi-test.py
