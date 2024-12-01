#!/usr/bin/env bash
#SBATCH --job-name=deepsphere
#SBATCH --output=data/logs/deepsphere/%j.log
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

module purge
module load spack conda gcc/13.2.0 cuda/12.4.1-vz7djzz hpc-x/2.17.1 openblas/0.3.26-sauswx6 amdlibflame/4.2-5lac6gu nccl/2.21.5-1-nnpmvmq cudnn/8.9.7.29-12-lo4uzx3
module list

conda activate deepsphere

time srun python -m mlpng.scn-unet
