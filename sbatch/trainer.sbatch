#!/usr/bin/env bash
#SBATCH --job-name=trainer
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --output=data/logs/trainer/%j.log
#SBATCH --time=1-00:00:00
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

module="mlpng.${1:-scn-jorik}"
echo "Running trainer with module $module on $(hostname)"
shift # remove the first argument from $@

mkdir -p data/logs/trainer

module purge
# module load miniforge gcc/13.2.0 cuda/12.4.1-vz7djzz hpc-x/2.17.1 openblas/0.3.26-sauswx6 amdlibflame/4.2-5lac6gu nccl/2.21.5-1-nnpmvmq cudnn/8.9.7.29-12-lo4uzx3
# export LAPACK=/hpc/mp/repos/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-13.2.0/amdlibflame-4.2-5lac6guhax5dyr5sxwjpfnaqvh36gbm5/lib/libflame.so
module load miniforge gcc openblas openmpi
mamba activate ds25

time srun python -m "$module" "$@"
