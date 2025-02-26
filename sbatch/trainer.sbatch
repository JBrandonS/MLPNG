#!/usr/bin/env bash
#SBATCH --job-name=trainer
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=data/logs/trainer/%j.log
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_P2P_LEVEL=NVL
export NCCL_LAUNCH_MODE=PARALLEL

module="mlpng.${1:-scn-jorik}"
shift # remove the first argument from $@

mkdir -p data/logs/trainer

echo "Running trainer with module $module on $(hostname)"

module purge
module load spack conda gcc/13.2.0 cuda/12.4.1-vz7djzz hpc-x/2.17.1 openblas/0.3.26-sauswx6 amdlibflame/4.2-5lac6gu nccl/2.21.5-1-nnpmvmq cudnn/8.9.7.29-12-lo4uzx3
module list
conda activate ds25

nvidia-smi
nvcc --version

time srun python -m "$module" "$@"
