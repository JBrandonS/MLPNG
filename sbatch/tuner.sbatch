#!/usr/bin/env bash
#SBATCH --job-name=tuning
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --output=data/logs/tuning/%j.log
#SBATCH --time=1-00:00:00
#SBATCH --account=jrmeyers_ml_cmb_0001

# keras tuner / possibly cuda, has a bug that means memory is not cleaned correctly and you get OOM
# this script therefore will loop through and retry the job until it completes successfully
# runs for max time, or 30 crashes
# note that keras tunes saves often so this is not a huge issue, but it is annoying

mkdir -p data/logs/tuning

module purge
module load spack conda gcc/13.2.0 cuda/12.4.1-vz7djzz hpc-x/2.17.1 openblas/0.3.26-sauswx6 amdlibflame/4.2-5lac6gu nccl/2.21.5-1-nnpmvmq cudnn/8.9.7.29-12-lo4uzx3

conda activate deepsphere

i=0
command="srun python -m mlpng.tuner"
echo "Running command: $command"
while [ $i -lt 3 ]; do
    if eval "$command"; then
        # If the exit status is 0, the command completed successfully, so break the loop
        break
    else
        # If the exit status is not 0, the command failed, so print an error message and try again
        echo "Job failed, retrying in 3 seconds..."
        sleep 3
    fi
    i=$((i + 1))
done
# rm "$FILE"
echo "Done!"
