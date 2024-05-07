#!/usr/bin/env bash
#SBATCH --job-name=tuning
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=2000G
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=8
#SBATCH --output=logs/tuning/%j.log
#SBATCH --time=2-00:00:00
#SBATCH --exclusive

# keras tuner / possibly cuda, has a bug that means memory is not cleaned correctly and you get OOM
# this script therefore will loop through and retry the job until it completes successfully
# runs for max time, or 30 crashes
# note that keras tunes saves often so this is not a huge issue, but it is annoying

# create the logging dir
mkdir -p logs/tuning

# copy the file to a runs dir so we can have many going at once
# this is needed if you are running multiple tuning jobs at once
# not working rn
# mkdir -p data/tuning/runs
# FILE=data/tuning/runs/"$SLURM_JOB_ID"-tuner.py
# cp scripts/tuner.py "$FILE"

module purge
module load conda nvidia/nvhpc
eval "$(conda shell.bash hook)"
conda activate mlpng-gpu 

i=0
command="srun python -m scripts.tuner $*"
echo "Running command: $command"
while [ $i -lt 30 ]; do
    if eval "$command"; then
        # If the exit status is 0, the command completed successfully, so break the loop
        break
    else
        # If the exit status is not 0, the command failed, so print an error message and try again
        echo "Job failed, retrying in 3 seconds..."
        sleep 3
    fi
    i=$((i+1))
done
# rm "$FILE"
echo "Done!"
