#! /bin/bash
#SBATCH --job-name=UNET
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=256G  # per node
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=jwryan@mail.smu.edu
#SBATCH --mail-type=ALL    # same as =BEGIN,FAIL,END

module load conda
module load gcc/11.2.0
module load cuda/11.8.0-vbvgppx
module load cudnn/8.7.0.84-11.8-fhn3dpf

eval "$(conda shell.bash hook)"

conda activate /users/jwryan/tensorflow_2.9

time srun python UNET_Fnl_JR_minimal.py