#! /bin/bash
#SBATCH --job-name=combiner
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=%j.log
#SBATCH --time=6:00:00

module purge
module load conda nvidia/nvhpc

eval "$(conda shell.bash hook)"
conda activate mlpng-gpu

files=(
"l1000_n512_ul-nn_T_100000.alms.hdf5" 
"l2000_n2048_ul-nn_T_100000.alms.hdf5"    
"l750_n256_ul_T_1000.alms.hdf5"
"l1000_n512_ul_T_1000.alms.hdf5"       
"l500_n128_ul-nn_T_100000.alms.hdf5"    
)

for file in "${files[@]}"
do
    time srun python scripts/converter.py "data/alms/$file"
done
