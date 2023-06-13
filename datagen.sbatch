#! /bin/bash
#SBATCH --job-name=datagen
#SBATCH --partition=highmem
#SBATCH --ntasks=1
#SBATCH --mem=1024G  # per node
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/dg-output_%j.log
#SBATCH --error=logs/dg-error_%j.log

module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl gcc-11.2.0/openmpi fftw

eval "$(conda shell.bash hook)"
conda activate mlpng

# export CPATH=$CPATH:/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/include
export FFTWROOT=/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/lib/
export CPATH="/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/lib/:$CPATH"
export C_INCLUDE_PATH="/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/include/:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/include/:$CPLUS_INCLUDE_PATH"

time srun python datagen.py > logs/dg-$SLURM_JOB_ID.log 2>&1
