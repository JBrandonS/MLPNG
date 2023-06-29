#! /bin/bash
#SBATCH --job-name=2k-datagen
#SBATCH --partition=highmem,standard-l,standard-s
#SBATCH --ntasks=1
#SBATCH --mem=500G  # per node
#SBATCH --cpus-per-task=5
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --array=1-2000

module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl gcc-11.2.0/openmpi fftw

eval "$(conda shell.bash hook)"
conda activate mlpng

# export CPATH=$CPATH:/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/include
export FFTWROOT=/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/lib/
export CPATH="/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/lib/:$CPATH"
export C_INCLUDE_PATH="/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/include/:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/hpc/m3/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.2.0/fftw-3.3.10-gz7qikisgnrxlrj5dwrxmpceu73vg434/include/:$CPLUS_INCLUDE_PATH"

time srun python datagen.py
