# Machine Learning for Primordial Non-Gaussianity

## Overview

## Installing

### Data Generator

1. Load in your modules. On M3 I use
    - `module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza`
2. Create and setup a python environment with the required packages
   - I have provided `conda-envs/mlpng.yml` which is the conda env I use on m3.
3. Clone, or download, and install [KSW](https://github.com/AdriJD/ksw) and [optweight](https://github.com/AdriJD/optweight)
   - `optweight` should be installed first, just need run `pip install -e .` in the root directory.
   - For `ksw` run `make && pip install -e . && make check` in the root.
      - You will probably see an error on the make check, this seems to be an issue with the ksw test code and does not affect anything.
4. You can now run the code, using the pipeline with `generator.sh` or manually with `scripts/almgen.py`, `scripts/patchgen.py`, `scripts/combiner.py`, and `scripts/estimator.py`.

### Trainer

1. Load in the modules on Superpod
   - `module load conda nvidia/nvhpc`
2. Create and setup a python environment with the required packages
   - I have provided `conda-envs/mlpng-gpu.yml` which is the conda env I use on Superpod.
       - If you have issues installing mpi4py with the nvhpc module, try `CFLAGS=-noswitcherror pip install mpi4py`
3. Ensure your data is available in the path expected by the settings file, you can override this with `base_dir` in the settings file or via command line.
4. You can now run the code, using `trainer.sh`.

## Running

> The best method to run large amounts data is to use slurm job arrays.

### Generator Script

For convenience, a `generator.sh` script is provided. To use it:

1. Make any necessary changes to your settings files in `settings/`.
2. Check the sbatch files in `sbatch/`. You may need to correct the array number to match the settings you will be using. Check and change the conda env used in all the `sbatch` files in `sbatch/`. You may also want to configure the SLURM options here such as `mem`, `cpus-per-task`, and `time` to help your jobs queue faster depending on your needs.
3. Point the `generator.sh` script to the correct settings files by changing `SETTING_DIR` and `SETTINGS` if needed and run it.

This script will use SLURMs run chaining to queue up runs but only start them once a previous run has completed.

### Data Generation

The data generator is controlled by settings files located in the `settings/` directory. You can specify a different settings file as an argument when running the Python scripts. For some arguments a command line option is available which will take priority. All options have defaults. See `scripts/utils/config.py` for how everything is implemented.

Generate the $A_{lm}s$ with the `almgen.py`, optionally, you can generate the patched data with `patchgen.py`

> **Note:** 
> 
> It's recommended to use slurm job arrays for this script. If you change the `array` setting in the `sbatch` files, make sure to update the `narray` value in your settings file to match the size of the job arrays, or you can also set the `ARGS('--narray')` and `SLURM_ARGS('--array')` options in the generator script. This ensures the `combiner` and `estimator` scripts handle the data correctly.



#### Data Combination

After data generation, if you're using job arrays, the data will be in separate files per job. You can use the `combiner.py` script to combine the data into a single file. Simply give it the correct settings. 

#### KSW Estimation

The `estimator` script applies the [KSW estimator](https://github.com/AdriJD/ksw) to the data. You run this like the other scripts, provide the settings via a settings file with optionally CLI overrides and run it. 

### Training

The training pipeline is very simple. From Superpod,

0. Ensure your data is fully generated and available on the Superpod filesystem.
1. Create your model, follow the example in `scripts/attn_alm.py`, or any of the other model files in `scripts/altmodels`.
2. Point the `trainer.sh` script to the correct settings and model files.
3. Run the `trainer.sh` script.

## Some Notes

### Notes on data format

The data is stored in `hdf5` files as they allow reading and appending data without the need to load the whole dataset into memory. You can think of these files as python dicts. The data is stored in the following format:

- For alms in `data/alms` the following key and values are stored:
  - `alm` : The $a_{\ell m}^{NG,loc'}$ values. `Shape: (nsims, len(polarizations), data)`
  - `fnl`: The $f_{nl}$ values to be used. `Shape (nsims, len(polarizations), 1)`
  - `settings`: A copy of the settings file used to generate the data, for reference.

- The estimator script will add the following values to the alm file:
  - `fisher`:  the fisher value found by the estimator. `Shape: (nsims,)`
  - `fisher_iso`: the isotropic fisher value found by the estimator. `Shape: (nsims,)`
  - `estimate`: the KSW estimates of the bispectrum. `Shape: (nsims,)`
  - `error`: the percent diff errors of the estimates vs true fnls. `Shape: (nsims,)`

- For the patch data files in `data/patches/`:
  - `fnl`: the $f_{nl}$ values used to generate the data, a copy from the alm file. `Shape: (nsims, len(polarizations), 1)`
  - `patch`: the patchs that have been cut from the full sky maps given by the $a_{\ell m}^{NG,loc'}$ values at `alm[nsims, pol]`. `Shape: (nsims, len(polarizations), npatches, nside, nside)`
  - `settings`: A copy of the settings file used to generate the data for reference.

### Notes on files

Simultaneous runs are supported as long as the filenames do not collide. The alms settings may match as long as they have been previously generated. Existing non-completed files (ending with `.nc`) will be overwritten in the data generation step. Completed alm files will be overwritten once the alms are fully generated and, if using slurm job arrays, combined. Completed data files will be overwritten once the estimator is finished.

For best results use the `cleanup.sh` script to remove all incomplete files before starting a new run.

#### Filenames

Filenames are generated from select settings for easy reading once you understand the format.

For alms: `l[lmax]_n[nside]_[lensing][?-disable_noise]_[polarizations]_[total sim]`
   - example: `l500_n128_l-nn_T_100000.alms.hdf5`

For the patches data: `l[lmax]_n[nside]_[lensing][?-_disable_noise]_[polarizations]_[nsims]x[npatches]_fnl[fnl_low]-[fnl_high]`.
   - example: `l500_n128_l-nn_T_100000x10_fnl-1000-1000.hdf5`

with
- `[lensing] = l | ul` for lensed or unlensed sims
- `[?_disable_noise] = -nn` is only included if `disable_noise` is True
- `[polarizations]`, will be one of `T`, `E`, or `TE`
- other values are just integers / float

## Thanks

Thomas and rest of dutch group (need to add the full groups here)

Adri Duivenvoorden:

- [Minimizing gravitational lensing contributions to the primordial bispectrum covariance](http://arxiv.org/abs/1912.07619)
- [Primordial Non-Gaussianity](http://arxiv.org/abs/1903.04409)
- [ksw github](https://github.com/AdriJD/ksw)
- [optweight github](https://github.com/AdriJD/optweight)