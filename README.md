# Machine Learning for Primordial Non-Gaussianity

## Overview

## Tasks

## Installing

### Data Generator

1. Load in your modules. On m3 I use
    - `module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza`
2. Create a python environment
   - I have provided `conda-envs/datagen-m3.yml` which is the conda env I use on m3.
3. Clone, or download, and install [KSW](https://github.com/AdriJD/ksw) and [optweight](https://github.com/AdriJD/optweight)
   - `optweight` should be installed first, just need run `pip install -e .` in the root directory.
   - For `ksw` run `make && pip install -e . && make check` in the root.
      - You will probably see an error on the make check, this seems to be an issue with the ksw test code and does not affect anything.
4. You can now run the code, using the pipeline with `datagenerator.sh` or manually with `scripts/datagen.py`, `scripts/combiner.py`, and `scripts/estimator.py`.

### Trainer

1. Load in the modules on Superpod
   - `module load conda nvidia/nvhpc`
2. Create a python environment with the required packages
   - I have provided `conda-envs/mlpng-gpu.yml` which is the conda env I use on Superpod.
       - If you have issues installing mpi4py with the nvhpc module, try `CFLAGS=-noswitcherror pip install mpi4py`
3. Ensure your data is avaible in the path expected by the settings file.
4. You can now run the code, using `trainer.sh`.

## Running

> You will need to change the conda env in the `.sbatch` scripts located inside `sbatch/`
>
> The best method to run large amounts data is to use slurm job arrays.

### Data Generation

The data generator is controlled by settings files located in the `settings/` directory. You can specify a different settings file as an argument when running the Python scripts.

The `datagen.py` script generates the alms and patches.

> **Note:** 
> 
> It's recommended to use slurm job arrays for this script. If you change the `array` setting in the `sbatch/datagen.sbatch` file, make sure to update the `narray` value in your settings file to match the size of the job arrays. This ensures the `combiner` and `estimator` scripts handle the data correctly. I may possibly automate this later.

#### Data Combination

After data generation, if you're using job arrays, the data will be in separate files per job. You can use the `combiner.py` script to combine the data into a single file. Simply point it to the correct settings file. See the `datagen.sh` script for an example of how to automate this.

#### Data Estimation

The `estimator` script applies the KSW estimator to the data. Point it to the correct settings file. This script also 'finalizes' the data by removing the `.nc` at the end of the file name.

#### Simplified Data Generation

For convenience, a `datagenerator.sh` script is provided. To use it:

1. Make any necessary changes to your settings file.
2. Check the sbatch file in `sbatch/datagen.sbatch`. You may need to correct the array number to match the settings you will be using. Check and change the conda env used in all the `sbatch` files in `sbatch/`.
3. Point the `datagen.sh` script to the correct settings file and run it.

For large runs you will want to use the command `nohup bash datagenerator.sh &`. This runs the script in the background (`&`) and keeps the process running if your connection drops (`nohup`). Once ran it will be save to log out of your ssh connection. **This script is safe to run on the log-in nodes as it does no intensive work and spends most of its time idle**.

### Training

The training pipeline is very simple. From superpod,

0. Ensure your data is fully generated and avaible on the superpod filesystem.
1. Create your model, follow the example in `scripts/isensee_attn.py`, or any of the other model files.
2. Point the `trainer.sh` script to the correct settings files you would like to train on and point the `sbatch $JOB1 "scripts/isensee_attn.py" "$SETTINGSFILE"` line to the correct model file.
3. Run the `trainer.sh` script.

## Some Notes

### Notes on data format

The data is stored in `hdf5` files as they allow reading and appending data without the need to load the whole dataset into memory. You can think of these files as python dicts. The data is stored in the following format:

- For alms in `data/alm_cache` the following key and values are stored:
  - `alm` : the gaussian $a_{\ell m}$ values in `shape: (nsims, len(polarizations), data)`
  - `almng`: the non-gaussian $a_{\ell m}^{NG}$ values in `shape :(nsims, len(polarizations), data)`
    - where the size of `data` depends on the settings used in a complicated way, see `healpy` or `pixell` documentation.
  - `settings`: A copy of the settings file used to generate the data, for reference.

- For the data files in `data/[un]lensed/`:
  - `estimates`: the KSW estimates of the bispectrum in `shape: (nsims,)`
  - `errors`: the percent diff errors of the estimates vs true fnls in `shape: (nsims,)`
  - `fnls`: the fnl values used to generate the data in `shape: (nsims,)`
    - if you need to align the fnls with the patches use `np.repeat(fnls, npatches)`.
  - `patches`: the patches used to generate the data in `shape: (nsims, len(polarizations), npatches, nside, nside)`
  - `settings`: A copy of the settings file used to generate the data for reference.

### Notes on files

Simultaneous runs are supported as long as the filenames do not collide. The alms settings may match as long as they have been previously generated. Existing non-completed files (ending with `.nc`) will be overwritten in the data generation step. Completed alm files will be overwritten once the alms are fully generated and, if using slurm job arrays, combined. Completed data files will be overwritten once the estimator is finished.

For best results use the `cleanup.sh` script to remove all incomplete files before starting a new run.

Filenames are generated from select settings.

For alms: `[nside][?_disable_noise]_[polarizations]_[nsims]`.

For the data: `[nside][?_disable_noise]_[polarizations]_[nsims]x[npatches]_fnl-[fnl_low]-[fnl_high]`.

- here `[?_disable_noise] = _nn` is only included if `disable_noise` is True
- `[polarizations]`, will be one of `T`, `E`, or `TE`
- other values are just integers

## Thanks

Thomas and rest of dutch group (need to add the full groups here)

Adri Duivenvoorden:

- [Minimizing gravitational lensing contributions to the primordial bispectrum covariance](http://arxiv.org/abs/1912.07619)
- [Primordial Non-Gaussianity](http://arxiv.org/abs/1903.04409)
- [ksw github](https://github.com/AdriJD/ksw)
- [optweight github](https://github.com/AdriJD/optweight)


https://github.com/ai4cmb/NNhealpix/tree/master

@article{ KrachmalnicoffTomasi2019,
	author = {{Krachmalnicoff, N.} and {Tomasi, M.}},
	title = {Convolutional neural networks on the HEALPix sphere: a pixel-based algorithm and its application to CMB data analysis},
	DOI= "10.1051/0004-6361/201935211",
	url= "https://doi.org/10.1051/0004-6361/201935211",
	journal = {A\&A},
	year = 2019,
	volume = 628,
	pages = "A129",
}
