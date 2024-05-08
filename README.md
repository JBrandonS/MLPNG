# Machine Learning for Primordial Non-Gaussianity

An $A_{lm}^{NG}$ generator and training pipeline for machine learning models to estimate the local non-gaussianity from the data.

## Overview

This code is designed to do two things, provide a generation framework for generating non-gaussian CMB data, provide a training framework to run models on the generated data. It will generate both full $A_{lm}$ arrays and flat sky patch cuts. These can be of arbitrary lmax and nside, T and/or E polarizations, and with or without lensing. It has been ran and tested extensively on SMU's Superpod and M3 systems, but may require some tweaking to run on other systems. In particular, the main bash scripts and the sbatch script will need to be updated for new systems. 

The code has been broken into 2 different parts, this is due to policies not allowing for the data generation on the same system as the training. If you are not limited by this you may want to combine the conda environments and the generation and training scripts into one.

The data generation is controlled by settings files, which are used to specify the parameters of the data to be generated. The data is then gereated in two steps, first the $A_{lm}\text{s}$ are created in batches according to the settings file. Secondly, if enabled in the `generator.sh` script, cut sky patches will be generated from the $A_{lm}\text{s}$. Finally the code will run the `combiner.py` script to combine the large number of data files into 2, one for the alms and one for the patches.

The training is controlled by the `scripts/models/` files, which specify the architecture of the model to be trained. The `trainer.sh` script will run the `scripts/trainer.py` script with the specified model and settings file. The model will be trained on the data matching the setting file used.

> The setting matching may not be perfect, and some collisions can occur. If in doubt generate with a unique `base_name` in the settings file.

## Installing

### Data Generator

1. Load in your modules, otherwise setup your dev environment to have access to `conda` and `MPI`. On M3 I use
    - `module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza`
2. Create and setup a python environment with the required packages
   - I have provided `conda-envs/mlpng.yml` which is the conda env I use on m3, you can view this file to find what packages to use. The exact versions should not be important for this code to run, but I cannot gaurentee that the code will work with newer versions of the packages.
3. Clone, or download, and install [KSW](https://github.com/AdriJD/ksw) and [optweight](https://github.com/AdriJD/optweight)
   - `optweight` should be installed first, just need run `pip install -e .` in the root directory.
   - For `ksw` run `make && pip install -e . && make check` in the root.
      - You will probably see an error on the make check, this seems to be an issue with the ksw test code and does not affect anything.
4. You can now run the code, using the pipeline with `generator.sh` or manually with `scripts/almgen.py`, `scripts/patchgen.py`, `scripts/combiner.py`, and `scripts/estimator.py`. See [Running](#running) for more information.

### Trainer

1. Load in the modules on Superpod
   - `module load conda nvidia/nvhpc`
2. Create and setup a python environment with the required packages
   - I have provided `conda-envs/mlpng-gpu.yml` which is the conda env I use on Superpod.
       - If you have issues installing mpi4py with the nvhpc module, try `CFLAGS=-noswitcherror pip install mpi4py`
3. Ensure your data is available in the path expected by the settings file, you can override this with `base_dir` in the settings file or via command line.
4. You can now run the code, using `trainer.sh`, or by running the `scripts/trainer.py` with a `--model` argument. Read below for more information.

## Running

> The best method to run large amounts data is to use slurm job arrays.

### Generator Script

For convenience, a `generator.sh` script is provided. To use it:

1. Make any necessary changes to your settings files in `settings/`.
2. Check the sbatch files in `sbatch/`. You may need to correct the array number to match the settings you will be using. Check and change the conda env used in all the `sbatch` files in `sbatch/`. You may also want to configure the SLURM options here such as `partition`, `mem`, `cpus-per-task`, and `time` to help your jobs queue faster depending on your needs.
3. Point the `generator.sh` script to the correct settings files by changing `SETTINGS`, and update any CLI overrides you want in the `ARGS`. If you change the `narray` value you will need to add this to the `SLURM_ARGS`.

This script will use SLURMs run chaining to queue up runs but only start them once a previous run has completed.

### Data Generation

The data generator is controlled by settings files located in the `settings/` directory. You can specify a different settings file as an argument when running the Python scripts. For some arguments a command line option is available which will take priority. All options have defaults. See `scripts/core.py` for how everything is implemented.

Generate the $A_{lm}$ values with the `almgen.py`, optionally, you can generate the patched data with `patchgen.py`


#### Data Combination

After data generation, if you're using job arrays, the data will be in separate files per job. You can use the `combiner.py` script to combine the data into a single file. Simply give it the correct settings. 

#### KSW Estimation

The `estimator` script applies the [KSW estimator](https://github.com/AdriJD/ksw) to the data. You run this like the other scripts, provide the settings via a settings file with optionally CLI overrides and run it. 

### Training

The training pipeline is very simple. From Superpod,

0. Ensure your data is fully generated and available on the Superpod filesystem.
1. Create your model
   - You can use the provided models in `scripts/models/` or create your own.
   - The model should be a class that inherits from `scripts.models.ModelCore`, needs a `_model` method that takes in at least the inputs and which outputs the final layer, and can be registered with `@scripts.models.register_model`
   - See `scripts/models/isensee.py` for a standalone example.
2. Point the `trainer.sh` script to the correct settings and model files.
3. Run the `trainer.sh` script.

[Weights and Biases](https://wandb.ai/) is used for logging and tracking the training, similar to an online tensorboard with a few extra features. You will need to setup an account and install the `wandb` package to use this feature. You can enable this in the `training.py` file.

## Some Notes

### Notes on data format

The data is stored in `hdf5` files as they allow reading and appending data without the need to load the whole dataset into memory. You can think of these files as python dicts. The data is stored in the following format:

- For alms in `data/alms` the following key and values are stored:
  - `alm` : The $a_{\ell m}^{NG,loc}$ values. `Shape: (nsims, len(polarizations), data)`
  - `fnl`: The $f_{nl}$ values to be used. `Shape (nsims, len(polarizations), 1)`

- For the patch data files in `data/patches/`:
  - `fnl`: the $f_{nl}$ values used to generate the data, a copy from the alm file. `Shape: (nsims, len(polarizations), 1)`
  - `patch`: the patch that have been cut from the full sky maps given by the $a_{\ell m}^{NG,loc'}$ values at `alm[nsims, pol]`. `Shape: (nsims, len(polarizations), npatches, nside, nside)`

- The estimator script will add the following values to the alm file:
  - `fisher`:  the fisher value found by the estimator. `Shape: (nsims,)`
  - `estimate`: the KSW estimates of the bispectrum. `Shape: (nsims,)` or `(1000,)` if `estimate_1k` is True
  - `estimate_1k`: A flag to indicate that we only calcualted the estimate for the first 1k values.
  - `error`: the percent diff errors of the estimates vs true fnls. `Shape: (nsims,)`


### Notes on files

Simultaneous runs are supported as long as the filenames do not collide. The alms settings may match as long as they have been previously generated. Existing non-completed files (ending with `.nc`) will be overwritten in the data generation step. Completed alm files will be overwritten once the alms are fully generated and, if using slurm job arrays, combined.

#### Filenames

Filenames are generated from select settings for easy reading once you understand the format. It is possible to provide a `base_name` in the settings file to override the default naming scheme. The default naming scheme is as follows:

For alms: `l[lmax]_n[nside]_[lensing][?-noise]_[polarizations]x[total sim]`
   - example: `l500_n128_l-nn_Tx100000.hdf5`
      - If you provide a `base_name` the output will be `base_name.hdf5`

For the patches data: `l[lmax]_n[nside]_[lensing][?-_noise]_[polarizations]x[total sim]x[npatches]`.
   - example: `l500_n128_l-nn_Tx100000x10.hdf5`
      - If you provide a `base_name` the output will be `base_namex[npathces].hdf5`

with
- `[lensing] = l | ul` for lensed or unlensed sims
- `[?_noise] = -nn` is only included if `noise` is True
- `[polarizations]`, will be one of `T`, `E`, or `TE`
- other values are just integers / float

See `scripts/core.py:_init_paths` for the where this gets set

## Thanks

Primary development by:
   Brandon Stevenson, Joe Ryan, Joel Meyers

Additional thanks to:

 - Daan Meerburg

 - Jorik Melsen 

 - Thomas Flöss
 
  - Adri Duivenvoorden

### Some References
   - [Minimizing gravitational lensing contributions to the primordial bispectrum covariance](http://arxiv.org/abs/1912.07619)
   - [Primordial Non-Gaussianity](http://arxiv.org/abs/1903.04409)
   - [ksw github](https://github.com/AdriJD/ksw)
   - [optweight github](https://github.com/AdriJD/optweight)

## Problems

 - Currently running with a beam_width will cause a bias, I plan to look into this later but if anyone wants to fix that feel free.
 - The tuning code is not great, and probably wont be imporved much as it isn't that useful.
 - E polarizations, and lensing is not well tested.

## License

Right now lets keep this code private, in future I'd like to GPL / MIT it, but I am open to talking about that.