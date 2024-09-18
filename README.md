# Machine Learning for Primordial Non-Gaussianity

An $A_{lm}^{NG}$ generator and training pipeline for machine learning models to estimate the local non-gaussianity from the data. The $A_{lm}\text{s}$ are generated using the modified [Hanson 2009](https://arxiv.org/abs/0905.4732) method. The pipeline is designed to be run on a HPC system with a slurm scheduler.

## Overview

This code is designed to do two things; provide a generation framework for generating non-gaussian CMB data, and provide a training framework to run ML models on the generated data. It will generate both full $A_{lm}$ arrays and full sky patch cuts. The code supports various nside, T or TE polarizations, and with or without lensing. It has been run and tested extensively on SMU's Superpod and M3 systems but may require some tweaking to run on other systems. In particular, the main bash scripts and the sbatch script will need to be updated for new systems.

The code has been broken into 2 different parts, this is due to SMU policies not allowing for the CPU based data generation on the same system as the GPU based training. If you are not limited by this you may want to combine the conda environments and the generation and training scripts into one.

Everything is controlled by settings files, which are used to specify the parameters of the data to be generated. The data is then generated in two steps, first the $A_{lm}\text{s}$ are created in accordance with Hanson 2009, then if lensing has been enabled the $A_{lm}\text{s}$ will lensed before cut sky patches are generated. Secondly, the code will run the `combiner.py` script to combine the large number of data files into one. Data will be saved to the `data_dir`, which by default will be `data/data`. Similar settings can be used to change where the plots, and other large files will be saved. Please see the `settings.md` file for descriptions of all the settings.

The training is controlled by the `scripts/models/` files, which specify the architecture of the model to be trained. The `trainer.sh` script will run the `scripts/trainer.py` script with the specified model and settings file. The model will be trained on the data matching the setting file used.

## Installing

### Data Generator

1. Load in your modules, otherwise set up your dev environment to have access to `conda` and `MPI`. On M3 I use
    - `module load spack conda gcc/11.2.0 gcc-11.2.0/intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza`
2. Create and setup a Python environment with the required packages
   - I have provided `conda-envs/mlpng.yml` which is the conda env I use on m3, you can view this file to find what packages to use. The exact versions should not be important for this code to run, but I cannot guarantee that the code will work with newer versions of the packages.
3. Clone or download, and install [optweight](https://github.com/AdriJD/optweight) and [KSW](https://github.com/AdriJD/ksw)
   - `optweight` should be installed first, you will just need to run `pip install -e .` in the root directory.
   - For `ksw`
      - You need to switch to the dev branch `git checkout -b dev origin/dev`
      - run `make && pip install -e . && make check` in the root.
4. You can now run the code, using the pipeline with `generator.sh` or manually with `scripts/generator.py`, `scripts/combiner.py`, and `scripts/estimator.py`. See [Running](#running) for more information.

### Trainer

1. Load in the modules on Superpod
   - `module load conda nvidia/nvhpc`
2. Create and setup a Python environment with the required packages
   - I have provided `conda-envs/mlpng-gpu.yml` which is the conda env I use on Superpod.
       - If you have issues installing mpi4py with the nvhpc module, try `CFLAGS=-noswitcherrors pip install mpi4py` this has helped in the past but may not work.
3. Ensure your data is available in the path expected by the settings file, you can override this with `base_dir` in the settings file or via the command line.
4. You can now run the code, using `trainer.sh`, or by running `scripts/trainer.py` with a `--model` argument. Read below for more information.

## Running

### Notebooks

Several jupyter notebooks have been created for testing and are located in the `notebooks/` folder. These will use a slightly modified KSW code to run with joblib parallel instead of openMPI as jupyter does not like openMPI. This is not recommended for large scale runs but is useful for testing and debugging.

The `simulator.ipynb` notebook will both simulate generating the data, acting like the `generator.py` script, and run the KSW estimator code like the `estimator.py` script. This can be ran standalone and is a good place to start when making modifications. This will not save any data.

The `estimator.ipynb` will run the KSW estimator code on previously generated data.

The `trainer.ipynb` will run the training code on previously generated data.

### Data Generation

For convenience, a `generator.sh` script is provided. To use it:

1. Make any necessary changes to your settings files in `settings/`.
2. Check the sbatch files in `sbatch/`. You may need to correct the array number to match the settings you will be using.
   - Change the conda env used in all the `sbatch` files in `sbatch/`.
   - You may also want to configure the slurm options here such as `partition`, `mem`, `cpus-per-task`, and `time` to help your jobs queue faster depending on your needs.
3. Point the `generator.sh` script to the correct settings files by changing `SETTINGS`, and update any CLI overrides you want in the `ARGS`.

You can then run the script.

> This script will use slurm's run chaining to queue up runs but only start them once a previous run has been completed.

#### Manual Data Generation

The data generator is controlled by settings files located in the `settings/` directory. You can specify a different settings file as an argument when running the Python scripts. For some arguments, a command line option is available which will take priority. All options have defaults. See `settings/settings.md` for more.

To manually generate the data you will need to run the generator python module, providing at least a settings file with optional CLI overrides, from the root folder:

```sh
python -m scripts.generator --lensing settings/planck.json
```

After data generation, the data will be in separate files based on the array size you have used. You can use the `combiner.py` module to combine the data into a single file. Simply give it the same settings that you used to generate the data.

```sh
python -m scripts.combiner --lensing settings/planck.json
```

The `estimator` script applies the [KSW estimator](https://github.com/AdriJD/ksw) to the data. You can run this once the data is combined. You run this like the other scripts

```sh
python -m scripts.estimator --lensing settings/planck.json
```

### Training

The training pipeline is very simple. From Superpod,

0. Ensure your data is fully generated and available on the Superpod filesystem.
1. Update the `sbatch/trainer.sbatch` file to match the cluster settings
2. Create your model
   - You can use the provided models in `scripts/models/` or create your own.
   - The model should be a class that inherits from `scripts.models.ModelCore`, needs a `_model` method that takes in at least the inputs and outputs the final layer, and must be registered with the `@scripts.models.register_model` decorator.
   - See `scripts/models/isensee.py` for a standalone example which can also be manually run.
3. Point the `trainer.sh` script to the correct settings for the data you are using and to the models.

#### Manual Training

To manually train a model, you can run the `scripts/trainer.py` module with the desired settings file, you will need to provide a CLI model option. For example, to train the `isensee` model on the `planck` data as created above:

```sh
python -m scripts.trainer --model isensee --lensing settings/planck.json
```

> [Weights and Biases](https://wandb.ai/) can be used for logging and tracking the training, similar to an online tensorboard with a few extra features. You will need to set up an account and install the `wandb` package to use this feature and then enable it in the run settings.

## Some Notes

### Notes on data format

The data is stored in `hdf5` files as they allow reading and appending data without needing to load the whole dataset into memory. You can think of these files as Python dictionaries. The data is stored in the following format:

- `alm` : The $a_{\ell m}^{NG,loc}$ values. `Shape: (nsims, npol, nelem)`
- `alm_lensed` : If lensing is enabled, The lensed $a_{\ell m}^{NG,loc}$ values. `Shape: (nsims, npol, nelem)`
- `fnl`: The $f_{nl}$ values used. `Shape (nsims, 1)`
- `patch`: The patch that has been cut from the full sky maps given by the $a_{\ell m}^{NG,loc'}$ values at `alm[nsims, pol]`. `Shape: (nsims, npol, npatches, nside, nside)`

- The estimator script will add the following values to the data file:
  - `fisher`:  The fisher value found by the estimator. `Shape: (nsims,)`
  - `estimate`: The KSW estimates of the bispecturm. `Shape: (nsims,)` or `(num_estimates,)` if `num_estimates` setting is provided and smaller than `nsims`.
  - `error`: The percent diff errors of the estimates vs true fnls. `Shape: (nsims,)`

### Notes on files

Simultaneous runs are supported as long as the filenames do not collide. If files are found to exists the scripts are designed to exit quickly but will not fail. This allows easy continuation of incomplete runs, i.e. if some of the sims timed out everything is designed so you can just change the time in the slurm settings and rerunning should then only generate the missing sims. This will cause issues if two runs are started at the same time with the same settings file, I do not plan to handel that case.

#### Filenames

Filenames are generated from select settings for easy reading once you understand the format. It is possible to provide a `base_name` in the settings file to override the default naming scheme. The default naming scheme is as follows:

For alms: `l[lmax]_n[nside]_[lensing]-[noise]_[polarizations]x[total sim]`

- example: `l383_n128_l-nn_Tx100000.hdf5`
  - If you provide a `base_name` the output will be `base_name.hdf5`

with

- `[lensing] = l | ul` for lensed or unlensed sims
- `[_noise] = nn | n` for no noise or for noise
- `[polarizations]`, will be one of `T`, or `TE`
- other values are just integers or floats

> See `scripts/core.py:_init_paths` for the where this gets set

## Thanks

Primary development by:
   Brandon Stevenson, Joe Ryan, Joel Meyers

Additional thanks to:

- Daan Meerburg
- Jorik Melsen
- Thomas Flöss
