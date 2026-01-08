# Machine Learning for Primordial Non-Gaussianity

An $A_{lm}^{NG}$ generator and training pipeline for machine learning models to estimate the local non-gaussianity from the data. The $A_{lm}\text{s}$ are generated using the modified [Hanson 2009](https://arxiv.org/abs/0905.4732) method. The pipeline is designed to be run on a HPC system with a slurm scheduler.

## Overview

This code is designed to do two things; provide a generation framework for generating non-gaussian CMB data, and provide a training framework to run ML models on the generated data. It will generate both full $A_{lm}$ arrays and full sky patch cuts. The code supports various nside, T, or E polarizations (with TE planned), and with or without lensing or noise. It has been run and tested extensively on SMU's Superpod and M3 systems but may require some tweaking to run on other systems. In particular, the main bash scripts and the sbatch script will need to be updated for new systems.

> The code has been broken into 2 different parts, due to SMU policies not allowing for the CPU based data generation on the same system as the GPU based training. If you are not limited by this you may want to combine the conda environments and the generation and training scripts into one.

## Installing

### Data Generator

1. Load in your modules, otherwise set up your dev environment to have access to `conda` and `MPI`. On M3 I use
    - `module load conda gcc/11.2.0 intel-oneapi-mkl/2022.2.1-c4efjsy fftw/3.3.10-gz7qiki openmpi/4.1.6-a4ksrza`
2. Create and setup a Python environment with the required packages
   - I have provided `conda-envs/mlpng.yml` which is the conda env I use on m3.
3. Clone or download, and install [optweight](https://github.com/AdriJD/optweight) and [KSW](https://github.com/AdriJD/ksw)
   - `optweight` should be installed first, just run `pip install -e .` in the root directory.
   - For `ksw`
      - You need to switch to the dev branch `git checkout -b dev origin/dev`
      - run `make && pip install -e . && make check` in the root.
4. You can now run the code, using the automated pipeline with `generator.sh` (which includes initialization, generation, and combining), or manually with the individual modules. See [Running](#running) for more information.

### Trainer

1. Load in the modules on Superpod
   - `module load spack conda gcc/13.2.0 cuda/12.4.1-vz7djzz hpc-x/2.17.1 openblas/0.3.26-sauswx6 amdlibflame/4.2-5lac6gu nccl/2.21.5-1-nnpmvmq cudnn/8.9.7.29-12-lo4uzx3`
2. Create and setup a Python environment with the required packages
   - I have provided `conda-envs/mlpng-gpu.yml` which is the conda env I use on Superpod.
3. You can now run the code, using the provided `trainer.sh`, or by running one of the `mlpng/scn-*.py` files.

## Running

### Notebooks

Several jupyter notebooks have been created for testing and are located in the `notebooks/` folder. These have been used in the testing and debugging phase and may be useful to look at for a simple example of how to run the code.

The `simulator.ipynb` notebook will both simulate generating the data, acting like the `generator.py` script, and run the KSW estimator code like the `estimator.py` script. This can be ran standalone and is a good place to start when making modifications. This will not save any data.

The `trainer.ipynb` will run the ML training code on previously generated data.

### Data Generation

For convenience, a `generator.sh` script is provided. To use it:

1. Make any necessary changes to your settings files in `settings/`.
2. Check the sbatch files in `sbatch/`. You may need to correct the array number to match the settings you will be using.
   - Change the conda env used in all the `sbatch` files in `sbatch/`.
   - You may also want to configure the slurm options here such as `partition`, `mem`, `cpus-per-task`, and `time` to help your jobs queue faster depending on your needs.
   - SMU uses coldfront for allocation management requireing an `--account` flag in the `sbatch` files, you may need to remove or change this line.
3. Point the `generator.sh` script to the correct settings files by changing `SETTINGS`, and update any CLI overrides you want in the `ARGS`.

You can then run the script.

> This script will use slurm's run chaining to queue up runs but only start them once a previous run has been completed.

#### Pipeline Overview

The data generation pipeline consists of three stages:

1. **Initialization** (Initializor): Pre-computes KSW Monte Carlo states for each bispectrum shape using MPI. This step is required when using the automated pipeline but can be skipped for quick manual runs.
   ```sh
   # For automated pipeline (MPI required)
   sbatch sbatch/initializor.sbatch settings/planck.json --lensing
   ```

2. **Generation** (Generator): Generates Gaussian and non-Gaussian CMB alms, computes Fisher information matrices, and optionally computes KSW estimates inline during generation. Uses Slurm array jobs for parallel per-array generation.
   ```sh
   sbatch --array=1-N sbatch/generator.sbatch settings/planck.json --lensing
   ```

3. **Combining** (Combiner): Merges per-array outputs into a single consolidated HDF5 file.
   ```sh
   sbatch sbatch/combiner.sbatch settings/planck.json --lensing
   ```

#### Manual Data Generation

The data generator is controlled by settings files located in the `settings/` directory. You can specify a different settings file as an argument when running the Python scripts. For some arguments, a command line option is available which will take priority. All options have defaults. See `settings/settings.md` for more.

To manually generate data without the initialization step (serial mode, no KSW estimates in MPI):

```sh
python -m mlpng.generator --lensing settings/planck.json
```

After data generation, if using `narray > 1`, combine the per-array output files into a single file:

```sh
python -m mlpng.combiner --lensing settings/planck.json
```

> **Note**: By default, the generator computes KSW estimates inline (controlled by the `--estimate` flag). In manual mode, estimates are computed on the main process only. For distributed MPI-based estimation, use the full pipeline with initializor.

### Training

The training pipeline is very simple. From Superpod,

0. Ensure your data is fully generated and available on the Superpod filesystem.
1. Update the `sbatch/trainer.sbatch` file to match the cluster settings
2. Create your model
   - See `mlpng/scn-jorik.py` for a standalone example.
3. Point the `trainer.sh` script to the correct settings for the data you are using and to the models.

> [Weights and Biases](https://wandb.ai/) can be used for logging and tracking the training, similar to an online tensorboard with a few extra features. You will need to set up an account and install the `wandb` package to use this feature and then enable it in the run settings.

## Some Notes

### Notes on data format

The data is stored in `hdf5` files as they allow reading and appending data without needing to load the whole dataset into memory. You can think of these files as Python dictionaries. Some important data values are:

**Gaussian alms:**
- `alm_l/unlensed` : The Gaussian $a_{\ell m}$ values (unlensed). `Shape: (nsims, npol, nelem)`
- `alm_l/lensed` : The Gaussian $a_{\ell m}$ values (if lensing enabled). `Shape: (nsims, npol, nelem)`

**Non-Gaussian alms (per bispectrum shape):**
- `alm_nl/{unlensed|lensed}/{local|equilateral|orthogonal}` : The non-Gaussian contributions. `Shape: (nsims, npol, nelem)`

**Fisher information and estimates (only if `--estimate` flag used):**
- `fisher/{unlensed|lensed}/{shape}` : Fisher information values for each shape.
- `estimates/{unlensed|lensed}/{shape}` : Estimated $f_{nl}$ values (shape: `(n_estimates,)`)
- `fisher_matrix/{...}` : Full Fisher matrices for multi-parameter estimation
- `marginal_likelihoods/{...}` : Marginal likelihood information

**Lensing (if enabled):**
- `phi_map` : The lensing potential map
- Associated keys may have `_lensed` suffix variants

**Other optional keys:**
- `error`, `error_lensed` : Estimation errors
- `fnl_norm` : Normalized $f_{nl}$ values

> **Note**: Estimate-related keys (`estimates`, `fisher_matrix`, `marginal_likelihoods`) are only present if the generator was run with the `--estimate` flag (default: True).

### Notes on files

Simultaneous runs are supported as long as the filenames do not collide. If files are found to exists the scripts are designed to exit quickly but will not fail. This allows easy continuation of incomplete runs, i.e. if some of the sims timed out everything is designed so you can just change the time in the slurm settings and rerunning should then only generate the missing sims. This will cause issues if two runs are started at the same time with the same settings file, I do not plan to handel that case.

#### Filenames

Filenames are generated from select settings for easy reading once you understand the format. It is possible to provide a `base_name` in the settings file to override the default naming scheme. The default naming scheme is as follows:

For alms: `l[lmax]_n[nside]_[polarizations]_[total sim]x[ndup]_f[fnl range]`

> See `scripts/core.py:_paths` for the where this gets set

## Configuration and CLI Flags

Both the settings JSON files and command-line arguments control the pipeline behavior. CLI arguments override JSON settings. Key flags include:

**Estimation (KSW):**
- `--estimate/--no-estimate` : Enable or disable inline KSW estimate computation during generation (default: True)
- `--num_estimates NUM` : Number of samples to estimate (default: min(nsims × narray, 1000))
- `--mc_steps STEPS` : Number of Monte Carlo steps for KSW initialization (default: 300)

**Data Generation:**
- `--force_generation` : Force regeneration, overwriting existing files
- `--shapes {local|equilateral|orthogonal|all}` : Specify which bispectrum shapes to generate (can specify multiple)
- `--lensing/--no-lensing` : Include or exclude gravitational lensing
- `--noise/--no-noise` : Include or exclude instrumental noise
- `--double_precision` : Use double precision (float64/complex128) instead of single precision

**Data I/O:**
- `--save_alms` : Save combined alms to disk (otherwise reconstructed on-the-fly during training)
- `--save_ksw` : Save KSW MC states to disk

**Slurm and Execution:**
- `--narray N` : Number of array job tasks (must equal SLURM_ARRAY_TASK_COUNT when running under Slurm)
- `--nsims N` : Number of simulations per array task

**Other:**
- `--phi_scale SCALE` : Scaling factor for the lensing potential (default: 1.0)
- `--plot/--no-plot` : Enable or disable plot generation
- `--seed SEED` : Random seed for reproducibility

See `settings/settings.md` for the complete list of configurable parameters.

## Thanks

Primary development by:
   Brandon Stevenson, Joe Ryan, Joel Meyers

Additional thanks to:

- Daan Meerburg
- Jorik Melsen
- Thomas Flöss
