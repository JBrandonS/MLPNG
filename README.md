# Machine Learning for Primordial Non-Gaussianity

[Google Doc](https://docs.google.com/document/d/1ogvANs4A7Ddb9--W0YCcQKWKaQXbMvZDO7PQbGw4Vsc/edit?usp=sharing)

## Overview

## Tasks

Use lenspyx

Fix issue with alm cache not using correct files

Duplicate fnls

## Installing

To install this project, follow these steps:

### datagen

1. Clone the repository using `git clone`, or download and extract the zip
2. Create a python environment
   - I have provided `conda-envs/datagen-m3.yml` which is the conda env I use on m3.
3. Clone, or download, and install [KSW](https://github.com/AdriJD/ksw) and [optweight](https://github.com/AdriJD/optweight)
   - `optweight` should be installed first, just need run `pip install -e .` in the root.
   - For `ksw` run `make && pip install -e . && make check` in the root.
      - You will probably see an error on the make check, this seems to be an issue with the test code.
4. You can now run the code, using the `datagenerator.sh`.

### trainer

TODO

## Running

> You will need to change the conda env in the `.sbatch` scripts located inside `datagen/`.
>
> The best method to run large amounts data is to use slurm job arrays.

### Data Generation

The data generator is controlled by settings files located in the `settings/` directory. By default, the `settings/settings.json` file is used. You can specify a different settings file as an argument when running the Python scripts.

The `datagen.py` script generates the alms and patches. 

**Note:** It's recommended to use Slurm job arrays for this script. If you change the value in the `datagen/datagen.sbatch` file, make sure to update the `narray` value in your settings file to match the size of the job arrays. This ensures the `combiner` and `estimator` scripts handle the data correctly.

### Data Combination

After data generation, if you're using job arrays, the data will be in separate files per job. You can use the `combiner.py` script to combine the data into a single file. Simply point it to the correct settings file.

### Data Estimation

The `estimator` script applies the KSW estimator to the data. Point it to the correct settings file. This script also 'finalizes' the data by removing the `.nc` at the end of the file name.

### Simplified Data Generation

For convenience, a `datagenerator.sh` script is provided. To use it:

1. Make any necessary changes to your settings file.
2. Check the sbatch file in `datagen/datagen.sbatch`. You may need to correct it. The other files should be fine, except for the name of the conda environment you're using.
3. Point the `datagen.sh` script to the correct settings file and run it.

You might want to use the command `nohup bash datagenerator.sh &`. This runs the script in the background (`&`) and keeps the process running if your connection drops (`nohup`). This script is safe to run on the log-in nodes as it does no intensive work.

### Training

### Post Training

## Some Notes

### Notes on data format

The data is stored in `hdf5` files as they allow reading and appending data without the need to load the whole dataset into memory. You can think of these files as python dicts. The data is stored in the following format:

- For alms in `data/alm_cache` the following key and values are stored:
  - `alm` : the gaussian $a_{\ell m}$ values in `shape: (nims, len(polarizations), data)`
  - `almng`: the non-gaussian $a_{\ell m}^{NG}$ values in `shape :(nsims, len(polarizations), data)`
    - where the size of `data` depends on the settings used in a complicated way.
  - `settings`: A copy of the settings file used to generate the data for reference.

- For the data files in `data/[un]lensed/`:
  - `estimates`: the KSW estimates of the bispectrum in `shape: (nsims,)`
  - `errors`: the percent diff errors of the estimates vs true fnls in `shape: (nsims,)`
  - `fnls`: the fnl values used to generate the data in `shape: (nsims,)`
    - if you need to align the fnls with the patches use `np.repeat(fnls, npatches)`.
  - `patches`: the patches used to generate the data in `shape: (nsims, len(polarizations), npatches, nside, nside)`
  - `settings`: A copy of the settings file used to generate the data for reference.

- The KSW uses a MC which is saved as `data/[un]lensed/kswmc_*`.
  - These are not important for us but are used to quicken future runs with the same settings (based on filename).

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
