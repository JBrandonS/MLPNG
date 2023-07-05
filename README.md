# Machine Learning for Primordial Non-Gaussianity

[Google Doc](https://docs.google.com/document/d/1ogvANs4A7Ddb9--W0YCcQKWKaQXbMvZDO7PQbGw4Vsc/edit?usp=sharing)

## Overview

## Tasks

## Installing

To install this project, follow these steps:

### datagen

1. Clone the repository using `git clone`, or download and extract the zip
2. Create a python environment
   - I have provided `datagen-m3.yml` which is the conda env I use on m3.
3. Clone, or download, and install [KSW](https://github.com/AdriJD/ksw) and [optweight](https://github.com/AdriJD/optweight)
   - `optweight` should be installed first, just need run `pip install -e .` in the root.
   - For `ksw` run `make && pip install -e . && make check` in the root.
      - You will probably see an error on the make check, this seems to be an issue with the test code.
4. You can now test the code, using the `datagen.ipynb` notebook with 1 sim should take about 10 minutes if it needs to generate the $a_{\ell m}$.

### trainer

TODO

## Running

> You will need to change the conda env in the sbatch scripts

The best method to run large amounts data is to use slurm job arrays. The Jupyter notebook is good for small sims but the risk of disconnecting, the large runtime, and the added benefit of being able to run multiple sims at once without risk of running out of memory 10 hours in makes slurm job arrays the best option.

1. Set a low number of sims (10) within the `datagen.py` and change any settings in this file that you want.
2. Modify the `datagen.sbatch` script to point to the correct conda env, provide enough ram, about 500G for 10 sims, and change any other settings you want.
3. You can then run the generator with `sbatch datagen.sbatch`.
4. Once done, you can run `sbatch combiner.sbatch` which will combine the partial data files into one and remove the partial files.
   - You will want to modify the `combiner.py` file to have the correct `base_name`, you will also need to make sure that it is pointing to the correct `data_dir` for lensed or unlensed data to combine the data.
   - I only use sbatch for this due to internet issues possibly causing problems with how I run jupyter notebooks, but you can run the `combiner.py` file in a notebook if you want.
5. You should be good now to run the training.

## Data Generation

One you have the env setup you should be able to run the notebook. You might need to create a `data/` dir, I would recommend using soft linking to $SCRATCH or $WORK. The notebook will generate the data and save it to the `data/` dir.

You can also use the `datagen.sbatch` to launch a slurm job. You **will** need to make a `logs` dir if you use this as is, or change the `output` and `error` flags to point to an existing folder. This will run the `datagen.py` script which is just the notebook exported to a .py file, it may or may not have different settings to the notebook so you will want to check everything is good. For large datasets you will want to take advantage of slurm's job arrays.

### Training

### Post Training

## Thanks

Thomas and rest of dutch group (need to add the full groups here)

Adri Duivenvoorden:

- [Minimizing gravitational lensing contributions to the primordial bispectrum covariance](http://arxiv.org/abs/1912.07619)
- [Primordial Non-Gaussianity](http://arxiv.org/abs/1903.04409)
- [ksw github](https://github.com/AdriJD/ksw)
- [optweight github](https://github.com/AdriJD/optweight)
