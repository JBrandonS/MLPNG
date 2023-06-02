# Machine Learning for Primordial Non-Gaussianity

## Overview

## Tasks

- Curved sky maps
    - Regularization
        - [x] More training data
        - [x] Reduce model complexity [**Joe found, on 03/02/2023, that a neural network with one convolution layer and one dense layer was not able to learn the features of the training, validation, or test sets (CNN-reconstruction.ipynb.**]
        - [x] Dropout rate [**Joe found, on 03/02/2023, that increasing the dropout rate to 0.5 (from 0.3) produced no change in the network’s ability to learn the features of the validation or test sets. Increasing the dropout rate to 0.99 also produced no change, when the network was trained over 10 epochs.**]
        - [ ] L1 and l2 regularization
    - [x] Generate maps with larger range of $f_{NL}$ [**Joe experimented with this between 02/28/23-03/01/23, and found no change to the network’s ability to learn the features of the validation or test sets.**]
    - [ ] Resolution / map size / tiling
    - [ ] Smoothing of maps

- CMB transfer function applied to flat sky maps
    - Linear transfer function should not impact results - is that true for CMB maps?

- Bias versus variance in Thomas tests
    - [x] Larger data set
    - [ ] Hyperparameters, early stopping, etc.
    - [ ] Model complexity

- $f_{NL}$ error estimates as function of:
    - [ ] $f_{sky}$
    - [ ] $\ell_{min}$
    - [ ] $\ell_{max}$
    - [ ] Noise level
    - [ ] Temperature, polarization, and both
    - [ ] With and without lensing

- [ ] Lensing applied to maps

- Generate our own non-Gaussian maps
    - [x] Local $f_{NL}$
    - [ ] Curved sky
    - [x] Flat sky

## Installing

 - Clone repo with `git clone --recursive`
 - Install ksw and optweight
 - ...

## Running

Please see libs to see the required packages and versions.

### Data Generation

Once the directories are set you should be good run `dataGenerator.ipynb` to test. 

If you want to run in SLURM you can use `sbatch dataGenerator.sbatch`, this will convert the `dataGenerator.ipynb` to a script and run it. You might need to change some settings in the sbatch file which is configured to run on the SuperPOD.

This process is currently slow, taking around 15 seconds per map, so it will take a while to generate a large dataset. I am working on a way to speed this up. For now, consider only using the `dataGenerator.ipynb` notebook to generate a small dataset for testing. Run the sbatch longer datasets, for datasets over a million we will probably need to rework the save code to allow multiple nodes to run in parallel.

### Training

### Post Training
TODO ???

## References
Should we add them here? Maybe useless

## Thanks
Thomas and rest of dutch group (need to add the full groups here)

https://github.com/AdriJD/ksw
https://github.com/AdriJD/optweight
