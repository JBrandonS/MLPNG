# Machine Learning for Primordial Non-Gaussianity

## Overview

## Tasks

## Installing

 - Clone repo with `git clone`
 - Clone submodules with `git submodule update --init --recursive`
    - You can also just download and install `KSW` and `optweight`.
 - You can see `ksw_datagen.sbatch` for an example of running the sciprt on m3
    - You will need to change the conda env in the sbatch scripts

- Currently datagen will work on m3 but not superpod. Training will be done on superpod but not working right now.

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
