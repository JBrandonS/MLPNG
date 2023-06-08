# Machine Learning for Primordial Non-Gaussianity

## Overview
---

## Tasks
---

## Installing
---

To install this project, follow these steps:

1. Clone the repository using `git clone` or download the zip file.
   - If you choose to download the zip file, extract the contents to a local directory.
2. Clone the submodules using `git submodule update --init --recursive` or download and install `KSW` and `optweight` if you prefer not to use git submodules.
3. See `datagen.sbatch` for an example of running the script on m3 and which modules to load.
4. Everything should now be set up and ready to use.

> Important
>
> You will need to change the conda env in the sbatch scripts
>
> There is also currently an issue with how the FFTW module is loaded.
> IT knows about this but for now see the sbatch script to see how to update your settings so the code can find the FFTW library

## Running
---

### Data Generation

### Training

### Post Training

## Thanks
---

Thomas and rest of dutch group (need to add the full groups here)

https://github.com/AdriJD/ksw
https://github.com/AdriJD/optweight

## License
---

Since KSW and optweight are GPL we will need to GPL this code. I have put a copy (taken from KSW) in the dir.
