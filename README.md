# Machine Learning for Primordial Non-Gaussianity

## Overview
---

## Tasks
---

## Installing
---

 - Clone repo with `git clone`
   - You can also just download the zip
 - Clone submodules with `git submodule update --init --recursive`
    - You can also just download and install `KSW` and `optweight` and ignore the submodules if you are not a git fan.
 - Install `optweight` and `KSW`
   - You can see `ksw_datagen.sbatch` for an example of running the sciprt on m3, in particular which modules to load
 - Everything should 'just work'

> Important
>
> You will need to change the conda env in the sbatch scripts
>
> There is also currently an issue with how the FFTW module is loaded.
> IT knows about this but for now see the sbatch script to see how to update your settings so the code can find the FFTW library

- Currently datagen will work on m3 but not superpod. Training will be done on superpod but not working right now.

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

Since KSW and optweight are GPL we will need to GPL this code.
