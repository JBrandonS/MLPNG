# Machine Learning for Primordial Non-Gaussianity

[Google Doc](https://docs.google.com/document/d/1ogvANs4A7Ddb9--W0YCcQKWKaQXbMvZDO7PQbGw4Vsc/edit?usp=sharing)

## Overview
---

## Tasks
---

## Installing
---

To install this project, follow these steps:
### datagen

#### Git
1. Clone the repository using `git clone`
2. Clone the submodules using `git submodule update --init --recursive`
3. Make sure your environment is setup correctly, currently there is an issue with the FFTW module that needs some export.
   - Look at `datagen.sbatch` for an example of how to set this up, use the same modules and the export commands
4. Create and activate a python env with the requirements listed in the READMEs of the submodules.
   - I have provided `datagen-m3.yml` which is the conda env I use on m3.
5. Install the submodules using the READMEs in each submodule directory.
   - `optweight` should be installed first, just need run `pip install -e .` in the root of the submodule.
   - For `ksw` run `make && pip install -e . && make check` in the root of the submodule.
6. Everything should now be set up and ready to use.

#### ZIP

1. Download the zip files for this project, optweight, and KSW from github
   - You can extract these files anywhere, they do not need to be in the MLPNG folder
2. Make sure your environment is setup correctly, currently there is an issue with the FFTW module that needs some export.
   - Look at `datagen.sbatch` for an example of how to set this up, use the same modules and the export commands
3. Create and activate a python env with the requirements listed in the READMEs of the submodules.
   - I have provided `datagen-m3.yml` which is the conda env I use on m3.
4. Install the submodules using the READMEs in each directory.
   - `optweight` should be installed first, just need run `pip install .` in the root of the submodule.
   - For `ksw` run `make && pip install . && make check` in the root of the submodule.
5. Everything should now be set up and ready to use. 

### trainer
TODO

## Running
---

> You will need to change the conda env in the sbatch scripts
>
> There is also currently an issue with how the FFTW module is loaded.
> IT knows about this but for now see the sbatch script to see how to update your settings so the code can find the FFTW library

I am working mostly in the Jupyter notebooks, I export these to the .py files but I might forget. The files `datagen.ipynb` and `trainer.ipynb` are the main files to look at, I have tried to document them but let me know if you see anything wrong.


### Data Generation

One you have the env setup you should be able to run the notebook. You might need to create a `data/` dir, I would recommend using soft linking to $SCRATCH or $WORK. The notebook will generate the data and save it to the `data/` dir.

You can also use the `datagen.sbatch` to launch a slurm job. You **will** need to make a `logs` dir if you use this as is, or change the `output` and `error` flags to point to an existing folder. This will run the `datagen.py` script which is just the notebook exported to a .py file, it may or may not have different settings to the notebook so you will want to check everything is good, and probably want to increase the `nsims` value at least. For large datasets you might want to change `save_fullsky` to save data if you are not using the fullsky maps. You will need to change the conda env in the sbatch script. You will probably want to change the debug flag to false since this just creates plots without saving to  test the output.

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

I have put a copy (taken from KSW) in the dir.
