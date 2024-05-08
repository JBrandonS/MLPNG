# Code Structure

## Important Files

The most important files for the code at the running scripts `generator.sh`, and `trainer.sh`. Each of these will kick off possibly multiple SLURM jobs using sbatch files found in `sbatch`. 

`scripts/core.py` serves as the main config object and sets up much of the code.

`scripts/models/modelcore.py` is similar as it servers as a base class for the ML models. This file also holds code for registering the model to a global register and loading the model from that register.

`scripts/models/isensee.py` is the model taken from Thomas' UNET_fnls, it also contains an entry point and thus can be used as a standalone script. 

`notebooks/` contains a few jupyter notebooks for quick testing, and can be a good starting point for getting familiar with the code / quick runs.

`sbatch/` contains the SLURM scripts which are used to run the code on the cluster. 