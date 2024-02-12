#!/bin/bash

# used to convert completed data to tdfs
# trainer is used as a general runner
# alm or hdf5 can be used as the data source
# alm-split can be used to generae alm data that is split into real, imag along last dimension
sbatch "sbatch/trainer.sbatch" "scripts/data_to_tfds.py" "alm-split"
