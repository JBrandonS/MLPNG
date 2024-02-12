#!/bin/bash

# used to convert completed data to tdfs
# trainer is used as a general runner
# alm or hdf5 can be used as the data source
sbatch "sbatch/trainer.sbatch" "scripts/data_to_tfds.py" "alm"
