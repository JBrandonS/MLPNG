# %% [markdown]
# This script just combines the partial data files generated when using SLURM job arrays. It is not necessary to run this script if you are not using SLURM job arrays.

# %%
import os
import glob
import h5py
import numpy as np
import re
from tqdm.auto import tqdm

def extract_number(filename):
    # Extracts the number from a filename
    matches = re.findall(r"\d+", filename)
    if matches:
        return int(matches[-2])  # Consider the last numerical value for ordering
    else:
        return None

def combine_data(directory, base_name, ext, remove_files=True):
    # Get a list of all h5py files that match the pattern, i.e., end with a SLURM job array index
    file_pattern = os.path.join(directory, base_name+"_[0-9]*"+ext)
    files_to_combine = glob.glob(file_pattern)
    files_to_combine = sorted(files_to_combine, key=extract_number)

    # Create a new h5py file to hold all the combined data
    with h5py.File(directory + base_name + ext + '.nc', 'w') as hf_combined:
        for file in tqdm(files_to_combine, desc='processing files'):
            with h5py.File(file, 'r') as hf:
                # For each key (dataset) in the file
                for key in hf.keys():
                    # If the dataset already exists in the combined file, append to it
                    if key in hf_combined:
                        hf_combined[key].resize((hf_combined[key].shape[0] + hf[key].shape[0],) + hf[key].shape[1:])
                        hf_combined[key][-hf[key].shape[0]:] = hf[key]
                    else:
                        # Else, copy the entire dataset to the combined file
                        hf_combined.create_dataset(key, data=hf[key], maxshape=(None,) + hf[key].shape[1:], compression="gzip")

    os.replace(directory + base_name + ext + '.nc', directory + base_name + ext)

    if remove_files:
        for file in tqdm(files_to_combine, desc='removing partial files'):
            os.remove(file)

# %%
if __name__ == '__main__':
    base_name = '1024_nn_T_10000'
    data_name = f'{base_name}x10_fnl-1000-1000'

    cache_dir = 'data/ksw/alm_cache/'
    data_dir = 'data/ksw/lensed/'

    combine_data(cache_dir, base_name, '.alms.hdf5')
    combine_data(data_dir, data_name, '.hdf5')
# %%

