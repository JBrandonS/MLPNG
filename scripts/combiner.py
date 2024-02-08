import os
import sys
import glob
import h5py
import re
from tqdm.auto import tqdm

from utils import SimConfig

import logging

def extract_number(filename):
    # Extracts the number from a filename
    matches = re.findall(r"\d+", filename)
    if matches:
        return int(matches[-2])  # Consider the last numerical value for ordering
    else:
        return None

def combine_data(directory, base_name, ext, remove_files=True, finalize=False):
    # Get a list of all h5py files that match the pattern, i.e., end with a SLURM job array index
    file_pattern = os.path.join(directory, base_name+"_[0-9]*"+ext+"*")
    files_to_combine = glob.glob(file_pattern)
    files_to_combine = sorted(files_to_combine, key=extract_number)

    if len(files_to_combine) == 0:
        log.error("did not find any files to combine with %s", file_pattern)
        return

    def recursive_copy(hf_source, hf_dest):
        # print(f"Found keys {hf_source.keys()} to add to {hf_dest.keys()}")
        for key in hf_source.keys():
            if isinstance(hf_source[key], h5py.Group):
                if key not in hf_dest:
                    hf_dest.create_group(key)
                recursive_copy(hf_source[key], hf_dest[key])
            else:
                if len(hf_source[key].shape) > 0:  # Non-scalar dataset
                    if key in hf_dest:
                        hf_dest[key].resize((hf_dest[key].shape[0] + hf_source[key].shape[0],) + hf_source[key].shape[1:])
                        hf_dest[key][-hf_source[key].shape[0]:] = hf_source[key]
                    else:
                        hf_dest.create_dataset(key, data=hf_source[key], maxshape=(None,) + hf_source[key].shape[1:])
                else:  # Scalar dataset
                    if key in hf_dest:
                        # Concatenate scalar values into a 1D dataset
                        hf_dest[key].resize((hf_dest[key].shape[0] + 1,))
                        hf_dest[key][-1:] = hf_source[key][()]
                    else:
                        # Create a new 1D dataset for scalar values
                        hf_dest.create_dataset(key, data=[hf_source[key][()]], maxshape=(None,))

    # Create a new h5py file to hold all the combined data
    with h5py.File(os.path.join(directory, base_name + ext + '.nc'), 'w') as hf_combined:
        for file in tqdm(files_to_combine, desc='processing files'):
            with h5py.File(file, 'r') as hf:
                recursive_copy(hf, hf_combined)

        log.debug("Final keys in file %s are %s", base_name + ext, hf_combined.keys())

    if finalize:
        os.replace(os.path.join(directory, base_name + ext + '.nc'), 
                   os.path.join(directory, base_name + ext))

    if remove_files:
        for file in tqdm(files_to_combine, desc='removing partial files'):
            os.remove(file)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%d-%b-%y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    log = logging.getLogger(__name__)
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'settings/settings.json'
    s = SimConfig(config_file)

    # check that we are not using a already completed file!
    if not os.path.isfile(s.alm_file_complete):
        combine_data(s.alm_cache_dir, s.alm_str, '.alms.hdf5', finalize=True)

    combine_data(s.data_dir, s.data_str, '.hdf5')
