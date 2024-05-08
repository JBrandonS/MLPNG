import sys
import glob
import logging
import os
import re

import h5py
from tqdm.auto import tqdm

from . import Core
from .utils import setup_logging

logger = setup_logging(__name__, level=logging.DEBUG)


def extract_number(filename):
    """
    Extracts the second-to-last number from a filename.

    Args:
        filename (str): The name of the file.

    Returns:
        int: The second-to-last number found in the filename, or 0 if no numbers are found.
    """
    matches = re.findall(r"\d+", filename)
    return int(matches[-2]) if matches else 0


def recursive_copy(hf_source, hf_dest):
    """
    Recursively copies datasets and groups from the source HDF5 file to the destination HDF5 file.

    Args:
        hf_source (h5py.File): The source HDF5 file.
        hf_dest (h5py.File): The destination HDF5 file.

    Returns:
        None
    """
    for key in hf_source.keys():
        if isinstance(hf_source[key], h5py.Group):
            logger.debug("%s: Found group", key)
            if key not in hf_dest:
                logger.debug("%s: Creating group", key)
                hf_dest.create_group(key)
            recursive_copy(hf_source[key], hf_dest[key])
        else:
            if len(hf_source[key].shape) > 0:  # Non-scalar dataset
                logger.debug("%s: Found non-scalar dataset", key)
                if key in hf_dest:
                    logger.debug("%s: Resizing dataset", key)
                    hf_dest[key].resize(
                        (hf_dest[key].shape[0] + hf_source[key].shape[0],)
                        + hf_source[key].shape[1:]
                    )
                    hf_dest[key][-hf_source[key].shape[0] :] = hf_source[key]
                else:
                    logger.debug("%s: Creating dataset", key)
                    hf_dest.create_dataset(
                        key,
                        data=hf_source[key],
                        maxshape=(None,) + hf_source[key].shape[1:],
                    )
            else:  # Scalar dataset
                if key in hf_dest:
                    logger.debug("%s: Appending scalar values to dataset", key)
                    # Concatenate scalar values into a 1D dataset
                    hf_dest[key].resize((hf_dest[key].shape[0] + 1,))
                    hf_dest[key][-1:] = hf_source[key][()]
                else:
                    logger.debug("%s: Creating new 1D dataset for scalar values", key)
                    # Create a new 1D dataset for scalar values
                    hf_dest.create_dataset(
                        key, data=[hf_source[key][()]], maxshape=(None,)
                    )
        logger.debug("%s: Done", key)


def combine_data(directory, base_name, ext=".hdf5", remove_files=True, expected=100):
    # Get a list of all h5py files that match the pattern, i.e., end with a SLURM job array index
    file_pattern = os.path.join(directory, f"{base_name}_[0-9]*{ext}*")
    files_to_combine = glob.glob(file_pattern)
    # combine in order, not needed but doing anyways
    files_to_combine = sorted(files_to_combine, key=extract_number)
    logger.debug("Found files to combine: %s", files_to_combine)

    if len(files_to_combine) == 0:
        logger.error("Did not find any files to combine with %s", file_pattern)
        return  # We do not exit(1) here since we want to keep going with other jobs ie. patchgen

    if len(files_to_combine) != expected:
        logger.error("Found %d files, expected %d", len(files_to_combine), expected)
        return  # We do not exit(1) here since we want to keep going with other jobs ie. patchgen

    # Create a new h5py file to hold all the combined data
    with h5py.File(os.path.join(directory, f"{base_name}{ext}.nc"), "x") as hf_combined:
        for file in tqdm(files_to_combine, desc="Processing files"):
            with h5py.File(file, "r") as hf:
                logger.debug("Processing file %s", file)
                recursive_copy(hf, hf_combined)

    # Move the combined file to remove the .nc extension
    # This will also override any existing file with the same name
    os.replace(
        os.path.join(directory, base_name + ext + ".nc"),
        os.path.join(directory, base_name + ext),
    )

    if remove_files:
        for file in tqdm(files_to_combine, desc="Removing partial files"):
            os.remove(file)


def main():
    core = Core()

    combine_data(core.alm_dir, core.base_name, ".hdf5", expected=core.narray)
    combine_data(core.patch_dir, core.patch_str, ".hdf5", expected=core.narray)


if __name__ == "__main__":
    sys.exit(main())
