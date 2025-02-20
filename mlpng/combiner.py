import sys
import glob
import os
import re
import logging
import h5py
from tqdm.auto import tqdm

from . import Core
from .utils import setup_logging

logger = setup_logging("mlpng.combiner", logging.INFO)


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


def recursive_copy(hf_source, hf_dest, num_files, index):
    """
    Recursively copies datasets from `hf_source` to `hf_dest` with specified index range.

    Args:
        hf_source (h5py.File): The source HDF5 file object.
        hf_dest (h5py.File): The destination HDF5 file object.
        num_files (int): The total number of files to be combined, used to inform dataset generation.
        index (int): The index of the current file being copied.

    Returns:
        None
    """
    for key in hf_source.keys():
        if isinstance(hf_source[key], h5py.Group):
            # If the key is a group, create the group in the destination file and recurse
            if key not in hf_dest:
                hf_dest.create_group(key)
            recursive_copy(hf_source[key], hf_dest[key], num_files, index)
        elif isinstance(hf_source[key], h5py.Dataset):
            # If the key is a dataset, copy the data
            num_elem = hf_source[key].shape[0]
            data_shape = hf_source[key].shape[1:]
            logger.debug("%s: Found non-scalar dataset of shape: %s", key, data_shape)

            if key not in hf_dest:
                ds_shape = (num_files * num_elem,) + data_shape
                dtype = hf_source[key].dtype

                logger.debug(
                    "%s: Creating dataset with shape %s, and dtype %s",
                    key,
                    ds_shape,
                    dtype,
                )
                hf_dest.create_dataset(key, shape=ds_shape, dtype=dtype)

            logger.debug(
                "%s: Copying data to index range %s",
                key,
                (index * num_elem, (index + 1) * num_elem),
            )
            logger.debug("%s: Data type of hf_source: %s", key, hf_source[key].dtype)
            logger.debug("%s: Data type of hf_dest: %s", key, hf_dest[key].dtype)
            hf_dest[key][index * num_elem : (index + 1) * num_elem] = hf_source[key]


def combine_data(directory, base_name, ext=".hdf5", remove_files=True, expected=100):
    """
    Combine multiple h5py files into a single file.

    Args:
        directory (str): The directory where the files are located.
        base_name (str): The base name of the files to combine.
        ext (str, optional): The file extension of the files to combine. Defaults to ".hdf5".
        remove_files (bool, optional): Whether to remove the partial files after combining. Defaults to True.
        expected (int, optional): The expected number of files to combine. Defaults to 100.

    Returns:
        None

    Raises:
        None
    """
    # Get a list of all h5py files that match the pattern, i.e., end with a slurm job array index
    file_pattern = os.path.join(directory, f"{base_name}_[0-9]*{ext}")
    files_to_combine = glob.glob(file_pattern)

    if len(files_to_combine) == 0:
        logger.info(
            "Did not find any files to combine with %s."
            "This is expected if some of the data has already been combined.",
            file_pattern,
        )
        return  # We do not exit(1) here since we want to keep going with other jobs

    if len(files_to_combine) != expected:
        logger.error(
            "Found %d files, expected %d for pattern %s",
            len(files_to_combine),
            expected,
            file_pattern,
        )
        sys.exit(1)

    # check for existing incomplete combined file and remove it
    # this should only happen if the combiner failed, i.e. timed out
    nc_file = os.path.join(directory, base_name + ext + ".nc")
    if os.path.exists(nc_file):
        os.remove(nc_file)

    # sort by the index number so that we combine in order, not needed but doing anyways
    files_to_combine = sorted(files_to_combine, key=extract_number)

    # Create a new h5py file to hold all the combined data
    logger.info("Saving combined data to %s", nc_file)
    with h5py.File(nc_file, "x") as hf_combined:
        for idx, file in tqdm(
            enumerate(files_to_combine), desc="Processing files", total=expected
        ):
            logger.info("Processing file %s", file)
            with h5py.File(file, "r") as hf:
                recursive_copy(hf, hf_combined, expected, idx)

    # Move the combined file to remove the .nc extension
    # This will also override any existing file with the same name
    logger.info("Moving combined file to %s", os.path.join(directory, base_name + ext))
    os.replace(
        nc_file,
        os.path.join(directory, base_name + ext),
    )

    if remove_files:
        for file in tqdm(files_to_combine, desc="Removing partial files"):
            os.remove(file)


if __name__ == "__main__":
    core = Core()
    combine_data(core.dirs["data"], core.name, ".hdf5", expected=core.narray)
