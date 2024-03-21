import glob
import os
import re
import logging

import h5py
from tqdm.auto import tqdm
from utils import Config, setup_logging

logger = setup_logging(__name__)
logger.setLevel(logging.DEBUG)


def extract_number(filename):
    # Extracts the number from a filename
    matches = re.findall(r"\d+", filename)
    logger.debug('%s matches %s', filename, matches)
    return int(matches[-2]) if matches else 0


def recursive_copy(hf_source, hf_dest):
    for key in hf_source.keys():
        logger.debug("Copying %s", key)
        if isinstance(hf_source[key], h5py.Group):
            logger.debug("Found group %s", key)
            if key not in hf_dest:
                logger.debug("Creating group %s", key)
                hf_dest.create_group(key)
            recursive_copy(hf_source[key], hf_dest[key])
        else:
            if len(hf_source[key].shape) > 0:  # Non-scalar dataset
                logger.debug("Found non-scalar dataset %s", key)
                if key in hf_dest:
                    logger.debug("Resizing dataset %s", key)
                    hf_dest[key].resize(
                        (hf_dest[key].shape[0] + hf_source[key].shape[0],)
                        + hf_source[key].shape[1:]
                    )
                    hf_dest[key][-hf_source[key].shape[0] :] = hf_source[key]
                else:
                    logger.debug("Creating dataset %s", key)
                    hf_dest.create_dataset(
                        key,
                        data=hf_source[key],
                        maxshape=(None,) + hf_source[key].shape[1:],
                    )
            else:  # Scalar dataset
                if key in hf_dest:
                    logger.debug("Appending scalar values to dataset %s", key)
                    # Concatenate scalar values into a 1D dataset
                    hf_dest[key].resize((hf_dest[key].shape[0] + 1,))
                    hf_dest[key][-1:] = hf_source[key][()]
                else:
                    logger.debug("Creating new 1D dataset for scalar values %s", key)
                    # Create a new 1D dataset for scalar values
                    hf_dest.create_dataset(
                        key, data=[hf_source[key][()]], maxshape=(None,)
                    )


def combine_data(directory, base_name, ext, remove_files=True):
    # Get a list of all h5py files that match the pattern, i.e., end with a SLURM job array index
    file_pattern = os.path.join(directory, f"{base_name}_[0-9]*{ext}*")
    files_to_combine = glob.glob(file_pattern)
    logger.debug("Found files to combine: %s", files_to_combine)

    # combine in order, might not be needed
    files_to_combine = sorted(files_to_combine, key=extract_number)

    if len(files_to_combine) == 0:
        logger.error("did not find any files to combine with %s", file_pattern)
        return

    if len(files_to_combine) != s.narray:
        logger.error(
            "Found %d files to combine, expected %d", len(files_to_combine), s.narray
        )
        return

    # Create a new h5py file to hold all the combined data
    with h5py.File(os.path.join(directory, f"{base_name}{ext}.nc"), "w") as hf_combined:
        for file in tqdm(files_to_combine, desc="processing files"):
            with h5py.File(file, "r") as hf:
                logger.debug("Processing file %s", file)
                recursive_copy(hf, hf_combined)

    # Rename the combined file to remove the .nc extension
    os.replace(
        os.path.join(directory, base_name + ext + ".nc"),
        os.path.join(directory, base_name + ext),
    )

    if remove_files:
        for file in tqdm(files_to_combine, desc="removing partial files"):
            os.remove(file)


if __name__ == "__main__":
    s = Config()

    combine_data(s.alm_dir, s.base_name, ".hdf5")
    combine_data(s.patch_dir, s.patch_str, ".hdf5")
