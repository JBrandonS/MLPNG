"""Combiner module for merging HDF5 batch outputs into a single file."""

import glob
import logging
import os
import re
import sys

import h5py
import numpy as np
from tqdm.auto import tqdm

from . import Core
from .utils import setup_logging

# Module-level logger following Python best practices
_logger = logging.getLogger(__name__)


def extract_number(filename: str) -> int:
    """
    Extract the second-to-last number from a filename.

    Args:
        filename: The name of the file.

    Returns:
        The second-to-last number found in the filename, or 0 if no numbers are found.
    """
    matches = re.findall(r"\d+", filename)
    return int(matches[-2]) if matches else 0


def recursive_copy(
    hf_source: h5py.File | h5py.Group,
    hf_dest: h5py.File | h5py.Group,
    key_positions: dict,
    path: str = "",
) -> None:
    """
    Recursively copy datasets from source to destination HDF5 file, appending data.

    Args:
        hf_source: The source HDF5 file/group object.
        hf_dest: The destination HDF5 file/group object.
        key_positions: Tracks the current write position for each key (by full path).
        path: Current path in the HDF5 hierarchy.
    """
    for key in hf_source.keys():
        full_key = f"{path}/{key}" if path else key
        if isinstance(hf_source[key], h5py.Group):
            if key not in hf_dest:
                hf_dest.create_group(key)
            _logger.debug("Copying group '%s' at path: %s", key, full_key)
            recursive_copy(hf_source[key], hf_dest[key], key_positions, full_key)
        elif isinstance(hf_source[key], h5py.Dataset):
            data = hf_source[key][...]
            _logger.debug(
                "Copying dataset '%s' shape=%s dtype=%s path=%s",
                key,
                data.shape,
                data.dtype,
                full_key,
            )
            num_elem = data.shape[0]
            data_shape = data.shape[1:]
            dtype = data.dtype

            if key not in hf_dest:
                ds_shape = (0,) + data_shape
                hf_dest.create_dataset(
                    key,
                    shape=ds_shape,
                    maxshape=(None,) + data_shape,
                    dtype=dtype,
                )
                key_positions[full_key] = 0
            if full_key not in key_positions:
                key_positions[full_key] = 0
            pos = key_positions[full_key]
            hf_dest[key].resize((pos + num_elem,) + data_shape)
            hf_dest[key][pos : pos + num_elem] = data
            key_positions[full_key] += num_elem


def combine_fisher_values(hf: h5py.File) -> None:
    """
    Combine Fisher values from multiple batches into a single averaged value.

    Fisher information computed from the same KSW estimator across different
    data realizations should be averaged to reduce Monte Carlo noise.

    Creates /fisher/combined/{lensed|unlensed}/{shape} datasets.

    Args:
        hf: The HDF5 file object (opened in read/write mode).
    """
    if "fisher" not in hf:
        _logger.warning("No 'fisher' group found in file, skipping Fisher combination")
        return

    fisher_group = hf["fisher"]

    # Create combined group if it doesn't exist
    if "combined" not in fisher_group:
        fisher_group.create_group("combined")
    combined_group = fisher_group["combined"]

    for lensed_key in ["lensed", "unlensed"]:
        if lensed_key not in fisher_group:
            _logger.debug("No '%s' Fisher data found, skipping", lensed_key)
            continue

        lensed_group = fisher_group[lensed_key]

        # Create lensed/unlensed subgroup in combined
        if lensed_key not in combined_group:
            combined_group.create_group(lensed_key)
        combined_lensed = combined_group[lensed_key]

        for shape_key in lensed_group.keys():
            if isinstance(lensed_group[shape_key], h5py.Dataset):
                fisher_values = lensed_group[shape_key][...]
                n_values = len(fisher_values)

                # Compute mean Fisher value (averaging reduces MC noise)
                fisher_combined = np.mean(fisher_values)
                sigma_combined = 1.0 / np.sqrt(fisher_combined)

                # Store as scalar wrapped in atleast_1d for HDF5 compatibility
                if shape_key in combined_lensed:
                    del combined_lensed[shape_key]
                combined_lensed.create_dataset(
                    shape_key, data=np.atleast_1d(fisher_combined)
                )

                _logger.info(
                    "Combined Fisher for %s/%s: %.6f (σ=%.4f) from %d batches",
                    lensed_key,
                    shape_key,
                    fisher_combined,
                    sigma_combined,
                    n_values,
                )


def combine_data(
    directory: str,
    base_name: str,
    ext: str = ".hdf5",
    remove_files: bool = True,
    expected: int = 100,
) -> None:
    """
    Combine multiple HDF5 files into a single file.

    Args:
        directory: The directory where the files are located.
        base_name: The base name of the files to combine.
        ext: The file extension of the files to combine.
        remove_files: Whether to remove the partial files after combining.
        expected: The expected number of files to combine.

    Raises:
        SystemExit: If the number of found files doesn't match expected.
    """
    # Get a list of all HDF5 files that match the pattern (ending with slurm job array index)
    file_pattern = os.path.join(directory, f"{base_name}_[0-9]*{ext}")
    files_to_combine = glob.glob(file_pattern)

    if len(files_to_combine) == 0:
        _logger.info(
            "No files found matching pattern '%s'. "
            "This is expected if data has already been combined.",
            file_pattern,
        )
        return

    if len(files_to_combine) != expected:
        _logger.error(
            "File count mismatch: found %d, expected %d for pattern '%s'",
            len(files_to_combine),
            expected,
            file_pattern,
        )
        sys.exit(1)

    _logger.info(
        "Found %d files to combine matching '%s'", len(files_to_combine), file_pattern
    )

    # Check for existing incomplete combined file and remove it
    # (this can happen if the combiner timed out previously)
    nc_file = os.path.join(directory, base_name + ext + ".nc")
    if os.path.exists(nc_file):
        _logger.warning("Removing incomplete combined file: %s", nc_file)
        os.remove(nc_file)

    # Sort by index number for consistent ordering
    files_to_combine = sorted(files_to_combine, key=extract_number)

    # Create new HDF5 file to hold combined data
    output_file = os.path.join(directory, base_name + ext)
    _logger.info("Combining data into: %s", nc_file)

    key_positions = {}
    with h5py.File(nc_file, "x") as hf_combined:
        for file in tqdm(files_to_combine, desc="Combining files", total=expected):
            _logger.debug("Processing: %s", os.path.basename(file))
            with h5py.File(file, "r") as hf:
                recursive_copy(hf, hf_combined, key_positions, "")

        # Combine Fisher values after all data is merged
        _logger.info("Computing combined Fisher values...")
        combine_fisher_values(hf_combined)

    # Atomically replace output file by renaming
    _logger.debug("Moving combined file to: %s", output_file)
    os.replace(nc_file, output_file)
    _logger.info("Successfully created combined file: %s", output_file)

    if remove_files:
        _logger.info("Removing %d partial files...", len(files_to_combine))
        for file in tqdm(files_to_combine, desc="Removing partial files"):
            os.remove(file)
        _logger.info("Cleanup complete")


if __name__ == "__main__":
    setup_logging(__name__, level=logging.INFO)

    core = Core()

    combine_data(core.dirs["data"], core.name, ".hdf5", expected=core.narray)
