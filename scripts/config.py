import json
import os

import healpy as hp
import numpy as np
from astropy import units as u

import pprint

import logging
log = logging.getLogger(__name__)

def safe_makedirs(dir):
    "Create a directory if it does not exist. Handles a race condition"
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
            log.debug("Created directory %s", dir)
        except FileExistsError:
            pass

class SimConfig:
    def __init__(self, settings_file, print_settings=True):
        with open(settings_file, "r") as f:
            self.settings = settings = json.load(f)

        if print_settings:
            log.info("Loaded settings from file: %s", settings_file)
            pp = pprint.PrettyPrinter(indent=2)
            pp.pprint(settings)

        # debug settings
        self.debug = settings.get("debug", False)
        self.verbose = settings.get("verbose", False)
        self.save_fullsky = settings.get("save_fullsky", False)
        self.force_alm_gen = settings.get("force_alm_gen", False)

        self.name = settings.get("name", "default")
        self.cosmo_params = settings.get("cosmo_params")

        # main parameters
        self.lmax = self.cosmo_params["lmax"]
        self.polarizations = settings.get("polarizations", "T")
        self.nside = settings.get("nside", 1024)
        self.npatches = settings.get("npatches", 10)
        self.narray = settings.get("narray", 1)
        self.npol = len(self.polarizations)

        self.lensing = settings.get("lensing", False)
        self.disable_noise = settings.get("disable_noise", True)

        # paths
        self.base_dir = settings.get("base_dir", "data")
        self.alm_cache_dir = settings.get("alm_cache_dir", "data/alm_cache")
        self.plot_dir = settings.get("plot_dir", "data/plots")
        self.data_dir = os.path.join(
            self.base_dir, "lensed" if self.lensing else "unlensed"
        )
        self.tb_dir = settings.get("tb_dir", "data/tb")
        self.model_dir = settings.get("model_dir", "data/models")

        # find out the number of sims
        self.nsims = settings.get("nsims", 1)
        self.ndup = settings.get("duplicate_backgrounds", 1)
        self.total_sims = self.nsims * self.ndup

        self.job_array_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if self.job_array_index is not None:
            self.job_array_index = int(self.job_array_index)
            njobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT"))  # type: ignore
            self.job_array_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN"))  # type: ignore
            self.job_array_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX"))  # type: ignore
            self.total_sims *= njobs
            log.info("Running with SLURM job array index %s", self.job_array_index)
        else:
            self.total_sims *= self.narray

        # some important derived parameters
        self.nell = self.lmax + 1
        self.nelem = hp.Alm.getsize(self.lmax)
        self.npix = hp.nside2npix(self.nside)
        self.ells = np.arange(self.nell)
        self.chars_of_polarizations = "".join(self.polarizations)
        self.fnl_min, self.fnl_max = settings.get("fnl_range", [0, 0])

        self.nthreads_sim = settings["nthreads_sim"]
        self.nthreads_alm = settings["nthreads_alm"]
        # set up units
        self.double_precision = self.settings.get("double_precision", False)
        if self.double_precision:
            self.r_dtype = np.float64
            self.c_dtype = np.complex128
        else:
            self.r_dtype = np.float32
            self.c_dtype = np.complex64

        # get the beam and noise
        self.beam_width = settings.get("beam_width", 1) * u.arcmin
        self.noise_scale_tt = settings.get("noise_scale_tt", 1) * u.arcmin
        self.noise_scale_ee = settings.get("noise_scale_ee", 1) * u.arcmin
        self.noise_scale_te = settings.get("noise_scale_te", 1) * u.arcmin
        self.noise_beam = self.get_noise_beam()

        # set up the file names
        nn_str = "nn_" if self.disable_noise else ""
        ja_str = (
            f"_{self.job_array_index}" if self.job_array_index is not None else ""
        )
        self.base_name = (
            f"{self.nside}_{nn_str}{self.chars_of_polarizations}_{self.total_sims}"
        )
        self.data_str = f'{self.base_name}x{self.npatches}x{self.ndup}_fnl{self.fnl_min}-{self.fnl_max}{ja_str}'
        self.alm_str = f"{self.base_name}{ja_str}"

        # setup the file paths
        self.data_file_nc = os.path.join(self.data_dir, f"{self.data_str}.hdf5.nc")
        self.data_file_complete = os.path.join(self.data_dir, f"{self.data_str}.hdf5")
        self.alm_file_nc = os.path.join(
            self.alm_cache_dir, f"{self.alm_str}.alms.hdf5.nc"
        )
        self.alm_file_partial = os.path.join(
            self.alm_cache_dir, f"{self.alm_str}.alms.hdf5"
        )
        self.alm_file_complete = os.path.join(
            self.alm_cache_dir, f"{self.base_name}.alms.hdf5"
        )

        for s in [self.data_dir, self.plot_dir, self.alm_cache_dir]:
            safe_makedirs(s)

    def get_noise_beam(self):
        beam_ell_pre = hp.gauss_beam(
            self.beam_width.to_value(u.radian), lmax=self.lmax, pol=True
        )
        beam_ell_pre = np.swapaxes(beam_ell_pre, 0, 1)

        noise_ell = []
        beam_ell = []
        if "T" in self.polarizations:
            noise = (
                np.ones((self.nell), dtype=self.r_dtype)
                * self.noise_scale_tt.to_value(u.radian) ** 2
            )
            noise_ell.append(noise)
            beam_ell.append(beam_ell_pre[0])

        if "E" in self.polarizations:
            noise = (
                np.ones((self.nell), dtype=self.r_dtype)
                * self.noise_scale_ee.to_value(u.radian) ** 2
            )
            noise_ell.append(noise)
            beam_ell.append(beam_ell_pre[1])

        if self.polarizations == ["T", "E"]:
            noise = (
                np.ones((self.nell), dtype=self.r_dtype)
                * self.noise_scale_te.to_value(u.radian) ** 2
            )
            noise_ell.append(noise)

        noise_ell = np.array(noise_ell).squeeze()
        beam_ell = np.array(beam_ell).squeeze()

        if self.disable_noise:
            noise_ell = noise_ell * 10**-12
            beam_ell = np.ones_like(beam_ell, dtype=self.r_dtype)

        return noise_ell, beam_ell
