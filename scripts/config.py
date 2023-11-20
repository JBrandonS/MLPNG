import json
import os

import healpy as hp
import numpy as np
from astropy import units as u

def safe_makedirs(dir, verbose=False):
    "Create a directory if it does not exist. Handles a race condition"
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
            if verbose:
                print(f"Created directory {dir}")
        except FileExistsError:
            pass

class SimConfig:
    def __init__(self, settings_file, print_settings=True):
        with open(settings_file, "r") as f:
            self.settings = settings = json.load(f)

        if print_settings:
            print("Loaded settings from file:", settings_file)

            import pprint

            pp = pprint.PrettyPrinter(indent=2)
            pp.pprint(settings)

        self.name = settings.get("name", "default")
        self.cosmo_params = settings.get("cosmo_params")
        self.lmax = self.cosmo_params["lmax"]
        self.polarizations = settings.get("polarizations", "T")
        self.debug = settings.get("debug", False)
        self.verbose = settings.get("verbose", False)
        self.lensing = settings.get("lensing", False)

        self.ndup = settings.get("duplicate_backgrounds", 1)

        self.base_dir = settings.get("base_dir", "data")
        self.alm_cache_dir = settings.get("alm_cache_dir", "data/alm_cache")
        self.plot_dir = settings.get("plot_dir", "data/plots")
        self.data_dir = os.path.join(
            self.base_dir, "lensed" if self.lensing else "unlensed"
        )
        self.tb_dir = settings.get("tb_dir", "data/tb")
        self.model_dir = settings.get("model_dir", "data/models")

        self.nside = settings.get("nside", 1024)
        self.nsims = settings.get("nsims", 1)
        self.npatches = settings.get("npatches", 10)
        self.narray = settings.get("narray", 1)
        self.npol = len(self.polarizations)
        self.disable_noise = settings.get("disable_noise", True)

        self.beam_width = settings.get("beam_width", 1) * u.arcmin
        self.noise_scale_tt = settings.get("noise_scale_tt", 1) * u.arcmin
        self.noise_scale_ee = settings.get("noise_scale_ee", 1) * u.arcmin
        self.noise_scale_te = settings.get("noise_scale_te", 1) * u.arcmin

        self.save_fullsky = settings.get("save_fullsky", False)

        # TODO: Find a better way to do this
        self.job_array_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        self.total_sims = self.nsims * self.ndup
        if self.job_array_index is not None:
            self.job_array_index = int(self.job_array_index)
            njobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT"))  # type: ignore
            self.job_array_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN"))  # type: ignore
            self.job_array_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX"))  # type: ignore
            self.total_sims *= njobs

            print("Running job array index", self.job_array_index)
        else:
            self.total_sims *= self.narray

        self.nell = self.lmax + 1
        self.nelem = hp.Alm.getsize(self.lmax)
        self.npix = hp.nside2npix(self.nside)
        self.ells = np.arange(self.nell)
        self.chars_of_polarizations = "".join(self.polarizations)

        if self.settings.get("double_precision", False):
            self.double_precision = True
            self.r_dtype = np.float64
            self.c_dtype = np.complex128
        else:
            self.double_precision = False
            self.r_dtype = np.float32
            self.c_dtype = np.complex64

        self.nn_str = "nn_" if self.disable_noise else ""
        self.ja_str = (
            f"_{self.job_array_index}" if self.job_array_index is not None else ""
        )

        self.base_name = (
            f"{self.nside}_{self.nn_str}{self.chars_of_polarizations}_{self.total_sims}"
        )

        self.data_str = f'{self.base_name}x{self.npatches}x{self.ndup}_fnl{settings.get("fnl_range")[0]}-{settings.get("fnl_range")[1]}{self.ja_str}'
        self.data_file_nc = os.path.join(self.data_dir, f"{self.data_str}.hdf5.nc")
        self.data_file_complete = os.path.join(self.data_dir, f"{self.data_str}.hdf5")

        self.alm_str = f"{self.base_name}{self.ja_str}"
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
