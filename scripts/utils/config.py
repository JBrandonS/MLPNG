import json
import os

import healpy as hp
import numpy as np
from astropy import units as u

import pprint

from utils import setup_logging

class Config:
    def __init__(self, settings_file, print_settings=True):
        logger = setup_logging("config")
        
        with open(settings_file, "r") as f:
            self.settings = settings = json.load(f)

        if print_settings:
            logger.info(f"Loading settings from file {settings_file}:")
            pprint.PrettyPrinter(indent=2).pprint(settings)

        self.force_alm_gen = settings.get("force_alm_gen", False)

        self.cosmo_params = settings.get("cosmo_params")

        # main parameters
        self.lmax = self.cosmo_params["lmax"]
        self.nside = settings.get("nside", 1024)
        self.npatches = settings.get("npatches", 10)
        self.narray = settings.get("narray", 1)

        self.pols = settings.get("polarizations", "T")
        self.npol = len(self.pols)
        self.pol_chars = "".join(self.pols)

        self.lensing = settings.get("lensing", False)
        self.disable_noise = settings.get("disable_noise", True)

        # find out the number of sims
        self.nsims = settings.get("nsims", 1)
        self.total_sims = self.nsims * self.narray

        job_tasks = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        if job_tasks is not None:
            job_tasks = int(job_tasks)

            if job_tasks != self.narray:
                raise ValueError(
                    f"SLURM_ARRAY_TASK_COUNT {job_tasks} does not match narray value {self.narray}"
                )
            
            self.job_array_index = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        else:
            self.job_array_index = None

        # some important derived parameters
        self.nell = self.lmax + 1
        self.nelem = hp.Alm.getsize(self.lmax)
        self.npix = hp.nside2npix(self.nside)
        self.ells = np.arange(self.nell)

        self.fnl_min, self.fnl_max = settings.get("fnl_range", [0, 0])

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

        # get the radii
        self.radii, self.drs = self.get_radii(1, 50000)

        # paths
        self.base_dir = settings.get("base_dir", "data")
        self.alm_cache_dir = os.path.join(
            self.base_dir, settings.get("alm_cache_dir", "alm_cache")
        )
        self.plot_dir = os.path.join(self.base_dir, settings.get("plot_dir", "plots"))
        self.data_dir = os.path.join(
            self.base_dir, "lensed" if self.lensing else "unlensed"
        )
        self.tb_dir = os.path.join(self.base_dir, settings.get("tb_dir", "tensorboard"))
        self.model_dir = os.path.join(
            self.base_dir, settings.get("model_dir", "models")
        )

        # add info to the strings
        nn_str = "nn_" if self.disable_noise else ""
        ja_str = f"_{self.job_array_index}" if self.job_array_index is not None else ""

        # set up the file names
        self.base_name = f"{self.nside}_{nn_str}{self.pol_chars}_{self.total_sims}"
        self.data_str = f"{self.base_name}x{self.npatches}_fnl{self.fnl_min}-{self.fnl_max}{ja_str}"
        self.alm_str = f"{self.base_name}{ja_str}"

        # setup the file paths
        self.data_file_nc = os.path.join(self.data_dir, f"{self.data_str}.hdf5.nc")
        self.data_file = os.path.join(self.data_dir, f"{self.data_str}.hdf5")
        
        self.alm_file_nc = os.path.join(
            self.alm_cache_dir, f"{self.alm_str}.alms.hdf5.nc"
        )
        self.alm_file_partial = os.path.join(
            self.alm_cache_dir, f"{self.alm_str}.alms.hdf5"
        )
        self.alm_file = os.path.join(
            self.alm_cache_dir, f"{self.base_name}.alms.hdf5"
        )

    def get_noise_beam(self):
        beam_ell_pre = hp.gauss_beam(
            self.beam_width.to_value(u.radian), lmax=self.lmax, pol=True
        )
        beam_ell_pre = np.swapaxes(beam_ell_pre, 0, 1)

        noise_ell = []
        beam_ell = []
        if "T" in self.pols:
            noise = (
                np.ones((self.nell), dtype=self.r_dtype)
                * self.noise_scale_tt.to_value(u.radian) ** 2
            )
            noise_ell.append(noise)
            beam_ell.append(beam_ell_pre[0])

        if "E" in self.pols:
            noise = (
                np.ones((self.nell), dtype=self.r_dtype)
                * self.noise_scale_ee.to_value(u.radian) ** 2
            )
            noise_ell.append(noise)
            beam_ell.append(beam_ell_pre[1])

        if self.pols == ["T", "E"]:
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

    def get_radii(self, r_min, r_max):
        # For the radii we follow Table 2. of Smith and Zaldarriaga which gives a greater density of points near reionization and recombination.
        # Spacing for all ranges but the last row are linear, with the last row having log spacing.
        # radii are in Mpc

        radii = []
        #    start,  stop, resolution
        ranges = [
            (0, 9500, 150),
            (9500, 11000, 300),
            (11000, 13800, 150),
            (13800, 14600, 400),
            (14600, 16000, 100),
            (16000, 50000, 100),
        ]

        for r in ranges:
            start = max(r_min, r[0])
            end = min(r_max, r[1])

            if start > end:
                continue

            if r == ranges[-1]:  # For the last range, use logspace
                temp_radii = np.logspace(np.log10(start), np.log10(end), num=r[2])
            else:
                temp_radii = np.linspace(start, end, num=r[2], endpoint=False)

            radii.extend(temp_radii)

        radii = np.array([r for r in radii if r_min <= r < r_max])
        drs = np.diff(radii) / 2.0
        return radii, drs
