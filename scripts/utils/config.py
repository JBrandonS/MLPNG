import json
import os
import argparse

import healpy as hp
import numpy as np
from astropy import units as u

import logging

logger = logging.getLogger(__name__)


def parse_args(args):
    # note: arguments must be stored in a var name matching the dict key
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file", help="which settings file to use")
    parser.add_argument("--nsims", type=int, help="The number of sims to use")
    parser.add_argument(
        "--narray",
        type=int,
        help="The number of slurm arrays used in the sbatch script",
    )
    parser.add_argument("--lensing", action="store_true", help="Use lensing")
    parser.add_argument(
        "--disable_lensing",
        action="store_false",
        dest="lensing",
        help="Do not use lensing",
    )
    parser.add_argument("--disable_noise", action="store_true", help="Disable noise")
    parser.add_argument(
        "--noise", action="store_false", dest="disable_noise", help="Enables noise"
    )
    parser.add_argument(
        "--force_alm_gen", action="store_true", help="Force alm generation"
    )
    parser.add_argument(
        "--fnl_range", type=float, nargs=2, help="The range of fnl values to use"
    )
    parser.add_argument("--base_dir", type=str, help="The base directory to use")

    logger.debug(f'Parsing args {args}')
    return parser.parse_args(args)


def get_cosmo_defaults():
    defaults = {
        "H0": 67.5,
        "r": 0,
        "As": 2.13e-09,
        "ns": 0.9624,
        "pivot_scalar": 0.05,
        "ombh2": 0.02233,
        "omch2": 0.1198,
        "mnu": 0.06,
        "tau": 0.0561,
        "TCMB": 2.7255,
        "max_l": 1000,
        "lmax": 500,
        "lens_potential_accuracy": 2,
        "DoLateRadTruncation": False,
        "AccuracyBoost": 2.0,
        "lSampleBoost": 2.0,
        "lAccuracyBoost": 2.0,
    }
    return defaults


class Config:
    def __init__(self, argv=None):
        args = parse_args(argv)
        logger.info(f"Loading settings from file {args.settings_file}")
        with open(args.settings_file, "r") as f:
            self.settings = json.load(f)

        # replace the settings with the command line arguments
        for key, value in vars(args).items():
            if value is not None:
                self.settings[key] = value

        # set the cosmological parameters defaults
        self.settings["cosmo_params"] = {
            **get_cosmo_defaults(),
            **self.settings.get("cosmo_params", {}),
        }

        # just clean up the code below a little
        settings = self.settings
        self.cosmo_params = self.settings["cosmo_params"]

        logger.info(f"Running with settings: \n{json.dumps(settings, indent=2)}")

        # main parameters
        self.lmax = self.cosmo_params["lmax"]
        self.nside = settings.get("nside", 1024)
        self.patch_side_deg = settings.get("patch_side_deg", 10)
        self.pols = settings.get("polarizations", "T")
        self.npol = len(self.pols)
        self.pol_chars = "".join(self.pols)
        self.fnl_min, self.fnl_max = settings.get("fnl_range", (-1, 1))
        self.lensing = settings.get("lensing", False)
        self.force_alm_gen = settings.get("force_alm_gen", False)

        # find out the number of sims
        self.nsims = settings.get("nsims", 1)
        self.narray = settings.get("narray", 1)
        self.npatches = settings.get("npatches", 10)
        self.total_sims = self.nsims * self.npol * self.narray
        self.total_patches = self.npatches * self.total_sims

        # some important derived parameters
        # most of these are not really used, delete?
        self.nell = self.lmax + 1
        self.nelem = hp.Alm.getsize(self.lmax)
        self.npix = hp.nside2npix(self.nside)
        self.ells = np.arange(self.nell)

        # set up units
        self.double_precision = self.settings.get("double_precision", False)
        if self.double_precision:
            self.r_dtype = np.float64
            self.c_dtype = np.complex128
        else:
            self.r_dtype = np.float32
            self.c_dtype = np.complex64

        # get the beam and noise
        self.disable_noise = settings.get("disable_noise", True)
        self.beam_width = settings.get("beam_width", 1) * u.arcmin
        self.noise_scale_tt = settings.get("noise_scale_tt", 1) * u.arcmin
        self.noise_scale_ee = settings.get("noise_scale_ee", 1) * u.arcmin
        self.noise_scale_te = settings.get("noise_scale_te", 1) * u.arcmin
        self.noise_ell, self.beam_ell = self.get_noise_beam()

        # get the radii
        self.radii, self.drs = self.get_radii(1, 50000)

        # get some info from SLURM
        self.sjob = os.getenv("SLURM_JOB_ID", 0)
        job_tasks = os.getenv("SLURM_ARRAY_TASK_COUNT")
        if job_tasks is not None:
            job_tasks = int(job_tasks)

            if job_tasks != self.narray:
                raise ValueError(
                    f"SLURM_ARRAY_TASK_COUNT {job_tasks} does not match narray value {self.narray}"
                )

            self.job_array_index = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
            ja_str = f"_{self.job_array_index}"

            if self.job_array_index == 1:
                self.is_main_job = True
            else:
                self.is_main_job = False

            logger.info(
                f"Running SLURM job {self.sjob} with job array index {self.job_array_index} of {self.narray}"
            )
        else:
            logger.info(f"Running SLURM job {self.sjob}")
            self.is_main_job = None  # unknown, only for estimator really
            self.job_array_index = None
            ja_str = ""


        # paths
        self.base_dir = settings.get("base_dir", "data")
        self.alm_dir = os.path.join(self.base_dir, settings.get("alm_dir", "alms"))
        self.plot_dir = os.path.join(self.base_dir, settings.get("plot_dir", "plots"))
        self.patch_dir = os.path.join(
            self.base_dir, settings.get("patch_dir", "patches")
        )
        self.tb_dir = os.path.join(self.base_dir, settings.get("tb_dir", "tensorboard"))
        self.model_dir = os.path.join(
            self.base_dir, settings.get("model_dir", "models")
        )

        # add info to the strings
        l_str = "l" if self.lensing else "ul"
        nn_str = "-nn" if self.disable_noise else ""

        # set up the file names
        self.base_name = f"l{self.lmax}_n{self.nside}_{l_str}{nn_str}_{self.pol_chars}_{self.total_sims}"
        logger.info(f"Base name: {self.base_name}")

        # setup the file paths
        self.patch_str = (
            f"{self.base_name}x{self.npatches}_fnl{self.fnl_min}-{self.fnl_max}{ja_str}"
        )
        self.patch_file_nc = os.path.join(self.patch_dir, f"{self.patch_str}.hdf5.nc")
        self.patch_file = os.path.join(self.patch_dir, f"{self.patch_str}.hdf5")

        self.alm_str = f"{self.base_name}{ja_str}"
        self.alm_file_nc = os.path.join(self.alm_dir, f"{self.alm_str}.alms.hdf5.nc")
        self.alm_file_partial = os.path.join(self.alm_dir, f"{self.alm_str}.alms.hdf5")
        self.alm_file = os.path.join(self.alm_dir, f"{self.base_name}.alms.hdf5")

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
