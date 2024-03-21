import json
import os
import argparse

from attr import s
import healpy as hp
import numpy as np
from astropy import units as u

import logging

logger = logging.getLogger(__name__)


# this could be cleaned up and improved a lot but little reward
def parse_args(args):
    # note: arguments must be stored in a var name matching the dict key
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file")
    parser.add_argument("--nsims", type=int)
    parser.add_argument("--narray", type=int)
    parser.add_argument("--lensing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--noise", action=argparse.BooleanOptionalAction)
    parser.add_argument("--force_alm_gen", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fnl_range", type=float, nargs=2)
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--save_settings", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base_name", type=str)

    logger.debug(f"parsing cli args: {args}")
    return parser.parse_args(args)


def get_cosmo_defaults():
    defaults = {
        "As": 2.13e-09,
        "ns": 0.9624,
        "pivot_scalar": 0.05,
        "max_l": 1000,
        "lmax": 500,
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
            if value is not None and key not in ["settings_file", "save_settings"]:
                logger.debug(f"Forcing setting {key} to {value}")
                self.settings[key] = value

        # set the cosmological parameters defaults
        self.settings["cosmo_params"] = {
            **get_cosmo_defaults(),
            **self.settings.get("cosmo_params", {}),
        }
        if logger.getEffectiveLevel() <= logging.DEBUG:
            # Log the differences
            defaults = get_cosmo_defaults()
            overrides = self.settings.get("cosmo_params", {})
            for key in defaults.keys():
                if key in overrides and defaults[key] != overrides[key]:
                    logger.debug(
                        f"Overriding cosmo param {key} from {defaults[key]} to {overrides[key]}"
                    )

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
        self.fnl_min, self.fnl_max = settings.get("fnl_range", (-1, 1))
        self.lensing = settings.get("lensing", False)
        self.force_alm_gen = settings.get("force_alm_gen", False)

        self.seed = settings.get("seed", np.random.default_rng().integers(0, 2**32 - 1))
        self.rng = np.random.default_rng(self.seed)
        logger.info(f"Using seed {self.seed}")

        # find out the number of sims
        self.nsims = settings.get("nsims", 1)
        self.narray = settings.get("narray", 1)
        self.total_sims = self.nsims * self.npol * self.narray

        self.npatches = settings.get("npatches", 10)
        self.total_patches = self.npatches * self.total_sims
        # self.npix = hp.nside2npix(self.nside)

        # some important derived parameters
        # most of these are not really used, delete?
        self.nell = self.lmax + 1
        self.nelem = hp.Alm.getsize(self.lmax)
        self.ells = np.arange(self.nell)

        if self.settings.get("double_precision", False):
            self.r_dtype = np.float64
            self.c_dtype = np.complex128
            self.precision = "double"
        else:
            self.r_dtype = np.float32
            self.c_dtype = np.complex64
            self.precision = "single"

        # get the beam and noise
        self.noise = settings.get("noise", True)
        self.beam_width = (settings.get("beam_width", 0) * u.arcmin).to_value(u.radian)
        self.noise_scale_tt = settings.get("noise_scale_tt", 1) * u.arcmin
        self.noise_scale_tt = self.noise_scale_tt.to_value(u.radian)
        self.noise_scale_ee = settings.get("noise_scale_ee", 1) * u.arcmin
        self.noise_scale_ee = self.noise_scale_ee.to_value(u.radian)
        self.noise_scale_te = settings.get("noise_scale_te", 1) * u.arcmin
        self.noise_scale_te = self.noise_scale_te.to_value(u.radian)
        self.noise_ell, self.beam_ell = self.get_noise_beam()

        # get the radii
        r_min = settings.get("r_min", 1)
        r_max = settings.get("r_max", 50000)
        self.radii, self.drs = self.get_radii(r_min, r_max)

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
                f"Running in SLURM job {self.sjob} with job array index {self.job_array_index} of {self.narray}"
            )
        else:
            logger.info(f"Running in SLURM job {self.sjob}")
            ja_str = ""

        # paths
        self.base_dir = settings.get("base_dir", "data")
        self.alm_dir = os.path.join(self.base_dir, settings.get("alm_dir", "alms"))
        self.plot_dir = os.path.join(self.base_dir, settings.get("plot_dir", "plots"))
        self.tb_dir = os.path.join(self.base_dir, settings.get("tb_dir", "tensorboard"))
        self.model_dir = os.path.join(
            self.base_dir, settings.get("model_dir", "models")
        )
        self.patch_dir = os.path.join(
            self.base_dir, settings.get("patch_dir", "patches")
        )

        # add info to the strings
        l_str = "l" if self.lensing else "ul"
        nn_str = "-nn" if not self.noise else ""

        # set up the file names
        pol_str = "".join(self.pols)

        self.base_name = settings.get(
            "base_name",
            f"l{self.lmax}_n{self.nside}_{l_str}{nn_str}_{pol_str}_{self.total_sims}",
        )
        logger.debug(f"Base name: {self.base_name}")

        # setup the file paths
        self.patch_str = f"{self.base_name}x{self.npatches}"
        self.patch_file_partial = os.path.join(
            self.patch_dir, f"{self.patch_str}{ja_str}.hdf5"
        )
        self.patch_file = os.path.join(self.patch_dir, f"{self.patch_str}.hdf5")

        self.alm_file_partial = os.path.join(
            self.alm_dir, f"{self.base_name}{ja_str}.hdf5"
        )
        self.alm_file = os.path.join(self.alm_dir, f"{self.base_name}.hdf5")

        # save a copy of the settings file if --save_settings is set
        if args.save_settings:
            dir = os.path.join("settings", "runs")
            os.makedirs(dir, exist_ok=True)
            
            file = os.path.join(dir, f"{self.sjob}_{self.base_name}.json")
            if not os.path.exists(file):
                logger.info(f"Saving run settings to file: {file}")
                with open(file, "w") as f:
                    json.dump(self.settings, f, indent=2)
            else:
                logger.warning(f"Settings file already exists: {file}, not overwriting")

        # print out the config object for the log
        logger.debug("Config Object:\n %s", vars(self))

    def get_noise_beam(self):
        beam_ell_pre = hp.gauss_beam(self.beam_width, lmax=self.lmax, pol=True)
        beam_ell_pre = np.swapaxes(beam_ell_pre, 0, 1)

        noise_ell = []
        beam_ell = []
        if "T" in self.pols:
            noise = np.ones((self.nell), dtype=self.r_dtype) * self.noise_scale_tt**2
            noise_ell.append(noise)
            beam_ell.append(beam_ell_pre[0])

        if "E" in self.pols:
            noise = np.ones((self.nell), dtype=self.r_dtype) * self.noise_scale_ee**2
            noise_ell.append(noise)
            beam_ell.append(beam_ell_pre[1])

        if self.pols == ["T", "E"]:
            noise = np.ones((self.nell), dtype=self.r_dtype) * self.noise_scale_te**2
            noise_ell.append(noise)

        noise_ell = np.array(noise_ell).squeeze()
        beam_ell = np.array(beam_ell).squeeze()

        if not self.noise:
            noise_ell = noise_ell * 10**-12
            beam_ell = np.ones_like(beam_ell, dtype=self.r_dtype)

        return noise_ell, beam_ell

    def get_radii(self, r_min, r_max):
        # For the radii we follow Table 2. of Smith and Zaldarriaga which gives a greater density of points near reionization and recombination.
        # Spacing for all ranges but the last row are linear, with the last row having log spacing.
        # radii are in Mpc

        #    start,  stop, resolution
        ranges = [
            (0, 9500, 150),
            (9500, 11000, 300),
            (11000, 13800, 150),
            (13800, 14600, 400),
            (14600, 16000, 100),
            (16000, 50000, 100),
        ]
        radii = []

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
