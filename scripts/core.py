import argparse
import json
import logging
import os
from itertools import product
from typing import Any

import camb
import healpy as hp
import numpy as np
import rich
from astropy import units as u
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Core:
    @staticmethod
    def parse_args(args=None):
        parser = argparse.ArgumentParser()

        parser.add_argument("settings_file")
        parser.add_argument("--nsims", type=int)
        parser.add_argument("--narray", type=int)
        parser.add_argument("--npatches", type=int)
        parser.add_argument("--base_dir", type=str)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--base_name", type=str)
        parser.add_argument("--noise_scale_tt", type=float)
        parser.add_argument("--beam_width", type=float)

        parser.add_argument("--fnl_range", type=float, nargs=2)

        # for BooleanOptionalAction: --flag will set the value `flag` to True, --no-flag will set `flag` to False
        # otherwise it will be none
        parser.add_argument("--lensing", action=argparse.BooleanOptionalAction)
        parser.add_argument("--noise", action=argparse.BooleanOptionalAction)
        parser.add_argument("--force_alm_gen", action=argparse.BooleanOptionalAction)

        # this allows us to save a copy of the final settings used for the run
        # only really useful for debugging, must be provided by the CLI and not in the settings file
        parser.add_argument("--save_settings", action="store_true")

        logger.debug(f"Parsing CLI args: {args}")
        return parser.parse_args(args)

    @staticmethod
    def cosmo_defaults():
        return {
            "As": 2.13e-09,
            "ns": 0.9624,
            "pivot_scalar": 0.05,
            "max_l": 1000,
            "lmax": 500,
        }

    def __init__(self, argv=None):
        args = Core.parse_args(argv)

        ## We start by loading in the settings from the provided file
        ## From there we store the settings in the settings dict attribute
        ## We then override the settings with the command line arguments
        ## We then set the cosmological parameters to the defaults and override them with the settings
        logger.info(f"Loading settings from file {args.settings_file}")
        with open(args.settings_file, "r") as f:
            settings = self.settings = json.load(f)

        # replace the settings with the command line arguments
        for key, value in vars(args).items():
            if value is not None and key not in ["settings_file", "save_settings"]:
                logger.debug(f"Forcing setting '{key}' to '{value}' due to CLI")
                settings[key] = value

        # set the cosmological parameters defaults
        cosmo_params = self._get("cosmo_params", {})
        self.cosmo_params = settings["cosmo_params"] = {
            **Core.cosmo_defaults(),
            **cosmo_params,
        }
        if logger.getEffectiveLevel() <= logging.DEBUG:
            # Log the differences, a little complex so only do this if we will print
            defaults = Core.cosmo_defaults()
            for key in defaults.keys():
                if key in cosmo_params and defaults[key] != cosmo_params[key]:
                    logger.debug(
                        f"Overriding cosmo param {key} from {defaults[key]} to {cosmo_params[key]}"
                    )
        logger.info(f"Running with settings: \n{json.dumps(settings, indent=2)}")

        ##
        ## Now we can setup the main parameters for the run
        ##

        # main parameters
        self.lmax = self.cosmo_params["lmax"]
        self.nside = self._get("nside", 1024)
        self.pols = self._get("polarizations", "T")
        self.lensing = self._get("lensing", False)
        self.npol = len(self.pols)

        # we setup a RNG here to help with reproducibility, not tested def
        # TODO: needs more implementation, and testing of reproducibility
        self.seed = self._get("seed", np.random.default_rng().integers(0, 2**32 - 1))
        self.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

        # find out the number of sims
        self.nsims = self._get("nsims", 100)
        self.narray = self._get("narray", 1)
        self.total_sims = self.nsims * self.npol * self.narray

        # get a tuple of the sim and pol, used a few times in the code
        self.sim_pol = list(product(range(self.nsims), range(self.npol)))
        self.sim_pol_len = self.nsims * self.npol

        # sim_pol_len is also the number of sims this job will process
        logger.info(f"Processing {self.sim_pol_len} of {self.total_sims} sims")

        # setup the fnl range
        self.fnl_min, self.fnl_max = self._get("fnl_range", (-1, 1))
        self.fnl_shape = (self.nsims, self.npol, 1)

        # setup the ell values
        self.nell = self.lmax + 1
        self.nelem = hp.Alm.getsize(self.lmax)
        self.ells = np.arange(self.nell)

        # setup our prevision types to be consistent
        # TODO: More testing
        if self._get("double_precision", False):
            self.r_dtype = np.float64
            self.c_dtype = np.complex128
            self.precision = "double"
        else:
            self.r_dtype = np.float32
            self.c_dtype = np.complex64
            self.precision = "single"
        logger.debug(f"Using {self.precision} precision, where possible")

        # setup various complex things that I split off for readability
        self._setup_noise_beam()
        self._setup_radii()

        # setup the rest of the run, split for readability
        self._init_slurm()
        self._init_cosmo()
        self._init_almgen()
        self._init_patchgen()
        self._init_paths()

        # save a copy of the settings file iff --save_settings is set
        if args.save_settings:
            dir = os.path.join("settings", "runs")
            file = os.path.join(dir, f"{self.sjob}_{self.base_name}.json")
            os.makedirs(dir, exist_ok=True)
            if not os.path.exists(file):
                logger.info(f"Saving run settings to file: {file}")
                with open(file, "w") as f:
                    json.dump(self.settings, f, indent=2)
            else:
                logger.warning(f"Settings file already exists: {file}, not overwriting")

        # logger.debug("Core Object:\n %s", vars(self))
        # rich.inspect(self, methods=True, private=True)

    def _get(self, name, default: Any = None):
        val = self.settings.get(name, None)
        if val is None:
            logger.debug(f"Setting '{name}' not found, using default: {repr(default)}")
            return default
        else:
            if val != default:
                logger.debug(
                    f"Found non-default value for '{name}': {repr(val)} (default: {repr(default)})"
                )
            return val

    def _init_slurm(self):
        self.sjob = os.getenv("SLURM_JOB_ID", "-1")
        logger.info(f"SLURM job id: {self.sjob}")

        job_tasks = os.getenv("SLURM_ARRAY_TASK_COUNT")
        if job_tasks is not None:
            if int(job_tasks) != self.narray:
                raise ValueError(
                    f"SLURM_ARRAY_TASK_COUNT {job_tasks} does not match narray value {self.narray}. "
                    "Make sure to update the sbatch scripts when changing narray."
                )

            self.job_array_index = int(os.getenv("SLURM_ARRAY_TASK_ID", "-1"))
            logger.info(
                f"SLURM array id: {os.getenv('SLURM_ARRAY_JOB_ID')}, index: {self.job_array_index} / {job_tasks}"
            )

            self.is_main_job = self.job_array_index == 1
        else:
            self.job_array_index = None
            self.is_main_job = True

    def _init_paths(self):
        l = "l" if self.lensing else "ul"
        nn = "-nn" if not self.noise else ""
        j = "" if self.job_array_index is None else f"_{self.job_array_index}"
        pol = "".join(self.pols)

        self.base_dir = self._get("base_dir", "data")
        self.base_name = self._get(
            "base_name",
            f"l{self.lmax}_n{self.nside}_{l}{nn}_{pol}_{self.total_sims}",
        )

        join_paths = lambda *args: os.path.join(self.base_dir, *args)
        self.plot_dir = join_paths(self._get("plot_dir", "plots"))
        self.tb_dir = join_paths(self._get("tb_dir", "tensorboard"))
        self.model_dir = join_paths(self._get("model_dir", "models"))
        self.alm_dir = join_paths(self._get("alm_dir", "alms"))
        self.patch_dir = join_paths(self._get("patch_dir", "patches"))
        self.alm_file_partial = os.path.join(self.alm_dir, f"{self.base_name}{j}.hdf5")
        self.alm_file = os.path.join(self.alm_dir, f"{self.base_name}.hdf5")
        self.patch_str = f"{self.base_name}x{self.npatches}"
        self.patch_file = os.path.join(self.patch_dir, f"{self.patch_str}{j}.hdf5")

    def _init_cosmo(self):
        try:
            from ksw import KSW, Cosmology, Data, Shape
        except ImportError:
            # needed since the trainer does not have ksw
            logging.warning(
                "The ksw module cannot be found. Please ensure it is installed and available."
            )
            self.cosmo = self.data = self.c_ells = self.icov = self.ksw = None
            return

        cosmo_params = self.cosmo_params
        camb_params_obj = camb.set_params(**cosmo_params)
        self.cosmo = cosmo = Cosmology(camb_params_obj)
        cosmo.compute_transfer(cosmo_params["max_l"])
        cosmo.compute_c_ell()

        # We only should need the shape information for the estimator, but we can add it here
        loc_shape = Shape.prim_local(cosmo_params["ns"], cosmo_params["pivot_scalar"])
        cosmo.add_prim_reduced_bispectrum(loc_shape, self.radii)

        self.data = Data(self.lmax, self.noise_ell, self.beam_ell, self.pols, cosmo)
        if self.lensing:
            self.c_ells = cosmo.c_ell["lensed_scalar"]  # type: ignore
            self.icov = self.data.icov_diag_lensed
        else:
            self.c_ells = cosmo.c_ell["unlensed_scalar"]  # type: ignore
            self.icov = self.data.icov_diag_nonlensed
        self.c_ells = self.c_ells["c_ell"][: self.nell]

        self.ksw = KSW(
            self.cosmo.red_bispectra,
            self.icov,
            self.conv_beam(),
            self.lmax,
            self.pols,
            self.precision,
        )

    def _init_almgen(self):
        self.alm_shape = (self.nsims, self.npol, self.nelem)
        self.force_alm_gen = self._get("force_alm_gen", False)

        tr_ells = self.cosmo.transfer["ells"]  # type: ignore
        mask = tr_ells <= self.lmax
        self.tr_ells = tr_ells[mask]
        self.tr_k = self.cosmo.transfer["k"]  # type: ignore
        self.tr_ell_k = self.cosmo.transfer["tr_ell_k"][mask]  # type: ignore

    def _init_patchgen(self):
        self.patch_side_deg = self._get("patch_side_deg", 10)
        self.npatches = self._get("npatches", 2)
        assert self.npatches % 2 == 0, "Number of patches must be even"
        self.total_patches = self.npatches * self.total_sims
        self.patch_shape = (
            self.nsims,
            self.npol,
            self.npatches,
            self.nside,
            self.nside,
        )

    def _noise_ell(self, scale=1.0):
        return np.full((self.nell), scale, dtype=self.r_dtype)

    def _beam(self, width=None, lmax=None, pol=True):
        if width is None:
            width = self.beam_width
        if lmax is None:
            lmax = self.lmax

        beam = hp.gauss_beam(width, lmax=lmax, pol=pol)
        return np.swapaxes(beam, 0, 1) if pol else np.array([beam])

    def conv_beam(self):
        # return lambda alm: alm
        if self.beam_width == 0.0:
            return lambda alm: alm

        def __beam(alm):
            ret = hp.almxfl(alm, self.beam_ell, inplace=False)
            return ret

        return __beam

    def _setup_noise_beam(self):
        self.noise = self._get("noise", True)
        if not self.noise:
            self.beam_width = 0
            self.noise_scale_tt = self.noise_scale_ee = self.noise_scale_te = None
            self.noise_ell = np.full((self.nell), 10**-16, dtype=self.r_dtype)
            self.beam_ell = np.ones((self.nell), dtype=self.r_dtype)
            return

        convert = lambda x: (x * u.arcmin).to_value(u.radian)
        self.beam_width = convert(self._get("beam_width", 0))
        self.noise_scale_tt = convert(self._get("noise_scale_tt", 1e-16))
        self.noise_scale_ee = convert(self._get("noise_scale_ee", 1e-16))
        self.noise_scale_te = convert(self._get("noise_scale_te", 1e-16))

        beam = self._beam()

        noise_ell = []
        beam_ell = []
        if "T" in self.pols:
            beam_ell.append(beam[0])
            noise_ell.append(self._noise_ell(self.noise_scale_tt))
        if "E" in self.pols:
            beam_ell.append(beam[1])
            noise_ell.append(self._noise_ell(self.noise_scale_ee))
        if self.pols == ["T", "E"]:
            beam_ell.append(beam[3])
            noise_ell.append(self._noise_ell(self.noise_scale_te))
        self.noise_ell = np.array(noise_ell)
        self.beam_ell = np.array(beam_ell)

    def _setup_radii(self):
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

        r_min = int(self._get("r_min", 1))
        r_max = int(self._get("r_max", 50000))

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

        self.radii = np.array([r for r in radii if r_min <= r < r_max])
        self.drs = np.diff(radii) / 2.0
