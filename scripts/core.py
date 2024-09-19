import sys
import argparse
import json
import logging
import os
from typing import Any

import camb
import healpy as hp
import numpy as np
from astropy import units as u

from ksw import Cosmology

logger = logging.getLogger(__name__)


def cosmo_defaults():
    """Some default settings that are required by KSW / camb."""
    return {"As": 2.13e-09, "ns": 0.9624, "pivot_scalar": 0.05}


class Core:
    """
    The `Core` class represents the core functionality of the MLPNG (Machine Learning for Primordial Non-Gaussianity) project.
    It handles parsing command line arguments, loading settings from a file, setting up cosmological parameters, and initializing various components for the run.

    Attributes:
        settings (dict): A dictionary containing the settings loaded from the settings file.
        cosmo_params (dict): A dictionary containing the cosmological parameters.
        lmax (int): The maximum multipole moment for the run. Default: 500
        nside (int): The number of pixels on each side of the HEALPix map. Default: 1024
        pols (str): The polarizations to consider (i.e., "T", "E"). Default: T
        lensing (bool): Whether to include lensing effects in the simulation. Default: False
        npol (int): The number of polarizations.
        seed (int): The random seed for reproducibility.
        rng (numpy.random.Generator): The random number generator.
        nsims (int): The number of simulations to run. Default: 100
        narray (int): The number of arrays to process. Default: 1
        total_sims (int): The total number of simulations to process.
        fnl_min (float): The minimum value of the local non-Gaussianity parameter (fnl). Default: -10
        fnl_max (float): The maximum value of the local non-Gaussianity parameter (fnl). Default: 10
        fnl_shape (tuple): The shape of the fnl array.
        nell (int): The number of ell values.
        nelem (int): The number of elements in the alm array.
        ells (numpy.ndarray): An array of ell values.
        r_dtype (numpy.dtype): The data type for real numbers. Default: float32
        c_dtype (numpy.dtype): The data type for complex numbers. Default: complex64
        precision (str): The precision level ("single" or "double"). Default: 'single'
        sjob (str): The slurm job ID, -1 if not found.
        job_array_index (int): The index of the job array.
        is_main_job (bool): Whether the current job is the main job.
        base_dir (str): The base directory for output files. Default: data
        base_name (str): The base name for output files.
        plot_dir (str): The directory for plot files. Default: plots
        tb_dir (str): The directory for TensorBoard files. Default: tensorboard
        model_dir (str): The directory for model files. Default: models
        alm_dir (str): The directory for alm files. Default: alms
        patch_dir (str): The directory for patch files. Default: patches
        alm_file_partial (str): The partial path to the alm file.
        alm_file (str): The full path to the alm file.
        patch_str (str): The string representation of the patch file.
        patch_file (str): The full path to the patch file.
        cosmo (ksw.Cosmology): The cosmology object.
        radii (numpy.ndarray): An array of radii for the bispectrum estimator.
        force_ksw (bool): Whether to force the KSW estimator to run otherwise can use save states.
        num_estimates (int): The number of estimates to compute. Default: The total number of simulations.
    """

    def __init__(self, argv=None):
        """
        Initializes a new instance of the `Core` class.

        Args:
            argv (list): List of the CLI Args
            If None, sys.argv will be used.
            inspect_class (bool): Whether to inspect the class using the `rich` library. Default is False.
        """

        # handle the CLI args first
        args = self.parse_args(argv)

        # We start by loading in the settings from the provided file
        # From there we store the settings in the settings dict attribute
        # We then override the settings with the command line arguments
        # We then set the cosmological parameters to the defaults and override them with the settings
        logger.info("Loading settings from file %s", args.settings_file)
        with open(args.settings_file, "r") as f:
            settings = self.settings = json.load(f)

        # replace the settings with the command line arguments
        for key, value in vars(args).items():
            # we do not want to save the settings_file or save_settings options, so ignore those
            if value is not None and key not in ["settings_file", "save_settings"]:
                logger.debug(f"Forcing setting '{key}' to '{value}' due to CLI")
                settings[key] = value

        # set the cosmological parameters defaults, and update based on cosmo_params
        cosmo_params = self._get("cosmo_params", cosmo_defaults())
        self.cosmo_params = settings["cosmo_params"] = {
            **cosmo_defaults(),
            **cosmo_params,
        }

        if logger.isEnabledFor(logging.DEBUG):
            # This just logs any changes to the defaults, but only if in debug mode
            # Not really needed, but good logs can be helpful
            defaults = cosmo_defaults()
            for key in defaults.keys():
                if key in cosmo_params and defaults[key] != cosmo_params[key]:
                    logger.debug(
                        "Overriding cosmo param %s from %s to %s",
                        key,
                        defaults[key],
                        cosmo_params[key],
                    )

        logger.info("Running with settings: \n%s", json.dumps(settings, indent=2))

        ##
        ## Now we can process the main parameters for the run, each is loaded in via the _get method
        ##

        # we setup a RNG here for reproducibility
        # TODO: needs more implementation, and testing of reproducibility, need to setup tensorflow seed and probably others
        # overall, dont expect reproducibility, not a high priority
        self.seed = self._get("seed", np.random.default_rng().integers(0, 2**32 - 1))
        self.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

        # main parameters
        self.nside = self._get("nside", 1024)
        self.lensing = self._get("lensing", False)
        self.nsims = self._get("nsims", 100)
        self.narray = self._get("narray", 100)
        self.force_gen = self._get("force_generation", False)
        self.force_ksw = self._get("force_ksw", False)
        self.num_estimates = self._get("num_estimates", self.nsims * self.narray)

        # setup the lmax values
        self.lmax = self._get("lmax", 3 * self.nside - 1)
        if self.lmax < 300:
            logger.warning(
                "lmax = %s, lmax < 300 not supported. Setting lmax to 300", self.lmax
            )
            self.lmax = 300
        self.max_l = self.lmax + self._get("lmax_buffer", 512)
        self.cosmo_params["lmax"] = self.lmax

        # setup the fnl values
        self.fnl_min, self.fnl_max = self._get("fnl_range", (-10, 10))
        self.fnl_shape = (self.nsims, 1, 1)

        # setup the polarization which should support --pols [T|E|TE]
        pols = self._get("pols", "T")
        if isinstance(pols, str):
            pols = tuple(pols)
        elif isinstance(pols, list):
            pols = tuple(pols)
        # we do not want to support B mode since it is so small, so lets remove anything with B in it.
        self.pols = tuple(c for p in pols for c in p if p != "B")
        self.npols = len(self.pols)
        self.iso = self._get("iso", False)

        self.use_t = "T" in self.pols
        self.use_e = "E" in self.pols
        self.use_b = False  # "B" in self.pols
        if not self.iso and self.use_t and self.use_e:
            self.use_te = True
        else:
            self.use_te = False
        logger.debug(
            "Using polarizations T: %s, E: %s, TE: %s",
            self.use_t,
            self.use_e,
            self.use_te,
        )

        # setup our precision types to be consistent
        # also, tensorflow seems to mostly use float32, and is actually moving to half-bit registers
        # aka float16, it probably gets us nothing to go higher other than making everything take longer
        # but we could test double_precision more
        if self._get("double_precision", False):
            self.r_dtype = np.float64
            self.c_dtype = np.complex128
            self.precision = "double"
        else:
            self.r_dtype = np.float32
            self.c_dtype = np.complex64
            self.precision = "single"
        logger.debug("Using %s precision, where possible", self.precision)

        # setup some derived parameters
        self.total_sims = self.nsims * self.npols * self.narray

        # setup some info parameters
        self.nell = self.lmax + 1
        self.nelem = hp.Alm.getsize(self.lmax)
        self.ells = np.arange(self.nell)
        self.alm_shape = (self.nsims, self.npols, self.nelem)

        # some patch settings
        self.patch_side_deg = self._get("patch_side_deg", 10)
        self.npatches = self._get("npatches", 10)
        self.total_patches = self.npatches * self.total_sims
        self.patch_shape = (
            self.nsims,
            self.npols,
            self.npatches,
            self.nside,
            self.nside,
        )

        # We need the number of patches to be even for the healpy code
        assert self.npatches % 2 == 0, "Number of patches must be even"

        # split the rest of this function into a few smaller functions for readability
        # each of these modifies attributes of the object
        self._setup_noise_beam()
        self._init_radii()
        self._init_slurm()
        self._init_cosmo()
        self._init_paths()

        # save a copy of the settings file iff --save-settings is set
        if args.save_settings:
            dir = os.path.join(self.base_dir, "settings")
            file = os.path.join(dir, f"{self.sjob}_{self.base_name}.json")
            os.makedirs(dir, exist_ok=True)

            if not os.path.exists(file):
                logger.info("Saving run settings to file: %s", file)
                with open(file, "w") as f:
                    json.dump(self.settings, f, indent=2)
            else:
                logger.warning("Settings already exists: %s, not overwriting", file)

    def parse_args(self, args=None):
        """
        Parse command line arguments.

        Args:
            args (list): List of command line arguments. If None, sys.argv will be used.

        Returns:
            argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("settings_file")

        # some standard arguments heres, we can add more as needed
        parser.add_argument("--nsims", type=int)
        parser.add_argument("--narray", type=int)
        parser.add_argument("--npatches", type=int)
        parser.add_argument("--base_dir", type=str)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--base_name", type=str)
        parser.add_argument("--lmax", type=int)
        parser.add_argument("--lmax_buffer", type=int)
        parser.add_argument("--fnl_range", type=float, nargs=2)
        parser.add_argument("--num_estimates", type=int)
        parser.add_argument("--pols", type=str, nargs="+")

        # for BooleanOptionalAction: --flag will set the value `flag` to True, --no-flag will set `flag` to False
        # otherwise it will be none
        parser.add_argument("--lensing", action=argparse.BooleanOptionalAction)
        parser.add_argument("--noise", action=argparse.BooleanOptionalAction)
        parser.add_argument("--force_generation", action=argparse.BooleanOptionalAction)
        parser.add_argument("--force_ksw", action=argparse.BooleanOptionalAction)

        # this allows us to save a copy of the final settings used for the run
        # only really useful for debugging, must be provided by the CLI and not in the settings file
        parser.add_argument("--save_settings", action="store_true")

        if args is None:
            args = sys.argv[1:]

        logger.info(f"Parsing CLI args: {args}")
        pargs, _ = parser.parse_known_args(args)
        return pargs

    def _get(self, name, default: Any = None, verbose=True):
        """
        Get the value of a setting, providing the default if the setting is not found in self.settings.
        Logs information about the setting value if it is found and differs from the default, at DEBUG level.

        Args:
            name (str): The name of the setting.
            default (Any, optional): The default value to return if the setting is not found. Defaults to None.
            verbose (bool, optional): Whether to log information about the setting value. Defaults to True.

        Returns:
            Any: The value of the setting if found, otherwise the default value.
        """
        val = self.settings.get(name, None)
        if val is None:
            if verbose:
                logger.debug(
                    "Setting '%s' not found, using default: %s", name, repr(default)
                )

            return default
        else:
            if val != default and verbose:
                logger.debug(
                    "Found non-default value for '%s': %s (default: %s)",
                    name,
                    repr(val),
                    repr(default),
                )
            return val

    def _setup_noise_beam(self):
        """
        Set up the noise and beam parameters for the Core object.

        This method initializes the noise and beam parameters based on the configuration settings.
        If the noise parameter is set to False, the noise is set to a very small value to avoid a singular matrix
        in the inverse covariance. The beam width is set to 0 in this case.

        If the noise parameter is set to True, the noise and beam parameters are retrieved from the configuration
        settings. There are two methods to load in noise:
            - noise_file and beam_file can be set to file, these files should be in T, E, B, TE order
            - alternativly the beam width and noise scales for temperature (TT), E-mode polarization (EE), B-mode polarization (BB), and
        temperature-E-mode polarization (TE) should be provided via the `beam_width` and `noise_XX` where noise_XX is the noise amplitude in muK arcminutes to muK radians.

        The beam and noise ell values are computed based on the polarization settings and stored in the
        `noise_ell` and `beam_ell` attributes.

        Returns:
            None
        """

        def convert(x):
            """Helper function to convert from arcmin to radians."""
            return (x * u.arcmin).to_value(u.radian)

        self.noise = self._get("noise", True)
        if not self.noise:
            # we cannot set the noise to 0 as this will cause a singular matrix in the inverse covariance
            # so we set it to a very small value in such a way that we will not get a singular matrix
            noise_scale = convert(1e-6)
            self.beam_width = 0

            self.beam_ell = np.ones((4, self.nell), dtype=self.r_dtype)
            self.noise_ell = (
                np.ones((4, self.nell), dtype=self.r_dtype) * noise_scale**2
            )
            # go ahead and set the B and Te to 0
            self.noise_ell[2:] = 0
            return

        self.beam_width = convert(self._get("beam_width", 0))
        beam = hp.gauss_beam(self.beam_width, lmax=self.lmax, pol=True)  # (nell, npol)
        self.beam_ell = np.swapaxes(beam, 0, 1)  # convert beam to (npol, nell)

        # T, E, B, TE
        self.noise_ell = np.ones((4, self.nell), dtype=self.r_dtype)
        self.noise_ell[0] = convert(self._get("noise_tt", 1)) ** 2
        self.noise_ell[1] = convert(self._get("noise_ee", 1)) ** 2
        self.noise_ell[2] = 0  # B is always 0
        self.noise_ell[3] = convert(self._get("noise_te", 2)) ** 2

    def _init_radii(self):
        """
        Setup the radii and drs arrays.

        This method sets up the radii and drs arrays based on the r_min and r_max attributes and a predefined set of ranges.
        See Smith and Zaldarriaga (2011) Section 5.2 for more details.

        Attributes:
            radii (numpy.ndarray): An array of radii for the bispectrum estimator.
        """
        #    start,  stop, resolution
        ranges = [
            (0, 9500, 150),
            (9500, 11000, 300),
            (11000, 13800, 150),
            (13800, 14600, 400),
            (14600, 16000, 100),
            (16000, 50000, 100),
        ]

        r_min = int(self._get("r_min", 1))
        r_max = int(self._get("r_max", 50000))

        radii = []
        for r in ranges:
            start = max(r_min, r[0])
            end = min(r_max, r[1])

            if start >= end:
                continue

            if r == ranges[-1]:  # For the last range, use logspace
                temp_radii = np.logspace(np.log10(start), np.log10(end), num=r[2])
            else:
                temp_radii = np.linspace(start, end, num=r[2], endpoint=False)

            radii.extend(temp_radii)

        self.radii = np.array([r for r in radii if r_min <= r <= r_max])

    def _init_slurm(self):
        """
        Initializes the slurm environment variables and sets the job array index and main job flag.

        This method retrieves the slurm environment variables, such as the job ID and array task count,
        and performs necessary checks. It sets the job array index and determines whether the current
        job is the main job.

        Attributes:
            sjob (str): The slurm job ID.
            n_cpus (int): The number of CPUs available for the job.
            job_array_id (str): The id of the slurm array job.
            job_array_index (int): The index of the current job in the slurm array.
            is_main_job (bool): Indicates whether the current job is the main job.

        Raises:
            ValueError: If the SLURM_ARRAY_TASK_COUNT does not match the narray value.
        """
        self.sjob = os.getenv("SLURM_JOB_ID", "-1")
        logger.debug("Slurm job id: %s", self.sjob)
        logger.debug("Slurm job name: %s", os.getenv("SLURM_JOB_NAME", "unknown"))

        # os.sched_getaffinity(0) gets the number of usable CPUs available, this is different from
        # os.cpu_count() which gets the number of CPUs on the system
        self.n_cpus = int(
            os.getenv("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0)))
        )
        logger.debug("Number of available CPUs: %s", self.n_cpus)

        job_tasks = os.getenv("SLURM_ARRAY_TASK_COUNT")
        if job_tasks is not None:
            if int(job_tasks) != self.narray:
                raise ValueError(
                    f"SLURM_ARRAY_TASK_COUNT {job_tasks} does not match narray value {self.narray}. "
                    "Make sure to update the sbatch scripts when changing narray."
                )

            self.job_array_id = os.getenv("SLURM_ARRAY_JOB_ID", "-1")
            self.job_array_index = int(os.getenv("SLURM_ARRAY_TASK_ID", "-1"))
            logger.debug(
                "Slurm array id: %s, index: %s of %s jobs",
                self.job_array_id,
                self.job_array_index,
                job_tasks,
            )

            self.is_main_job = self.job_array_index == 1
        else:
            self.job_array_id = self.job_array_index = None
            self.is_main_job = True

    def _init_paths(self):
        """
        Initializes the paths used by the MLPNG script.

        The method sets various directory paths based on the configuration parameters.
        These paths include the base directory, plot directorymodel directory, data directory,
        and file paths for data files.

        Attributes:
            base_dir (str): The base directory for the script.
            base_name (str): The base name for the files generated by the script.
            plot_dir (str): The directory for saving plot files.
            model_dir (str): The directory for saving model files.
            data_dir (str): The directory for saving data files.
            file_partial (str): The file path for the partial data file.
            file_complete (str): The file path for the complete data file.
            mc_dir (str): The directory for saving KSW Monte Carlo files.
            mc_file (str): The file path for the KSW Monte Carlo file.

        Returns:
            None
        """

        def join_paths(*args):
            return os.path.join(self.base_dir, *args)

        lens = "l" if self.lensing else "ul"
        nn = "nn" if not self.noise else "n"
        j: str = "" if self.job_array_index is None else f"_{self.job_array_index}"
        pol_str = "".join(self.pols)

        def_name = f"l{self.lmax}_n{self.nside}_{lens}-{nn}_{pol_str}x{self.total_sims}_f{self.fnl_min}-{self.fnl_max}"
        self.base_name = self._get("base_name", def_name)

        self.base_dir = self._get("base_dir", "data")
        self.plot_dir = join_paths(self._get("plot_dir", "plots"))
        self.model_dir = join_paths(self._get("model_dir", "models"))

        self.data_dir = join_paths(self._get("data_dir", "data"))
        self.file_partial = os.path.join(self.data_dir, f"{self.base_name}{j}.hdf5")
        self.file_complete = os.path.join(self.data_dir, f"{self.base_name}.hdf5")

        self.mc_dir = join_paths(self._get("mc_dir", "kswmc"))
        self.mc_file = os.path.join(self.mc_dir, f"{self.base_name}.hdf5")

        self.tb_dir = join_paths(self._get("tb_dir", "tensorboard"))
        self.wandb_dir = join_paths(self._get("wandb_dir", "wandb"))

        # lets just make sure the main directories exist, saving the need to do this later
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.mc_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.wandb_dir, exist_ok=True)

        logger.debug("Using base name: '%s'", self.base_name)
        logger.debug("Using base directory: '%s'", self.base_dir)
        logger.debug("Using plot directory: '%s'", self.plot_dir)
        logger.debug("Using model directory: '%s'", self.model_dir)
        logger.debug("Using data directory: '%s'", self.data_dir)
        logger.debug("Using data file: '%s'", self.file_partial)
        logger.debug("Using complete data file: '%s'", self.file_complete)
        logger.debug("Using KSW MC file: '%s'", self.mc_file)
        logger.debug("Using TensorBoard directory: '%s'", self.tb_dir)
        logger.debug("Using Weights & Biases directory: '%s'", self.wandb_dir)

    def _init_cosmo(self):
        """
        Initializes the cosmology parameters and sets up the necessary objects for computation.

        Attributes:
            cosmo (Cosmology): An instance of the `Cosmology` class.
            c_ells (numpy.ndarray): An array of C_ell values from camb. These values have been noised and beamed via B_\\ell^2 C_\\ell + N_\\ell.

        Returns:
            None
        """
        logger.debug("Initializing cosmology")

        camb_params = camb.set_params(**self.cosmo_params)
        # ip = camb.initialpower.InitialPowerLaw()
        # ip.set_params(As=self.cosmo_params["As"], ns=self.cosmo_params["ns"])
        # camb_params.set_initial_power(ip)
        self.cosmo = cosmo = Cosmology(camb_params)

        cosmo.compute_transfer(self.max_l)
        cosmo.compute_c_ell()

        if self.lensing:
            self.c_ells = cosmo.c_ell["lensed_scalar"]["c_ell"].T
        else:
            self.c_ells = cosmo.c_ell["unlensed_scalar"]["c_ell"].T

        # trim c_ells to the correct length
        self.c_ells = self.c_ells[:, : self.nell]
        self.cov_tot = self.beam_ell**2 * self.c_ells + self.noise_ell

        if self.use_te:
            # we need to square the matrix and invert it properly
            cov = np.zeros((2, 2, self.nell))
            cov[0, 0] = self.cov_tot[0]  # TT
            cov[1, 1] = self.cov_tot[1]  # EE
            cov[1, 0] = self.cov_tot[3]  # ET
            cov[0, 1] = self.cov_tot[3]  # TE
            icov = np.linalg.inv(cov.T).T

            # lets flatten this for our code, will be TT, EE, BB, TE order to match alms
            self.icov_tot = np.array(
                [
                    icov[0, 0],  # TT
                    icov[1, 1],  # EE
                    np.zeros(self.nell),  # BB
                    icov[1, 0],  # TE
                ],
            )
        else:
            # to avoid a divide by zero we skip the mono and dipole terms, and b mode
            self.icov_tot = np.zeros_like(self.cov_tot)
            self.icov_tot[[0, 1, 3], 2:] = 1 / self.cov_tot[[0, 1, 3], 2:]

    def pol_idxs(self, keep_b=False, keep_te=False, pretrimmed=False):
        """
        Get the polarization indices based on the configuration settings.

        Parameters:
            keep_b (bool): Whether to keep the B-mode polarization. Default is False.
            keep_te (bool): Whether to keep the TE-mode polarization. Default is False.
            pretrimmed (bool): Whether the alms are pretrimmed, i.e. have we removed the t-modes. Default is False.
        """
        start = 0 if self.use_t else (0 if pretrimmed else 1)
        num_to_take = 1 if self.use_t else 0

        if self.use_e:
            num_to_take += 2 if keep_b else 1  # we also need B
        if keep_te and self.use_te:
            num_to_take += 1

        return np.array(range(start, start + num_to_take))