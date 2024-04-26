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

from ksw import KSW, Cosmology, Data, Shape

logger = logging.getLogger(__name__)


def cosmo_defaults():
    return {
        "As": 2.13e-09,
        "ns": 0.9624,
        "pivot_scalar": 0.05,
        "max_l": 1000,
        "lmax": 500,
    }


class Core:
    """
    The `Core` class represents the core functionality of the MLPNG (Machine Learning for Primordial Non-Gaussianity) project.
    It handles parsing command line arguments, loading settings from a file, setting up cosmological parameters, and initializing various components for the run.

    Attributes:
        settings (dict): A dictionary containing the settings loaded from the settings file.
        cosmo_params (dict): A dictionary containing the cosmological parameters.
        lmax (int): The maximum multipole moment for the run.
        nside (int): The number of pixels on each side of the HEALPix map.
        pols (str): The polarizations to consider (e.g., "T", "E", "B").
        lensing (bool): Whether to include lensing effects in the simulation.
        npol (int): The number of polarizations.
        seed (int): The random seed for reproducibility.
        rng (numpy.random.Generator): The random number generator.
        nsims (int): The number of simulations to run.
        narray (int): The number of arrays to process.
        total_sims (int): The total number of simulations to process.
        sim_pol (list): A list of tuples representing the simulation and polarization indices.
        sim_pol_len (int): The number of simulations and polarizations to process.
        fnl_min (float): The minimum value of the local non-Gaussianity parameter (fnl).
        fnl_max (float): The maximum value of the local non-Gaussianity parameter (fnl).
        fnl_shape (tuple): The shape of the fnl array.
        nell (int): The number of ell values.
        nelem (int): The number of elements in the alm array.
        ells (numpy.ndarray): An array of ell values.
        r_dtype (numpy.dtype): The data type for real numbers.
        c_dtype (numpy.dtype): The data type for complex numbers.
        precision (str): The precision level ("single" or "double").
        sjob (str): The SLURM job ID.
        job_array_index (int): The index of the job array.
        is_main_job (bool): Whether the current job is the main job.
        base_dir (str): The base directory for output files.
        base_name (str): The base name for output files.
        plot_dir (str): The directory for plot files.
        tb_dir (str): The directory for TensorBoard files.
        model_dir (str): The directory for model files.
        alm_dir (str): The directory for alm files.
        patch_dir (str): The directory for patch files.
        alm_file_partial (str): The partial path to the alm file.
        alm_file (str): The full path to the alm file.
        patch_str (str): The string representation of the patch file.
        patch_file (str): The full path to the patch file.
        cosmo (ksw.Cosmology): The cosmology object.
        radii (numpy.ndarray): An array of radii for the bispectrum estimator.

    Methods:
        parse_args(args=None): Parses the command line arguments and returns the parsed arguments.
        cosmo_defaults(): Returns a dictionary of default cosmological parameters.
        __init__(argv=None): Initializes a new instance of the `Core` class.
        _get(name, default=None): Gets the value of a setting from the settings dictionary.
        _init_slurm(): Initializes SLURM-related attributes.
        _init_paths(): Initializes file path-related attributes.
        _init_cosmo(): Initializes the cosmology object.
    """

    def parse_args(self, args=None):
        """
        Parse command line arguments.

        Args:
            args (list): List of command line arguments. If None, sys.argv[1:] will be used.

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
        parser.add_argument("--noise_scale_tt", type=float)
        parser.add_argument("--beam_width", type=float)

        parser.add_argument("--fnl_range", type=float, nargs=2)
        parser.add_argument("--polarizations", nargs="*", type=str)

        # for BooleanOptionalAction: --flag will set the value `flag` to True, --no-flag will set `flag` to False
        # otherwise it will be none
        parser.add_argument("--lensing", action=argparse.BooleanOptionalAction)
        parser.add_argument("--noise", action=argparse.BooleanOptionalAction)
        parser.add_argument("--force_alm_gen", action=argparse.BooleanOptionalAction)

        # this allows us to save a copy of the final settings used for the run
        # only really useful for debugging, must be provided by the CLI and not in the settings file
        parser.add_argument("--save_settings", action="store_true")

        logger.debug(f"Parsing CLI args: {args}")
        parsed_args = parser.parse_args(args)

        if parsed_args.polarizations is not None:
            for pol in parsed_args.polarizations:
                if pol not in ["T", "E"]:
                    raise ValueError(
                        f"Invalid polarization: {pol}. Polarizations must be either 'T' or 'E'."
                    )

        return parsed_args

    def __init__(self, argv=None, inspect_class=False):
        """
        Initializes a new instance of the `Core` class.

        Args:
            argv (list): List contating the CLI Args, first arg should be the settings file, others follow arg_parse.
            If None, sys.argv[1:] will be used.
            inspect_class (bool): Whether to inspect the class using the `rich` library. Default is False.
        """
        args = self.parse_args(argv)

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

        # set the cosmological parameters defaults, and update based on cosmo_params
        cosmo_params = self._get("cosmo_params", cosmo_defaults())
        self.cosmo_params = settings["cosmo_params"] = {
            **cosmo_defaults(),
            **cosmo_params,
        }
        if logger.getEffectiveLevel() <= logging.DEBUG:
            # This just logs any changes to the defaults, but only if in debug mode
            defaults = cosmo_defaults()
            for key in defaults.keys():
                if key in cosmo_params and defaults[key] != cosmo_params[key]:
                    logger.debug(
                        f"Overriding cosmo param {key} from {defaults[key]} to {cosmo_params[key]}"
                    )
        logger.info(f"Running with settings: \n{json.dumps(settings, indent=2)}")

        ##
        ## Now we can process the main parameters for the run, each is loaded in via the _get method
        ##

        # we setup a RNG here for reproducibility
        # TODO: needs more implementation, and testing of reproducibility, need to setup tensorflow seed and probably others
        self.seed = self._get("seed", np.random.default_rng().integers(0, 2**32 - 1))
        self.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

        # main parameters
        self.lmax = self.cosmo_params["lmax"]
        self.nside = self._get("nside", 1024)
        self.pols = self._get("polarizations", "T")
        self.lensing = self._get("lensing", False)
        self.npol = len(self.pols)
        self.nsims = self._get("nsims", 100)
        self.narray = self._get("narray", 1)
        self.total_sims = self.nsims * self.npol * self.narray

        # setup the fnl and shape
        self.fnl_min, self.fnl_max = self._get("fnl_range", (-1, 1))
        self.fnl_shape = (self.nsims, self.npol, 1)

        # get a tuple of the sim and pol, used a few times in the code
        self.sim_pol = list(product(range(self.nsims), range(self.npol)))
        self.sim_pol_len = self.nsims * self.npol

        # setup the ell values, just to have them for later
        self.nell = self.lmax + 1
        self.nelem = hp.Alm.getsize(self.lmax)
        self.ells = np.arange(self.nell)

        # setup our precision types to be consistent
        # TODO: More testing with double precision, some areas default to single, some double
        if self._get("double_precision", False):
            self.r_dtype = np.float64
            self.c_dtype = np.complex128
            self.precision = "double"
        else:
            self.r_dtype = np.float32
            self.c_dtype = np.complex64
            self.precision = "single"
        logger.debug(f"Using {self.precision} precision, where possible")

        # split the rest of this function into a few smaller functions for readability
        # each of these modifies attributes of the object
        self._setup_noise_beam()
        self._setup_radii()
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

        if inspect_class:
            rich.inspect(self, all=True)

    def _get(self, name, default: Any = None):
        """
        Get the value of a setting, providing the default if the setting is not found in self.settings.
        Logs information about the setting value if it is found and differs from the default, at DEBUG level.

        Args:
            name (str): The name of the setting.
            default (Any, optional): The default value to return if the setting is not found. Defaults to None.

        Returns:
            Any: The value of the setting if found, otherwise the default value.
        """
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
        """
        Initializes the SLURM environment variables and sets the job array index and main job flag.

        This method retrieves the SLURM environment variables, such as the job ID and array task count,
        and performs necessary checks. It sets the job array index and determines whether the current
        job is the main job.

        Attributes:
            sjob (str): The SLURM job ID.
            job_array_index (int): The index of the current job in the SLURM array.
            is_main_job (bool): Indicates whether the current job is the main job.

        Raises:
            ValueError: If the SLURM_ARRAY_TASK_COUNT does not match the narray value.
        """
        self.sjob = os.getenv("SLURM_JOB_ID", "-1")
        logger.info(f"SLURM job id: {self.sjob}")

        self.scpus = int(os.environ["SLURM_CPUS_PER_TASK"])
        logger.debug(f"SLURM cpus per task: {self.scpus}")

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
        """
        Initializes the paths used by the MLPNG script.

        The method sets various directory paths based on the configuration parameters.
        These paths include the base directory, plot directory, tensorboard directory,
        model directory, alm directory, patch directory, and file paths for alm and patch files.

        Attributes:
            base_dir (str): The base directory for the script.
            base_name (str): The base name for the files generated by the script.
            plot_dir (str): The directory for saving plot files.
            tb_dir (str): The directory for saving TensorBoard files.
            model_dir (str): The directory for saving model files.
            alm_dir (str): The directory for saving alm files.
            patch_dir (str): The directory for saving patch files.
            alm_file_partial (str): The file path for the partial alm file.
            alm_file (str): The file path for the complete alm file.
            patch_str (str): The string representation of the patch file.
            patch_file (str): The file path for the patch file.

        Returns:
            None
        """
        lens = "l" if self.lensing else "ul"
        nn = "-nn" if not self.noise else ""
        j = "" if self.job_array_index is None else f"_{self.job_array_index}"
        pol = "".join(self.pols)

        self.base_dir = self._get("base_dir", "data")
        self.base_name = (
            self._get("base_name", f"l{self.lmax}_n{self.nside}_{lens}{nn}_{pol}")
            + f"x{self.total_sims}"
        )
        logger.info(f"Base name: {self.base_name}")

        def join_paths(*args):
            return os.path.join(self.base_dir, *args)

        self.plot_dir = join_paths(self._get("plot_dir", "plots"))
        self.tb_dir = join_paths(self._get("tb_dir", "tensorboard"))
        self.model_dir = join_paths(self._get("model_dir", "models"))
        self.alm_dir = join_paths(self._get("alm_dir", "alms"))
        self.patch_dir = join_paths(self._get("patch_dir", "patches"))
        self.alm_file_partial = os.path.join(self.alm_dir, f"{self.base_name}{j}.hdf5")
        self.alm_file = os.path.join(self.alm_dir, f"{self.base_name}.hdf5")
        self.patch_str = f"{self.base_name}x{self.npatches}"
        self.patch_file = os.path.join(self.patch_dir, f"{self.patch_str}{j}.hdf5")

    def icov(self, alm):
        """
        Returned the inverse covariance of the data, used in the KSW estimator.

        Function takes (npol, nelem) alm-like complex array "a" and returns the
        inverse-variance-weighted version of that array. Specifically:
        (B^{-1} N B^{-1} + S)^{-1} B^{-1} a, where a = B s + n, B is the beam
        and N^{-1} and S^{-1} are the inverse noise and signal covariance
        matrices, respectively.

        Parameters:
        - alm (ndarray): (npol, nelem) alm-like complex array.

        Returns:
        - ndarray: Inverse-variance-weighted version of the input array.

        """
        ret = np.empty_like(alm)
        for pol in range(alm.shape[0]):
            B = self.beam_ell[pol]
            iB = 1 / B
            N = self.noise_ell[pol]
            S = self.c_ells[:, pol]
            factor = (iB * N * iB + S) ** -1 * iB
            ret[pol] = hp.almxfl(alm[pol], factor, inplace=False)
        return ret

    def _init_cosmo(self):
        """
        Initializes the cosmology parameters and sets up the necessary objects for computation.

        Attributes:
            cosmo (Cosmology): An instance of the `Cosmology` class.
            c_ells (numpy.ndarray): An array of C_ell values from camb. These values have been noised and beamed via B_\\ell^2 C_\\ell + N_\\ell.
            icov (func): gives a mapping function from alm to the inverse covariance of the data. Is lensed or unlested depending on the lensing flag.
            data (Data): An instance of the `Data` class.
            ksw (KSW): An instance of the `KSW` class.

        Returns:
            None
        """
        cosmo_params = self.cosmo_params
        camb_params_obj = camb.set_params(**cosmo_params)
        self.cosmo = cosmo = Cosmology(camb_params_obj)

        logger.debug("Computing transfer functions and C_ell")
        cosmo.compute_transfer(cosmo_params["max_l"])
        cosmo.compute_c_ell()

        # We only should need the shape information for the estimator, but we can add it here
        loc_shape = Shape.prim_local(cosmo_params["ns"], cosmo_params["pivot_scalar"])
        cosmo.add_prim_reduced_bispectrum(loc_shape, self.radii)

        logger.debug("Setting up data and KSW")
        logger.info(
            f"beam {self.beam_ell.shape}, noise {self.noise_ell.shape}, pols {self.pols}"
        )
        self.data = Data(self.lmax, self.noise_ell, self.beam_ell, self.pols, cosmo)
        if self.lensing:
            self.c_ells = cosmo.c_ell["lensed_scalar"]  # type: ignore
        else:
            self.c_ells = cosmo.c_ell["unlensed_scalar"]  # type: ignore
        self.c_ells = self.c_ells["c_ell"][: self.nell]

        self.ksw = KSW(
            self.cosmo.red_bispectra,
            self.icov,
            self.conv_beam_func(),
            self.lmax,
            self.pols,
            self.precision,
        )
        logger.debug("done with KSW")

    def _init_almgen(self):
        """
        Initialize the alm generator attributes.

        This method initializes the alm generator by setting up the alm shape,
        determining whether to force alm generation, and preparing the transfer
        function for the given cosmology.

        Attributes:
            alm_shape (tuple): The shape of the alm array, given by (nsims, npol, nelem).
            force_alm_gen (bool): Whether to force alm generation. Default is False.
            tr_ells (array): The ells values from the transfer function that are less than or equal to lmax.
            tr_k (array): The k values from the transfer function.
            tr_ell_k (array): The tr_ell_k values from the transfer function that correspond to tr_ells.

        Side Effects:
            Modifies the alm_shape, force_alm_gen, tr_ells, tr_k, and tr_ell_k attributes.

        Assumptions:
            Assumes that the cosmology object has generated transfers with ells, k, and tr_ell_k keys.
        """
        self.alm_shape = (self.nsims, self.npol, self.nelem)
        self.force_alm_gen = self._get("force_alm_gen", False)

        tr_ells = self.cosmo.transfer["ells"]  # type: ignore
        mask = tr_ells <= self.lmax
        self.tr_ells = tr_ells[mask]
        self.tr_k = self.cosmo.transfer["k"]  # type: ignore
        self.tr_ell_k = self.cosmo.transfer["tr_ell_k"][mask]  # type: ignore

    def _init_patchgen(self):
        """
        Initialize the patch generator attributes.

        This method initializes the patch generator by setting up the patch side in degrees,
        the number of patches, the total number of patches, and the patch shape.

        Attributes:
            patch_side_deg (int): The side of the patch in degrees. Default is 10.
            npatches (int): The number of patches. Default is 2. Must be even.
            total_patches (int): The total number of patches, given by npatches * total_sims.
            patch_shape (tuple): The shape of the patch array, given by (nsims, npol, npatches, nside, nside).

        Side Effects:
            Modifies the patch_side_deg, npatches, total_patches, and patch_shape attributes.

        Raises:
            AssertionError: If the number of patches is not even.
        """
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

    def conv_beam_func(self):  # change name
        """
        Returns a function which KSW can use to convolve the alms with the beam.
        If beam_width is 0, returns the identity function.

        Returns:
            function: A function that takes alm values and returns the convolved beam.
        """
        if self.beam_width == 0.0:
            # Dont need to bother with anything if beam_width is 0
            return lambda alm: alm

        def __beam(alm):
            # Convolve the beam with the alm values
            ret = np.empty_like(alm)
            for pol in range(alm.shape[0]):
                ret[pol] = hp.almxfl(alm[pol], self.beam_ell[pol], inplace=False)
            return ret

        return __beam

    def _setup_noise_beam(self):
        """
        Set up the noise and beam parameters for the Core object.

        This method initializes the noise and beam parameters based on the configuration settings.
        If the noise parameter is set to False, the noise is set to a very small value to avoid a singular matrix
        in the inverse covariance. The beam width is set to 0 in this case.

        If the noise parameter is set to True, the noise and beam parameters are retrieved from the configuration
        settings. The beam width and noise scales for temperature (TT), E-mode polarization (EE), and
        temperature-E-mode polarization (TE) are converted from muK arcminutes to muK radians.

        The beam and noise ell values are computed based on the polarization settings and stored in the
        `noise_ell` and `beam_ell` attributes.

        Returns:
            None
        """
        self.noise = self._get("noise", True)
        if not self.noise:
            # we cannot set the noise to 0 as this will cause a singular matrix in the inverse covariance
            # so we set it to a very small value
            epsilon_noise = (1e-6 * u.arcmin).to_value(u.radian)
            self.beam_width = 0  # the beam can be 0, no problems

            self.noise_scale_tt = self.noise_scale_ee = self.noise_scale_te = (
                epsilon_noise
            )
            self.noise_ell = np.full(
                (self.npol, self.nell), epsilon_noise**2, dtype=self.r_dtype
            )
            self.beam_ell = np.ones((self.npol, self.nell), dtype=self.r_dtype)
            return

        def convert(x):
            """Helper function to convert from arcmin to radians."""
            return (x * u.arcmin).to_value(u.radian)

        self.beam_width = convert(self._get("beam_width", 0))
        self.noise_scale_tt = convert(self._get("noise_scale_tt", 1))
        self.noise_scale_ee = convert(self._get("noise_scale_ee", 1))
        self.noise_scale_te = convert(self._get("noise_scale_te", 1))

        beam = hp.gauss_beam(self.beam_width, lmax=self.lmax, pol=True)  # (nell, npol)
        beam = np.swapaxes(beam, 0, 1)  # convert beam to (npol, nell)
        noise = np.ones((self.nell), dtype=self.r_dtype)

        # here we setup the noise_ell and beam_ell attributes, these are in TT, EE, TE order
        # Note that B modes are not supported by the KSW code, and only TT has been tested to any extent
        noise_ell = []
        beam_ell = []
        if "T" in self.pols:
            beam_ell.append(beam[0])
            noise_ell.append(noise * self.noise_scale_tt**2)
        if "E" in self.pols:
            beam_ell.append(beam[1])
            noise_ell.append(noise * self.noise_scale_ee**2)
        if self.pols == ["T", "E"]:
            # beam_ell.append(beam[3]) # beam doesnt need to be set for TE
            noise_ell.append(noise * self.noise_scale_te**2)
        self.noise_ell = np.array(noise_ell)
        self.beam_ell = np.array(beam_ell)

    def _setup_radii(self):
        """
        Setup the radii and drs arrays.

        This method sets up the radii and drs arrays based on the r_min and r_max attributes and a predefined set of ranges.
        See Smith and Zaldarriaga (2011) Section 5.2 for more details.

        Side Effects:
            Modifies the radii and drs attributes.
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
