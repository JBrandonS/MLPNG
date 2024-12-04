import math
import sys
import argparse
import json
import logging
import os

import healpy as hp
import numpy as np

from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Core:
    """
    The `Core` class represents the core functionality of the MLPNG (Machine Learning for Primordial Non-Gaussianity) project.
    It handles parsing command line arguments, loading settings from a file, setting up cosmological parameters, and initializing various components for the run.

    Attributes:
        settings (dict): A dictionary containing the settings loaded from the settings file.
        cosmo_params (dict): A dictionary containing the cosmological parameters.
        lmin (int): The minimum multipole moment for the run. Default: 2
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
        fnl_min (float): The minimum value of the local non-Gaussianity parameter (fnl). Default: -100
        fnl_max (float): The maximum value of the local non-Gaussianity parameter (fnl). Default: 100
        nell (int): The number of ell values.
        nelem (int): The number of elements in the alm array.
        ells (numpy.ndarray): An array of ell values.
        r_dtype (numpy.dtype): The data type for real numbers. Default: float32
        c_dtype (numpy.dtype): The data type for complex numbers. Default: complex64
        precision (str): The precision level ("single" or "double"). Default: 'single'
        dirs (dict): A directory containing (key: path) pairs of directories.
        file (str): The partial path to the data file for the job to use.
        radii (numpy.ndarray): An array of radii for the bispectrum estimator.
        force_gen (bool): Whether to force the generation of the data, i.e. remove any existing data files instead of exiting.
        force_ksw (bool): Whether to force the KSW estimator to run otherwise can use save states or iso methods.
        num_estimates (int): The number of estimates to compute. Default: The total number of simulations.
        plot (bool): Whether to plot the results, will only be used if is_main. Default: True
        slurm (Slurm): A data object containing some SLURM job settings for reference.
    """

    # just typing info here
    lmin: int
    lmax: int
    nside: int
    pols: str | tuple
    lensing: bool
    npol: int
    seed: int
    rng: np.random.Generator
    nsims: int
    narray: int
    total_sims: int
    """Total simulations for the complete experiment."""
    fnl_min: float
    fnl_max: float
    nell: int
    nelem: int
    ells: np.ndarray
    r_dtype: type
    c_dtype: type
    precision: str
    file: str
    radii: np.ndarray
    c_ell: np.ndarray
    n_ell: np.ndarray
    b_ell: np.ndarray
    s_ell: np.ndarray
    # cosmo: Any
    # estimator: Any

    def __init__(self, argv=None):
        """
        Initializes a new instance of the `Core` class.

        Args:
            argv (list): List of the CLI Args
            If None, sys.argv will be used.
        """

        # handle the CLI args first
        args = self.parse_args(argv)

        # We start by loading in the settings from the provided file
        # From there we store the settings in the settings dict attribute
        # We then override the settings with the command line arguments
        # We then set the cosmological parameters to the defaults and override them with the settings
        logger.info("Loading settings from file '%s'", args.settings_file)
        with open(args.settings_file, "r", encoding="utf-8") as f:
            settings = self.settings = json.load(f)

        # replace the settings with the command line arguments
        for key, value in vars(args).items():
            # we do not want to save the settings_file or save_settings options, so ignore those
            if value is not None and key not in ["settings_file", "save_settings"]:
                logger.debug("Forcing setting '%s' to %s due to CLI", key, value)
                settings[key] = value

        # set the cosmological parameters defaults, and update based on cosmo_params
        cosmo_params = self._get("cosmo_params", cosmo_defaults())
        self.cosmo_params = settings["cosmo_params"] = {
            **cosmo_defaults(),
            **cosmo_params,
        }

        # This just logs any changes to the defaults, but only if in debug mode
        # Not really needed, but good logs can be helpful
        if logger.isEnabledFor(logging.DEBUG):
            overridden_params = {
                key: (value, cosmo_params[key])
                for key, value in cosmo_defaults().items()
                if key in cosmo_params and value != cosmo_params[key]
            }

            for key, (default_value, overridden_value) in overridden_params.items():
                logger.debug(
                    "Overriding cosmo param '%s' from %s to %s",
                    key,
                    default_value,
                    overridden_value,
                )

        logger.info("Running with settings: \n%s", json.dumps(settings, indent=2))

        # init our core object, split for readability
        self._init()
        self._noise_beam()
        self._radii()
        self._slurm()
        self._paths()

        # save a copy of the settings file iff --save-settings is set
        if args.save_settings:
            base_dir = os.path.join(self.dirs["base"], "settings")
            file = os.path.join(base_dir, f"{self.slurm.job}_{self.name}.json")
            os.makedirs(base_dir, exist_ok=True)

            if not os.path.exists(file):
                logger.info("Saving run settings to file: '%s'", file)
                with open(file, "w", encoding="utf-8") as f:
                    json.dump(self.settings, f, indent=2)
            else:
                logger.warning("Settings already exists: '%s', not overwriting", file)

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
        parser.add_argument("--lmin", type=int)
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
        parser.add_argument("--double_precision", action=argparse.BooleanOptionalAction)
        parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
        parser.add_argument("--isotropic", action=argparse.BooleanOptionalAction)

        # this allows us to save a copy of the final settings used for the run
        # only really useful for debugging, must be provided by the CLI and not in the settings file
        parser.add_argument("--save_settings", action="store_true")

        if args is None:
            args = sys.argv[1:]

        logger.info("Parsing CLI args: %s", args)
        pargs, _ = parser.parse_known_args(args)
        return pargs

    def _get(self, name, default):
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
            logger.debug(
                "Setting '%s' not found, using default: %s", name, repr(default)
            )
            return default

        if val != default:
            logger.debug(
                "Found non-default value for '%s': %s (default: %s)",
                name,
                repr(val),
                repr(default),
            )
        else:
            logger.debug("Found default value for '%s': %s", name, repr(val))
        return val

    def _init(self):
        """
        Initialize the core parameters and settings for the simulation.
        This method sets up various parameters required for the simulation, including
        random number generation, main parameters, lmax values, fnl values, polarization
        settings, precision types, derived parameters, patch settings, and CPU settings.

        Attributes:
            seed (int): Seed for random number generation.
            rng (np.random.Generator): Random number generator.
            nside (int): Resolution parameter for HEALPix.
            lensing (bool): Flag to enable or disable lensing.
            nsims (int): Number of simulations to run.
            narray (int): Number of arrays.
            force_gen (bool): Flag to force generation.
            force_ksw (bool): Flag to force KSW.
            num_estimates (int): Number of estimates.
            plot (bool): Flag to enable or disable plotting.
            lmin (int): Minimum multipole moment.
            lmax (int): Maximum multipole moment.
            fnl_min (int): Minimum fnl value.
            fnl_max (int): Maximum fnl value.
            pols (tuple): Tuple of polarization modes.
            npols (int): Number of polarization modes.
            isotropic (bool): Flag to enable or disable isotropic mode.
            use_t (bool): Flag to use T polarization.
            use_e (bool): Flag to use E polarization.
            use_b (bool): Flag to use B polarization.
            use_te (bool): Flag to use TE polarization.
            r_dtype (type): Real number precision type.
            c_dtype (type): Complex number precision type.
            precision (str): Precision type ('single' or 'double').
            total_sims (int): Total number of simulations.
            nell (int): Number of multipole moments.
            nelem (int): Number of elements in Alm array.
            ells (np.ndarray): Array of multipole moments.
            alm_shape (tuple): Shape of the Alm array.
            patch_side_deg (int): Side length of patches in degrees.
            npatches (int): Number of patches.
            total_patches (int): Total number of patches.
            patch_shape (tuple): Shape of the patch array.
            n_cpus (int): Number of CPUs available.
        Raises:
            AssertionError: If the number of patches is not even.
        """

        # we setup a RNG here for reproducibility
        # TODO: needs more implementation, and testing of reproducibility, need to setup tensorflow seed and probably others
        # overall, dont expect reproducibility, not a high priority
        self.seed = self._get("seed", np.random.default_rng().integers(0, 2**32 - 1))
        self.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

        # main parameters
        self.nside = self._get("nside", 128)
        self.lensing = self._get("lensing", True)
        self.nsims = self._get("nsims", 100)
        self.ndups = self._get("ndups", 25)
        self.narray = self._get("narray", 100)
        self.force_gen = self._get("force_generation", False)
        self.force_ksw = self._get("force_ksw", False)
        self.num_estimates = self._get(
            "num_estimates", min(self.nsims * self.narray, 300)
        )
        self.plot = self._get("plot", True)
        self.save_alms = self._get("save_alms", True)

        # setup the lmax values
        self.lmin = self._get("lmin", 2)
        self.lmax = self._get("lmax", 3 * self.nside - 1)
        if self.lmax < 300:
            logger.warning(
                "lmax = %s, lmax < 300 not supported. Setting lmax to 300", self.lmax
            )
            self.lmax = 300
        self.cosmo_params["lmax"] = self.lmax + self._get("lmax_buffer", 128)

        # setup the fnl values
        self.fnl_min, self.fnl_max = self._get("fnl_range", (-1000, 1000))

        # setup the polarization which should support --pols [T|E|TE]
        pols = self._get("pols", "T")
        if isinstance(pols, str):
            pols = tuple(pols)
        elif isinstance(pols, list):
            pols = tuple(pols)
        # we do not want to support B mode since it is so small, so lets remove anything with B in it.
        self.pols = tuple(c for p in pols for c in p if p != "B")
        self.npols = len(self.pols)

        self.use_t = "T" in self.pols
        self.use_e = "E" in self.pols
        self.use_b = False  # "B" in self.pols
        self.isotropic = self._get("isotropic", not (self.use_t and self.use_e))
        if self.isotropic:
            self.use_te = False
        else:
            self.use_te = True

        logger.debug(
            "Using polarizations T: %s, E: %s, TE: %s",
            self.use_t,
            self.use_e,
            self.use_te,
        )

        # setup our precision types to be consistent
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
        self.total_sims = self.ndups * self.nsims * self.npols * self.narray

        # setup some info parameters
        self.nell = self.lmax + 1
        self.npix = hp.nside2npix(self.nside)
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

        # here we just get the number of cpus, but read in from SLURM if available
        # os.sched_getaffinity(0) gets the number of usable CPUs available, this is different from
        # os.cpu_count() which gets the number of CPUs on the system
        self.n_cpus = int(
            os.getenv("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0)))
        )

        # We need the number of patches to be even for the healpy code
        assert self.npatches % 2 == 0, "Number of patches must be even"

    def _noise_beam(self):
        """
        Set up the noise and beam parameters for the Core object.

        This method initializes the noise and beam parameters based on the configuration settings.
        If the noise parameter is set to False, then noise_ell is set to none

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
            # converts from arcmin to radians
            return x * (np.pi / 180 / 60)

        self.noise = self._get("noise", True)
        if self.noise:
            fwhm = convert(self._get("beam_width", 1))
            b_ell = hp.sphtfunc.gauss_beam(fwhm, self.lmax, True).T

            # T, E, B, TE
            n_ell = np.ones((4, self.nell), dtype=self.r_dtype)
            n_ell[0] = convert(self._get("noise_tt", 5)) ** 2
            n_ell[1] = convert(self._get("noise_ee", 5)) ** 2
            n_ell[2] = 0  # B is always 0
            n_ell[3] = convert(self._get("noise_te", 25)) ** 2
        else:
            n_ell = np.full((4, self.nell), 1e-16, dtype=self.r_dtype)
            b_ell = np.ones((4, self.nell), dtype=self.r_dtype)

        self.n_ell = n_ell[self.pol_idxs(keep_b=True, keep_te=True)]
        self.b_ell = b_ell[self.pol_idxs(keep_b=True, keep_te=True)]

    def _radii(self):
        """
        Setup the radii and drs arrays.

        This method sets up the radii and drs arrays based on the r_min and r_max attributes and a predefined set of ranges.
        See Smith and Zaldarriaga (2011) Section 5.2 for more details.

        Attributes:
            radii (numpy.ndarray): An array of radii for the bispectrum estimator.
        """
        #    start,  stop, resolution
        ranges = [
            (1e-3, 9500, 150),
            (9500, 11000, 300),
            (11000, 13800, 150),
            (13800, 14600, 400),
            (14600, 16000, 100),
            (16000, 50000, 100),
        ]

        radii = []
        for r in ranges:
            start, end, num = r

            if r == ranges[-1]:  # For the last range, use logspace
                temp_radii = np.logspace(np.log10(start), np.log10(end), num, True)
            else:
                temp_radii = np.linspace(start, end, num, False)

            radii.extend(temp_radii)

        r_min = int(self._get("r_min", 0))
        r_max = int(self._get("r_max", 50000))
        self.radii = np.array([r for r in radii if r_min <= r <= r_max])

    def _slurm(self):
        """
        Reads in some important slurm variables into the slurm dictionary.

        Attributes:
            slurm (dict): A dictionary containing the SLURM job settings.
                name (str): The name of the SLURM job.
                job (str): The SLURM job ID.
                task_count (str): The SLURM array task count.
                array (str): The SLURM array job ID.
                array_index (int): The SLURM array task ID.
                is_main (bool): Whether the current job is the main job.

        Raises:
            ValueError: If the SLURM_ARRAY_TASK_COUNT does not match the narray value.
        """
        array_index = int(os.getenv("SLURM_ARRAY_TASK_ID", default="-1"))
        self.slurm = slurm = Slurm(
            name=os.getenv("SLURM_JOB_NAME", "unknown"),
            job=int(os.getenv("SLURM_JOB_ID", "0")),
            task_count=int(os.getenv("SLURM_ARRAY_TASK_COUNT", "0")),
            array=int(os.getenv("SLURM_ARRAY_JOB_ID", "0")),
            array_index=array_index,
            is_main=array_index in {1, -1},
        )

        if slurm.task_count > 0 and slurm.task_count != self.narray:
            raise ValueError(
                f"SLURM_ARRAY_TASK_COUNT {slurm.task_count} does not match narray value {self.narray}. "
                "Make sure to update the sbatch scripts when changing narray."
            )

        logger.debug("Running with slurm settings: %s", slurm)
        logger.debug("Number of available CPUs: %s", self.n_cpus)

    def _paths(self):
        """
        Generates and sets up directory paths and filenames based on the object's attributes.
        This method constructs various directory paths and filenames required for the simulation
        and analysis based on the object's configuration. It ensures that the necessary directories
        exist and constructs filenames that incorporate various settings such as lensing, noise,
        polarization, and simulation parameters.
        The following directories are created and stored in the `self.dirs` dictionary:
        - base: The base directory for data storage.
        - plot: The directory for storing plot files.
        - data: The directory for storing data files.
        - mc: The directory for storing Monte Carlo files (if `force_ksw` is True).
        The following filenames are constructed and stored:
        - self.file: The main data file.
        - self.mc_file: The Monte Carlo file (if `force_ksw` is True).
        The filenames incorporate various settings such as:
        - `lmax` and `nside` for resolution.
        - `lensing` and `noise` settings.
        - Polarization settings.
        - Number of simulations.
        - `fnl` range.
        - SLURM array index (if applicable).
        Logging is used to debug the paths and filenames being used.
        Raises:
            OSError: If there is an issue creating the directories.
        """

        def join_paths(*args):
            path = os.path.join(self.dirs["base"], *args)
            os.makedirs(path, exist_ok=True)
            return path

        # generate some base strings the files based on settings
        csims = self.nsims * self.narray  # get the number of simulations
        lens = "l" if self.lensing else "ul"
        nn = "nn" if not self.noise else "n"
        pol_str = "".join(self.pols)
        tstr = f"_{self.slurm.array_index}" if self.slurm.array_index > 0 else ""

        # simplify the fnl range string if abs(min) and max are the same
        if self.fnl_max == abs(self.fnl_min):
            fstr = f"{self.fnl_max}"
        else:
            fstr = f"{self.fnl_min}-{self.fnl_max}"

        # finally we build our string
        name = self._get("base_name", f"l{self.lmax}_n{self.nside}")
        self.name = f"{name}_{pol_str}_{csims}x{self.ndups}_f{fstr}"

        self.dirs = {}
        self.dirs["base"] = self._get("base_dir", "data")
        self.dirs["plot"] = join_paths(self._get("plot_dir", "plots"))
        self.dirs["data"] = join_paths(self._get("data_dir", "data"))
        self.dirs["tb"] = join_paths(self._get("tb_dir", "tensorboard"))
        self.dirs["model"] = join_paths(self._get("model_dir", "models"))

        self.file = os.path.join(self.dirs["data"], f"{self.name}{tstr}.hdf5")
        logger.debug("Using data file: %s", self.file)

        self.dirs["mc"] = join_paths(self._get("mc_dir", "kswmc"))
        self.mc_file = os.path.join(self.dirs["mc"], f"{self.name}.hdf5")
        self.mc_steps = self._get("mc_steps", 100)
        logger.debug(
            "Using KSW MC file: %s, with nsteps %s", self.mc_file, self.mc_steps
        )

    def init_estimator(self, verbose=False):
        """
        Initializes the estimator for cosmological parameter estimation.
        This method sets up the cosmological parameters, computes the transfer functions,
        and initializes the KSW estimator for bispectrum analysis.
        Parameters:
        -----------
            verbose : bool, optional
                If True, enables verbose logging for debugging purposes. Default is False.
        Attributes:
        -----------
            cosmo : Cosmology
                An instance of the Cosmology class initialized with the given parameters.
            c_ell : ndarray
                The computed C_ell values, either lensed or unlensed, based on the configuration.
            s_ell : ndarray
                The signal C_ell values with noise added and monopole/dipole terms removed.
            estimator : KSW
                An instance of the KSW estimator initialized with the computed bispectra.
        """

        # we do local imports since this will not work on the superpod due to mpi issues, but we dont need ksw there anyways
        # pylint: disable=C0415
        import camb
        from ksw import Cosmology, Shape, KSW

        camb_params = camb.set_params(**self.cosmo_params, verbose=verbose)

        if verbose:
            logger.debug(camb_params)
            logger.debug(camb.get_results(camb_params))

        self.cosmo: Cosmology = Cosmology(camb_params, verbose)
        self.cosmo.compute_transfer(self.lmax + 128, verbose)
        self.cosmo.compute_c_ell()

        if self.lensing:
            c_ell = self.cosmo.c_ell["lensed_scalar"]["c_ell"].T
        else:
            c_ell = self.cosmo.c_ell["unlensed_scalar"]["c_ell"].T
        self.c_ell = c_ell[self.pol_idxs(keep_b=True, keep_te=True), : self.nell]
        self.c_ell = self.c_ell.astype(self.r_dtype)

        ns = self.cosmo_params["ns"]
        ps = self.cosmo_params["pivot_scalar"]
        shape = Shape.prim_local(ns, pivot=ps)
        self.cosmo.add_prim_reduced_bispectrum(shape, self.radii)

        self.estimator: KSW = KSW(
            self.cosmo.red_bispectra,
            lambda a: a,  # we will send the alms directly
            self.lmax,
            self.pols,
            self.precision,
        )

    def pol_idxs(self, keep_b=False, keep_te=False, pretrimmed=False):
        """
        Get the polarization indices based on the configuration settings.

        Parameters:
            keep_b (bool): Whether to keep the B-mode polarization. Default is False.
            keep_te (bool): Whether to keep the TE-mode polarization. Default is False.
            pretrimmed (bool): Whether the alms are pretrimmed, i.e. have we removed the t-modes. Default is False.
        """
        # TODO: Look into simplifying this method
        start = 0 if self.use_t else (0 if pretrimmed else 1)

        num_to_take = 1 if self.use_t else 0
        if self.use_e:
            num_to_take += 2 if keep_b else 1  # we also need B
        if keep_te and self.use_te:
            num_to_take += 1

        return np.array(range(start, start + num_to_take))

    def get_plot_file(self, name, base_dir=None, extension=".png", create_dir=True):
        """
        Generate the file path for a plot image.
        Parameters:
            name (str): The name of the plot file.
            base_dir (str, optional): The base directory where the plot file will be saved.
                        Defaults to None, which sets it to a subdirectory
                        under the 'plot' directory named after the instance's name.
            extension (str, optional): The file extension for the plot file. Defaults to ".png".
            create_dir (bool, optional): Whether to create the directory if it does not exist.
                            Defaults to True.
        Returns:
            str: The full file path for the plot image.
        Raises:
            ValueError: If the directory does not exist and `create_dir` is set to False.
        """

        if base_dir is None:
            base_dir = os.path.join(self.dirs["plot"], str(self.name))

        if not os.path.exists(base_dir):
            if create_dir:
                os.makedirs(base_dir, exist_ok=True)
            else:
                raise ValueError(f"Directory {base_dir} does not exist")

        return os.path.join(base_dir, f"{self.name}_{self.slurm.job}_{name}{extension}")


@dataclass
class Slurm:
    """
    A class to represent a Slurm job configuration.
    Attributes:
        name (str): The name of the job. Default is "unknown".
        job (int): The job ID. Default is 0.
        task_count (int): The number of tasks. Default is 0.
        array (int): The array job ID. Default is 0.
        array_index (int): The index of the array job. Default is 0.
        is_main (bool): Indicates if this is the main job. Default is False.
    """

    name: str = "unknown"
    job: int = 0
    task_count: int = 0
    array: int = 0
    array_index: int = 0
    is_main: bool = False


def cosmo_defaults():
    """Some default settings that are required by KSW / camb."""
    return {"As": 2.13e-09, "ns": 0.9624, "pivot_scalar": 0.05}


def get_itotcov_ell(icov_signal_ell, icov_noise_ell=None, b_ell=None):
    """
    Combine signal and noise power spectra into total inverse
    isotropic covariance: S^-1 (S^-1 + B N^-1 B)^-1 B N^-1 B
    = (S + B^-1 N B^-1)^-1.

    Parameters
    ----------
    icov_signal_ell : (npol, npol, nell) or (npol, nell) array
        Inverse signal covariance
    icov_noise_ell : (npol, npol, nell) or (npol, nell) array
        Inverse noise covariance matrix
    b_ell : (npol, nell) array
        Beam transfer function.

    Returns
    -------
    itotcov_ell : (npol, npol, nell) array
        Total inverse covariance matrix.
    """
    if icov_noise_ell is None:
        return icov_signal_ell.copy()

    # Check if icov_signal_ell is (npol, nell) and convert to (npol, npol, nell)
    if icov_signal_ell.ndim == 2:
        npol, _ = icov_signal_ell.shape
        icov_signal_ell = (
            icov_signal_ell[:, np.newaxis, :] * np.eye(npol)[:, :, np.newaxis]
        )

    # Check if icov_noise_ell is (npol, nell) and convert to (npol, npol, nell)
    if icov_noise_ell.ndim == 2:
        npol, _ = icov_noise_ell.shape
        icov_noise_ell = (
            icov_noise_ell[:, np.newaxis, :] * np.eye(npol)[:, :, np.newaxis]
        )

    if b_ell is not None:
        b_ell = b_ell * np.eye(b_ell.shape[0])[:, :, np.newaxis]
        in_mat = np.einsum("ijl, jkl, kol -> iol", b_ell, icov_noise_ell, b_ell)
    else:
        in_mat = icov_noise_ell

    imat = np.linalg.inv((icov_signal_ell + in_mat).T).T
    itotcov_ell = np.einsum("ijl, jkl -> ikl", imat, in_mat)
    itotcov_ell = np.einsum("ijl, jkl -> ikl", icov_signal_ell, itotcov_ell)
    return itotcov_ell
