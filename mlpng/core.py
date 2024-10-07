import sys
import argparse
import json
import logging
import os

from attr import dataclass
import healpy as hp
import numpy as np

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

        # This just logs any changes to the defaults, but only if in debug mode
        # Not really needed, but good logs can be helpful
        if logger.isEnabledFor(logging.DEBUG):
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

        # init our core object, split for readability
        self._init()
        self._noise_beam()
        self._radii()
        self._slurm()
        self._paths()

        # save a copy of the settings file iff --save-settings is set
        if args.save_settings:
            dir = os.path.join(self.dirs["base"], "settings")
            file = os.path.join(dir, f"{self.slurm.job}_{self.name}.json")
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

        logger.info(f"Parsing CLI args: {args}")
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
        else:
            if val != default:
                logger.debug(
                    "Found non-default value for '%s': %s (default: %s)",
                    name,
                    repr(val),
                    repr(default),
                )
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
        self.nside = self._get("nside", 1024)
        self.lensing = self._get("lensing", False)
        self.nsims = self._get("nsims", 100)
        self.narray = self._get("narray", 100)
        self.force_gen = self._get("force_generation", False)
        self.force_ksw = self._get("force_ksw", False)
        self.num_estimates = self._get("num_estimates", self.nsims * self.narray)
        self.plot = self._get("plot", True)

        # setup the lmax values
        self.lmin = self._get("lmin", 2)
        self.lmax = self._get("lmax", 3 * self.nside - 1)
        if self.lmax < 300:
            logger.warning(
                "lmax = %s, lmax < 300 not supported. Setting lmax to 300", self.lmax
            )
            self.lmax = 300
        self.cosmo_params["lmax"] = self.lmax

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
        self.isotropic = self._get("isotropic", False)

        self.use_t = "T" in self.pols
        self.use_e = "E" in self.pols
        self.use_b = False  # "B" in self.pols
        if not self.isotropic and self.use_t and self.use_e:
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
        # some parts of the code which interface with cython will use float64, look into
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
            return x * (np.pi / 180 / 60)

        self.noise = self._get("noise", True)
        if not self.noise:
            # we cannot set the noise to 0 as this will cause a singular matrix in the inverse covariance
            # so we set it to a very small value in such a way that we will not get a singular matrix
            # TODO: Consider just using None here and checking for None in the code
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

    def _slurm(self):
        """
        Reads in some important slurm variables into the slurm dictonary.

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
        self.slurm = slurm = Slurm()
        slurm.name = os.getenv("SLURM_JOB_NAME", "unknown")
        slurm.job = int(os.getenv("SLURM_JOB_ID", 0))
        slurm.task_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 0))
        slurm.array = int(os.getenv("SLURM_ARRAY_JOB_ID", 0))

        if slurm.task_count > 0:
            if int(slurm.task_count) != self.narray:
                raise ValueError(
                    f"SLURM_ARRAY_TASK_COUNT {slurm.task_count} does not match narray value {self.narray}. "
                    "Make sure to update the sbatch scripts when changing narray."
                )

            slurm.array_index = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
            slurm.is_main = slurm.array_index == 1
        else:
            slurm.is_main = True

        logger.debug("Running with slurm settings: %s", self.slurm)
        logger.debug("Number of available CPUs: %s", self.n_cpus)

    def _paths(self):
        def join_paths(*args):
            path = os.path.join(self.dirs["base"], *args)
            os.makedirs(path, exist_ok=True)
            return path

        # generate the base name for the files based on settings
        lens = "l" if self.lensing else "ul"
        nn = "nn" if not self.noise else "n"
        pol_str = "".join(self.pols)
        if self.isotropic and self.use_t and self.use_e:
            pol_str = f"i{pol_str}"
        name = self._get("base_name", f"l{self.lmax}_n{self.nside}")
        csims = self.nsims * self.narray  # get the number of simulations
        # simplify the fnl range string if abs(min) and max are the same
        if self.fnl_max == abs(self.fnl_min):
            fstr = f"{self.fnl_max}"
        else:
            fstr = f"{self.fnl_min}-{self.fnl_max}"
        # finally we build our string
        self.name = f"{name}_{lens}_{nn}_{pol_str}x{csims}_f{fstr}"

        self.dirs = {}
        self.dirs["base"] = self._get("base_dir", "data")
        self.dirs["plot"] = join_paths(self._get("plot_dir", "plots"))
        self.dirs["data"] = join_paths(self._get("data_dir", "data"))

        tstr = f"_{self.slurm.array_index}" if self.slurm.array_index > 0 else ""
        self.file = os.path.join(self.dirs["data"], f"{self.name}{tstr}.hdf5")
        logger.debug("Using data file: %s", self.file)

        if self.force_ksw:
            self.dirs["mc"] = join_paths(self._get("mc_dir", "kswmc"))
            self.mc_file = os.path.join(self.dirs["mc"], f"{self.name}.hdf5")
            logger.debug("Using KSW MC file: %s", self.mc_file)

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

    def get_plot_file(self, name, dir=None, extension=".png", create_dir=True):
        if dir is None:
            dir = os.path.join(self.dirs["plot"], str(self.name))

        if not os.path.exists(dir):
            if create_dir:
                os.makedirs(dir, exist_ok=True)
            else:
                raise ValueError(f"Directory {dir} does not exist")

        return os.path.join(dir, f"{self.name}_{self.slurm.job}_{name}{extension}")


@dataclass
class Slurm:
    name: str = "unknown"
    job: int = 0
    task_count: int = 0
    array: int = 0
    array_index: int = 0
    is_main: bool = False
