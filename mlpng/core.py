import sys
import argparse
import json
import logging
import os
import h5py
import healpy as hp
import numpy as np

from .utils import Slurm, setup_logging


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

    # just typing info here, good for using pylance
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
    fnl_min: int
    fnl_max: int
    nell: int
    nelem: int
    npix: int
    ells: np.ndarray
    r_dtype: type
    c_dtype: type
    precision: str
    file: str
    radii: np.ndarray
    c_ell: np.ndarray
    n_ell: np.ndarray
    b_ell: np.ndarray
    shapes: list[str]

    _cosmo_defaults = {"As": 2.13e-09, "ns": 0.9624, "pivot_scalar": 0.05}

    def __init__(self, argv=None, log_level=logging.DEBUG):
        """
        Initializes a new instance of the `Core` class.

        Args:
            argv (list): List of the CLI Args
            If None, sys.argv will be used.
        """
        self.logger = setup_logging("mlpng.core", level=log_level)

        # init our core object, split for readability
        self._process_settings(argv)

        self._init()
        self._paths()
        self._noise_beam()
        self._radii()

    def _parse_args(self, args=None):
        """
        Parse command line arguments.

        Args:
            args (list): List of command line arguments. If None, sys.argv will be used.

        Returns:
            argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser()

        # only required argument is the settings file
        parser.add_argument("settings_file", help="Path to the settings file")

        # some standard arguments here, we can add more as needed
        parser.add_argument(
            "--nsims",
            type=int,
            help="Number of simulations to run, per array job",
        )
        parser.add_argument(
            "--narray",
            type=int,
            help="Number of arrays to process",
        )
        parser.add_argument(
            "--nside",
            type=int,
            help="Resolution parameter for HEALPix",
        )
        parser.add_argument(
            "--ndups",
            type=int,
            help="Number of duplicates",
        )
        parser.add_argument(
            "--base_dir",
            type=str,
            help="Base directory for data storage",
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Random seed for reproducibility",
        )
        parser.add_argument(
            "--base_name",
            type=str,
            help="Base name for the output files",
        )
        parser.add_argument(
            "--lmin",
            type=int,
            help="Minimum multipole moment",
        )
        parser.add_argument(
            "--lmax",
            type=int,
            help="Maximum multipole moment",
        )
        parser.add_argument(
            "--lmax_buffer",
            type=int,
            help="Buffer for the maximum multipole moment for some camb calculations",
        )
        parser.add_argument(
            "--fnl_range",
            type=int,
            nargs=2,
            help="Range for the fnl parameter in min max, i.e. --fnl_range -100 100",
        )
        parser.add_argument(
            "--num_estimates",
            type=int,
            help="Number of estimates to compute, if using --estimate",
        )
        parser.add_argument(
            "--pols",
            type=str,
            help="Polarizations to consider, options are T, E, TE; Right now only T is fully tested",
        )
        parser.add_argument(
            "--phi_scale",
            type=float,
            help="Scale for the lensing maps",
        )
        parser.add_argument(
            "--shapes",
            choices=["local", "equilateral", "orthogonal", "all"],
            nargs="+",
            help="Specify the shape type. Must be any of: local, equilateral, orthogonal, or all. Multiple shapes can be specified.",
        )

        # for BooleanOptionalAction: --flag will set the value `flag` to True, --no-flag will set `flag` to False
        parser.add_argument(
            "--lensing",
            action=argparse.BooleanOptionalAction,
            help="Enable or disable lensing",
        )
        parser.add_argument(
            "--noise",
            action=argparse.BooleanOptionalAction,
            help="Enable or disable noise",
        )
        parser.add_argument(
            "--force_generation",
            action=argparse.BooleanOptionalAction,
            help="Force data generation, delete files if existing",
        )
        parser.add_argument(
            "--estimate",
            action=argparse.BooleanOptionalAction,
            help="Use KSW estimator to run tests",
        )
        parser.add_argument(
            "--double_precision",
            action=argparse.BooleanOptionalAction,
            help="Use double precision",
        )
        parser.add_argument(
            "--plot",
            action=argparse.BooleanOptionalAction,
            help="Enable or disable plotting",
        )
        parser.add_argument(
            "--save_alms",
            action=argparse.BooleanOptionalAction,
            help="Save the combined alms, otherwise will need to create them on the fly",
        )
        parser.add_argument(
            "--save_ksw",
            action=argparse.BooleanOptionalAction,
            help="Save the ksw states",
        )

        # ML trainer settings
        parser.add_argument(
            "--data_fraction",
            type=float,
            help="Fraction of data to use for training",
        )
        parser.add_argument(
            "--wandb",
            action=argparse.BooleanOptionalAction,
            help="Enable or disable Weights & Biases logging",
        )
        parser.add_argument(
            "--tensorboard",
            action=argparse.BooleanOptionalAction,
            help="Enable or disable TensorBoard logging",
        )

        # this allows us to save a copy of the final settings used for the run
        # only really useful for debugging, must be provided by the CLI and not in the settings file
        parser.add_argument(
            "--save_settings",
            action="store_true",
            help="Save a copy of the final settings used for the run",
        )

        if args is None:
            args = sys.argv[1:]

        self.logger.debug("Parsing CLI args: %s", args)
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
            # logger.debug(
            #     "Setting '%s' not found, using default: %s", name, repr(default)
            # )
            return default

        if val != default:
            # I dont like the cosmo_params printing, so we will ignore it
            if name != "cosmo_params":
                self.logger.debug(
                    "Found non-default value for '%s': %s (default: %s)",
                    name,
                    repr(val),
                    repr(default),
                )
        # else:
        #     self.logger.debug("Found default value for '%s': %s", name, repr(val))
        return val

    def _process_settings(self, argv):
        args = self._parse_args(argv)

        # We start by loading in the settings from the provided file
        # From there we store the settings in the settings dict attribute
        # We then override the settings with the command line arguments
        # We then set the cosmological parameters to the defaults and override them with the settings
        self.logger.info("Loading settings from file '%s'", args.settings_file)
        with open(args.settings_file, "r", encoding="utf-8") as f:
            settings = self.settings = json.load(f)

        # replace the settings with the command line arguments
        for key, value in vars(args).items():
            # we do not want to save the settings_file or save_settings options, so ignore those
            if value is not None and key not in ["settings_file", "save_settings"]:
                self.logger.debug("Forcing setting '%s' to %s due to CLI", key, value)
                settings[key] = value

        # set the cosmological parameters defaults, and update based on cosmo_params
        cosmo_params = self._get("cosmo_params", self._cosmo_defaults)
        self.cosmo_params = settings["cosmo_params"] = {
            **self._cosmo_defaults,
            **cosmo_params,
        }

        # This just logs any changes to the defaults, but only if in debug mode
        # Not really needed, but good logs can be helpful
        if self.logger.isEnabledFor(logging.DEBUG):
            overridden_params = {
                key: (value, cosmo_params[key])
                for key, value in self._cosmo_defaults.items()
                if key in cosmo_params and value != cosmo_params[key]
            }

            for key, (default_value, overridden_value) in overridden_params.items():
                self.logger.debug(
                    "Overriding cosmo param '%s' from %s to %s",
                    key,
                    default_value,
                    overridden_value,
                )

        self.logger.debug("Running with settings: \n%s", json.dumps(settings, indent=2))

        # save a copy of the settings file iff --save_settings is set
        if args.save_settings:
            base_dir = os.path.join(self.dirs["base"], "settings")
            file = os.path.join(base_dir, f"{self.slurm.job}_{self.name}.json")
            os.makedirs(base_dir, exist_ok=True)

            if not os.path.exists(file):
                self.logger.info("Saving run settings to file: '%s'", file)
                with open(file, "w", encoding="utf-8") as f:
                    json.dump(self.settings, f, indent=2)
            else:
                self.logger.warning(
                    "Settings already exists: '%s', not overwriting", file
                )

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
        self.slurm = Slurm()
        self.logger.debug("slurm: %s", self.slurm)

        self.seed = self._get("seed", np.random.randint(1, 2**30))
        self.seed += self.slurm.array_index
        self.logger.debug("Using seed: %s", self.seed)
        self.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

        # main parameters
        self.nside = self._get("nside", 128)
        self.lensing = self._get("lensing", True)
        self.nsims = self._get("nsims", 100)
        self.ndups = self._get("ndups", 25)
        self.narray = self._get("narray", self.slurm.task_count or 1)
        self.fnl_min, self.fnl_max = self._get("fnl_range", (-1000, 1000))

        self.shapes = self._get("shapes", "all")
        if isinstance(self.shapes, str):
            self.shapes = [self.shapes]
        if "all" in self.shapes:
            self.shapes = ["local", "equilateral", "orthogonal"]

        self.phi_scale = self._get("phi_scale", 1)

        self.force_gen = self._get("force_generation", False)
        self.mc_steps = self._get("mc_steps", 300)
        self.estimate = self._get("estimate", True)
        self.num_estimates = self._get(
            "num_estimates",
            min(self.nsims * self.narray, 1000),
        )

        self.plot = self._get("plot", True)
        self.save_alms = self._get("save_alms", False)
        self.save_ksw = self._get("save_ksw", False)

        if self.slurm.task_count > 0 and self.slurm.task_count != self.narray:
            raise ValueError(
                f"SLURM_ARRAY_TASK_COUNT {self.slurm.task_count} does not match narray value {self.narray}. "
                "Make sure to update the sbatch scripts when changing narray."
            )

        # setup the lmax values
        self.lmin = self._get("lmin", 2)
        self.lmax = self._get("lmax", 3 * self.nside - 1)
        self.lmax_buffer = self._get("lmax_buffer", 128)
        self.cosmo_params["lmax"] = self.lmax
        self.nell = self.lmax + 1
        self.ells = np.arange(self.nell)

        # setup the polarization which should support --pols [T|TE]
        self.pols = list(self._get("pols", "T").upper())
        self.npols = len(self.pols)
        self.use_t = "T" in self.pols
        self.use_e = "E" in self.pols
        self.use_b = False  # "B" in self.pols
        self.use_pols = self.pols == 3  # used for hp commands

        self.use_wandb = self._get("wandb", False)
        self.use_tb = self._get("tensorboard", False)

        # setup our precision types to be consistent
        if self._get("double_precision", False):
            self.r_dtype = np.float64
            self.c_dtype = np.complex128
            self.precision = "double"
        else:
            self.r_dtype = np.float32
            self.c_dtype = np.complex64
            self.precision = "single"

        # setup some derived parameters
        self.total_sims = self.nsims * self.narray

        # setup some info parameters
        self.npix = hp.nside2npix(self.nside)
        self.nelem = hp.Alm.getsize(self.lmax)
        self.alm_shape = (self.nsims, self.npols, self.nelem)
        self.map_shape = (self.nsims, self.npols, self.npix)

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
            n_ell[0] = convert(self._get("noise_tt", 1)) ** 2
            n_ell[1] = convert(self._get("noise_ee", 5)) ** 2
            n_ell[2] = 0  # B is always 0
            n_ell[3] = 0  # no TE noise
        else:
            n_ell = np.full((4, self.nell), 1e-16, dtype=self.r_dtype)
            b_ell = np.ones((4, self.nell), dtype=self.r_dtype)

        self.n_ell = n_ell
        self.b_ell = b_ell

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
        total_sims = self.nsims * self.narray  # get the number of simulations
        pol_str = "".join(self.pols)
        tstr = f"_{self.slurm.array_index}" if self.slurm.array_index > 0 else ""

        # finally we build our string
        base_name = f"l{self.lmax}_n{self.nside}"
        base = self._get("base_name", base_name)
        if base.startswith("+"):
            base = f"{base_name}{base[1:]}"
        self.name = f"{base}_{pol_str}_{total_sims}_p{self.phi_scale}"  # _f{fstr}"

        self.dirs = {}
        self.dirs["base"] = self._get("base_dir", "data")
        self.dirs["plot"] = join_paths(self._get("plot_dir", "plots"))
        self.dirs["data"] = join_paths(self._get("data_dir", "data"))
        self.dirs["tb"] = join_paths(self._get("tb_dir", "tensorboard"))
        self.dirs["model"] = join_paths(self._get("model_dir", "models"))
        self.dirs["mc"] = join_paths(self._get("mc_dir", "kswmc"))
        self.dirs["wandb"] = join_paths(self._get("wandb_dir", "wandb"))

        self.file = os.path.join(self.dirs["data"], f"{self.name}{tstr}.hdf5")
        self.mc_file = os.path.join(self.dirs["mc"], f"{self.name}.hdf5")

    def pol_idxs(self, keep_b=False, keep_te=False, pretrimmed=False):
        """
        Get the polarization indices based on the configuration settings.

        Parameters:
            keep_b (bool): Whether to keep the B-mode polarization. Default is False.
            keep_te (bool): Whether to keep the TE-mode polarization. Default is False.
            pretrimmed (bool): Whether the alms are pretrimmed, i.e. have we removed the t-modes. Default is False.
        """
        conditions = [
            self.use_t,
            self.use_e,
            keep_b and self.use_e,
            keep_te and (self.use_t and self.use_e),  # and not self.isotropic),
        ]
        idxs = np.array([i for i, v in enumerate(conditions) if v])
        if pretrimmed and not self.use_t:
            idxs -= 1
        return idxs

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

        return os.path.join(base_dir, f"{self.slurm.job}_{name}{extension}")

    def check_existing_data_file(self):
        """Check if the data file already exists and handle it based on the `force_gen` setting.
        If the file exists and `force_gen` is True, the file is removed."""
        if os.path.exists(self.file):
            if self.force_gen:
                self.logger.info("Removing existing data file '%s'", self.file)
                os.remove(self.file)
            else:
                self.logger.info("Data file '%s' exists, exiting", self.file)
                sys.exit(0)

    def should_plot(self):
        """Determine if plotting should be performed based on the `plot` setting and the SLURM job status."""
        return self.plot and self.slurm.is_main

    def shapes_str(self):
        """Returns a string representation of the shapes used in the simulation.
        If there is only one shape, it returns that shape as a string, i.e 'local'.
        If there are multiple shapes, it returns a concatenated string of the first letter of each shape, e.g. 'leo'.
        """
        if len(self.shapes) == 1:
            return self.shapes[0]
        return "".join([s[0] for s in self.shapes])

    def get_likelihoods(self):
        """Get the marginal likelihoods for the shapes used in the simulation.

        If there is only one shape this will return the fisher error for the shape,
        otherwise, will return the marginal likelihoods for all shapes.
        Returns:
            list: A list of marginal likelihoods for each shape.
        """
        with h5py.File(self.file, mode="r", swmr=True, locking=False) as f:
            if len(self.shapes) == 1:
                # just get the fisher error for the shape
                like = [np.sqrt(1 / f["fisher"][self.shapes[0]][0])]
            else:
                # here we generate a list of indices based on the shapes values
                indxs = []
                for s in self.shapes:
                    match s:
                        case "local":
                            indxs.append(0)
                        case "equilateral":
                            indxs.append(1)
                        case "orthogonal":
                            indxs.append(2)
                        case _:
                            self.logger.warning("Unknown shape %s, ignoring", s)

                like = f.get("marginal_likelihoods")[indxs]
        return like
