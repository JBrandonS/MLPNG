# Settings file

## Description

These settings file control the simulation and provide a large number of customization options. All options have defaults which will be used if the setting is not found and can be overridden by the user, these are indicated below by the value after the parameter name. The settings file is a JSON file and can be edited with any text editor. Some settings have a CLI option which will take precedence over the file settings. These settings can be placed in any order in the file.

## Settings

- `cosmo_params` : Dictionary of cosmological parameters used by CAMB. This will be passed into CAMB via the [camb.set_params](https://camb.readthedocs.io/en/latest/results.html#camb.results.CAMBdata.set_params) function, and supports any CAMB option supported there. The only required options have the following defaults:
  - `As`, 2.13e-09: Amplitude of the primordial power spectrum.
  - `pivot_scalar`, 0.05 : Pivot scale for the primordial power spectrum.
  - `ns`, 0.9624 : Spectral index of the primordial power spectrum.
  - `lmax`: Will be set by the `lmax` value below, do not set it here as it will be overridden.

- `seed` : Seed for the random number generator. This is used to generate the random noise and the random field. The code has not been fully tested for reproducibility but this should help. There is no default as the default is to use a random seed.
- `force_generation`, False : Boolean to force the generation of the CMB maps. If this is set to true the code will generate the CMB maps even if they already exist, otherwise it will check for existing data files and stop running. The code has been designed so that if a run partially fails you can restart it and it will only generate the missing data, this will override that behavior.
- `lensing`, True : Boolean to include lensing in the simulation. If true the lensing potential will be generated and added to the CMB map.
- `pols`, `T` : a string of [T, E, TE] for which polarization in the simulation.
- `double_precision`, False : Boolean to use double precision. This will increase the memory usage and has not been tested much.

- `nsims`, 100 : Number of simulations to generate per run.
- `ndups`, 25 : Number of sims to make per A_lm calculation, giving multiple sims created from the same background.
- `narray`, 1 : Number of runs that will be used. This allows using slurm to run multiple simulations in parallel while limiting the resources used. This does not actually run the simulations so you will need to match this with slurm settings.
- `nside`, 128 : Healpix resolution parameter. This is the resolution of the Healpix map used in the simulation.

  - > Note: The code used healpy's pixel weights to improve accuracy, and thus only supports a limited number of nsides found [here](https://github.com/healpy/healpy-data/tree/master/full_weights). Currently these are [32, 64, 128, 256, 512, 1024, 2048, 4096]. To change this behavior set `use_pixel_weights=False` in the map2alm function calls.

- `mc_steps`, 300 : Number of Monte Carlo steps used to initialize the KSW estimator.
- `estimate`, True : Boolean to compute KSW estimates during generation.
- `num_estimates`, 1000 : Number of estimates to use in the KSW estimator. This is useful if you are using a large number of simulations and want to use a subset of them for time.
- `save_ksw`, False : Boolean to save the KSW MC states to disk.
- `shapes`, `all` : Bispectrum shapes to generate. One of `local`, `equilateral`, `orthogonal`, or `all`, or a list of them.
- `phi_scale`, 1 : Scaling factor applied to the lensing potential.
- `lmin`, 2 : The lmin parameter used in calculations.
- `lmax`, 3*nside - 1 : The lmax parameter used in calculations. Defaults to optimal value for the nside parameter based on the [anafast documentation](https://healpix.sourceforge.io/html/fac_anafast.htm).
- `lmax_buffer`, 128 : Buffer for the lmax parameter used in some camb calculations to ensure that the code can calculate the required power spectra and lensing to sufficient lmax.
- `fnl_range`, [-1000, 1000] : Range of fNL values to use in the KSW estimator.

- `noise`, True : Boolean to include noise and beaming in the simulation.
- `beam_width`, 1 : Beam width in arcmin.
- `noise_tt`, 1 : Noise scale for the TT power spectrum, in arcmin.
- `noise_ee`, 5 : Noise scale for the EE power spectrum, in arcmin.

- `r_min`, 0 : The minimum value of r to use in the simulation.
- `r_max`, 50000 : The maximum value of r to use in the simulation.
  - > See [Smith and Zaldarriaga 2011](https://arxiv.org/abs/astro-ph/0612571) table 2 for information on the r values used in the simulation.

- `base_name` : Base name for the output files. This will be used to generate the output file names and is automatically generated based on settings if not provided.
- `base_dir`, "data" : Base directory for the output files. It is recommend to put this on a performative file system.
- `plot_dir`, "plots" : Directory for the output plots. This should be a subdirectory of `base_dir`.
- `model_dir`, "models" : Directory for the output models. This should be a subdirectory of `base_dir`.
- `data_dir`, "data" : Directory for the output data. This should be a subdirectory of `base_dir`.
- `mc_dir`, "kswmc" : Directory for the output monte carlo data. This should be a subdirectory of `base_dir`.
- `tb_dir`, "tensorboard" : Directory for TensorBoard logs. This should be a subdirectory of `base_dir`.
- `wandb_dir`, "wandb" : Directory for Weights & Biases logs. This should be a subdirectory of `base_dir`.

- `plot`, True : Boolean to generate plots. If false the code will not generate plots.
- `save_alms`, False : Boolean to save the alm data. If false the code will not save the alm data.
- `tensorboard`, False : Boolean to enable TensorBoard logging for the trainer.
- `wandb`, False : Boolean to enable Weights & Biases logging for the trainer.
- `tf_cache`, True : Boolean to enable file-based disk caching of the TensorFlow datasets.
- `tf_mem_cache`, True : Boolean to enable in-memory caching of the TensorFlow datasets.

## CLI args

The scripts can be run with a number of command line arguments which will override the settings file. This is to help with scripted runs. These are:

- `--nsims`, int : Number of simulations to generate per run.
- `--narray`, int : Number of runs that will be used.
- `--nside`, int : Healpix resolution parameter.
- `--ndups`, int : Number of sims to make per A_lm calculation, giving multiple sims created from the same background.
- `--base_dir`, str : Base directory for the output files.
- `--seed`, int : Seed for the random number generator.
- `--base_name`, str : Base name for the output files.
- `--lmin`, int : The lmin parameter used.
- `--lmax`, int : The lmax parameter used.
- `--lmax_buffer`, int : Buffer for the lmax parameter used in some camb calculations to ensure that the code can calculate the required power spectra to sufficient lmax.
- `--fnl_range`, [int, int] : Range of fNL values to use in the KSW estimator.
- `--num_estimates`, int : Number of estimates to use in the KSW estimator.
- `--mc_steps`, int : Number of Monte Carlo steps used to initialize the KSW estimator.
- `--phi_scale`, float : Scaling factor for the lensing potential.
- `--shapes`, [str, ...] : Bispectrum shapes to generate: local, equilateral, orthogonal, or all.
- `--pols`, str : a string of [T, E] for which polarization in the simulation.

The following arguments can be provided as is or with a `--no` prefix to negate them, e.g. `--no-lensing`:

- `--lensing` : Force lensing in the simulation.
- `--noise` : Force noise and beaming in the simulation.
- `--force_generation` : Force the generation of the CMB maps, overriding existing data.
- `--estimate` : Compute KSW estimates during generation.
- `--double_precision` : Force double precision.
- `--plot` : Force generation of plots.
- `--save_alms` : Save the combined alms.
- `--save_ksw` : Save the KSW MC states.
- `--tensorboard` : Enable TensorBoard logging.
- `--wandb` : Enable Weights & Biases logging.
- `--tf_cache` : Enable disk caching of TensorFlow datasets.
- `--tf_mem_cache` : Enable in-memory caching of TensorFlow datasets.

If scripting runs you can save the settings with:

- `--save_settings` : Save the settings to a file, including CLI changes. This will save to `base_dir/settings/<job>_<name>.json`.

> Note: the save_settings option is the only option which must be provided by CLI, otherwise it would do nothing but make a copy.

## Full example with defaults

```JSON
{
    "cosmo_params" : {
        "As": 2.13e-09,
        "pivot_scalar": 0.05,
        "ns": 0.9624
    },
    "seed": null,
    "force_generation": false,
    "lensing": true,
    "pols": "T",
    "double_precision": false,

    "nsims" : 100,
    "ndups": 25,
    "narray": 1,
    "nside": 128,

    "mc_steps": 300,
    "estimate": true,
    "num_estimates": 1000,
    "save_ksw": false,
    "shapes": "all",
    "phi_scale": 1,
    "lmin": 2,
    "lmax": 383,
    "lmax_buffer": 128,
    "fnl_range": [-1000, 1000],

    "noise": true,
    "beam_width" : 1,
    "noise_tt" : 1,
    "noise_ee" : 5,

    "r_min": 0,
    "r_max": 50000,

    "plot": true,
    "save_alms": false,
    "tensorboard": false,
    "wandb": false,
    "tf_cache": true,
    "tf_mem_cache": true,

    "base_name": null,
    "base_dir": "data",
    "plot_dir": "plots",
    "model_dir": "models",
    "data_dir": "data",
    "mc_dir": "kswmc",
    "tb_dir": "tensorboard",
    "wandb_dir": "wandb"
}
```

> For more information look at the `mlpng/core.py` file.
