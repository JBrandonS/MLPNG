# Settings file

## Description

These settings file control the simulation and provide a large number of customization options. All options have defaults which will be used if the setting is not found and can be overridden by the user, these are indicated below by the value after the parameter name. The settings file is a JSON file and can be edited with any text editor. Some settings have a CLI option which will take precedence over the file settings. These settings can be placed in any order in the file.

## Settings

- `cosmo_params` : Dictionary of cosmological parameters used by CAMB. This will be passed into CAMB via the [camb.set_params](https://camb.readthedocs.io/en/latest/results.html#camb.results.CAMBdata.set_params) function, and supports any CAMB option supported there. The only required options have the following defaults:
  - `As`, 2.13e-09: Amplitude of the primordial power spectrum.
  - `pivot_scalar`, 0.05 : Pivot scale for the primordial power spectrum.
  - `ns`, 0.9624 : Spectral index of the primordial power spectrum.

  - > Note: The value of `cosmo_params.lmax` used will be set by the `nside` parameter to be `lmax=3*nside-1`, this allows the function to be fully spectrally resolved with anafast. See the [Anafast documentation](https://healpix.sourceforge.io/html/fac_anafast.htm) for more information.

- `seed` : Seed for the random number generator. This is used to generate the random noise and the random field. The code has not been fully tested for reproducibility but this should help. There is no default as the default is to use a random seed.
- `force_generation`, False : Boolean to force the generation of the CMB maps. If this is set to true the code will generate the CMB maps even if they already exist, otherwise it will check for existing data files and stop running. The code has been designed so that if a run partially fails you can restart it and it will only generate the missing data, this will override that behavior.
- `force_ksw`, False : Boolean to force the creation of the KSW MC data, otherwise it will check for an existing file and load the MC state from the file.
- `lensing`, False : Boolean to include lensing in the simulation. If true the lensing potential will be generated and added to the CMB map.
- `pols`, False : Boolean to include polarization in the simulation. If true the code will generate both T and E modes of the CMB map.
- `double_precision`, False : Boolean to use double precision. This will increase the memory usage and has not been tested much.

- `nsims`, 100 : Number of simulations to generate per run.
- `narray`, 1 : Number of runs that will be used. This allows using SLURM to run multiple simulations in parallel while limiting the resources used. This does not actually run the simulations so you will need to match this with SLURM settings.
- `nside`, 1024 : Healpix resolution parameter. This is the resolution of the Healpix map used in the simulation.

  - > Note: The code used healpy's pixel weights to improve accuracy, and thus only supports a limited number of nsides found [here](https://github.com/healpy/healpy-data/tree/master/full_weights). Currently these are [32, 64, 128, 256, 512, 1024, 2048, 4096]. To change this behavior set `use_pixel_weights=False` in the map2alm function calls, currently only used in the `integrand` function in `scripts/generator.py`.

- `npatches`, 10 : Number of patches to generate per simulation. Must be divisible by 2.
- `patch_side_deg`, 10 : Size of the patches in degrees.

- `num_estimates`, nsims : Number of estimates to use in the KSW estimator. This is useful if you are using a large number of simulations and want to use a subset of them for time.
- `lmax_buffer`, 512 : Buffer for the lmax parameter used in some camb calculations to ensure that the code can calculate the required power spectra to sufficient lmax.
- `fnl_range`, [-10, 10] : Range of fNL values to use in the KSW estimator.

- `noise`, True : Boolean to include noise and beaming in the simulation.
- `beam_width`, 0 : Beam width in arcmin.
- `noise_scale_tt`, 1 : Noise scale for the TT power spectrum, in arcmin.
- `noise_scale_ee`, 1 : Noise scale for the EE power spectrum, in arcmin.
- `noise_scale_bb`, 0 : Noise scale for the BB power spectrum, in arcmin. Not really used.
- `noise_scale_te`, 1 : Noise scale for the TE power spectrum, in arcmin.

- `r_min`, 1 : The minimum value of r to use in the simulation.
- `r_max`, 50000 : The maximum value of r to use in the simulation.

- `base_name` : Base name for the output files. This will be used to generate the output file names and is automatically generated based on settings if not provided.
- `base_dir`, "data" : Base directory for the output files. It is recommend to put this on a performative file system.
- `plot_dir`, "plots" : Directory for the output plots. This should be a subdirectory of `base_dir`.
- `model_dir`, "models" : Directory for the output models. This should be a subdirectory of `base_dir`.
- `data_dir`, "data" : Directory for the output data. This should be a subdirectory of `base_dir`.
- `mc_dir`, "kswmc" : Directory for the output monte carlo data. This should be a subdirectory of `base_dir`.

- `model`, "isensee" : Model to use for the ML simulations. This is the name of the model to use for the ML simulations. This should be the name of a file in the `models` directory.

## CLI args

The scripts can be run with a number of command line arguments which will override the settings file. This is to help with scripted runs. These are:

- `--nsims`, int : Number of simulations to generate per run.
- `--narray`, int : Number of runs that will be used.
- `--npatches`, int : Number of patches to generate per simulation.
- `--base_dir`, str : Base directory for the output files.
- `--seed`, int : Seed for the random number generator.
- `--base_name`, str : Base name for the output files.
- `--lmax_buffer`, int : Buffer for the lmax parameter used in some camb calculations to ensure that the code can calculate the required power spectra to sufficient lmax.
- `--fnl_range`, [int, int] : Range of fNL values to use in the KSW estimator.
- `--num_estimates`, int : Number of estimates to use in the KSW estimator.

- `--save_settings`, flag: Save the settings to a file, including CLI changes. This will save to a file in the `base_dir/runs` directory.
- `--model`, str : Model to use for the ML simulations.

> Note: the save_settings option is the only option which must be provided by CLI, otherwise it would do nothing but make a copy.

The following arguments can be provided as is or with a `--no` prefix to set them to false, e.g. `--no-lensing`:

- `--lensing`, bool : Boolean to include lensing in the simulation.
- `--pols`, bool : Boolean to include polarization in the simulation.
- `--noise`, bool : Boolean to include noise and beaming in the simulation.
- `--force_generation`, bool : Boolean to force the generation of the CMB maps.
- `--force_ksw`, bool : Boolean to force the creation of the KSW MC data.

## Full example with defaults

```JSON
{
    "cosmo_params" : {
        "As": 2.13e-09,
        "pivot_scalar": 0.05,
        "ns": 0.9624,
    },
    "seed": null,
    "force_generation": false,
    "force_ksw": false,
    "lensing": false,
    "pols": false,
    "double_precision": false,

    "nsims" : 100,
    "narray": 1, 
    "nside":1024,
    "npatches": 10, 
    "patch_side_deg": 10,

    "num_estimates": 100,
    "lmax_buffer": 512,
    "fnl_range":[-10, 10],

    "noise": true,
    "beam_width" : 0,
    "noise_scale_tt" : 1,
    "noise_scale_ee" : 1,
    "noise_scale_bb" : 0,
    "noise_scale_te" : 1,

    "r_min": 1,
    "r_max": 50000,

    "base_name": null,
    "base_dir": "data",
    "plot_dir": "plots",
    "model_dir": "models",
    "data_dir": "data",
    "mc_dir": "kswmc",

    "model": "isensee"
}
```

> For more information look at the `scripts/core.py` file.
