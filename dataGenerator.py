# %% [markdown]
# # Data Generator Notebook

# %% [markdown]
# This notebook generates (non-)Gaussian fields using CAMB and saves them in a .npy file and as tensorboard datasets.

# %% [markdown]
# Base code was taken from Thomas
# 
# Modifications by Brandon

# %%
# %load_ext autoreload
# %autoreload 2

import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

from DensityFields import DensityField2D

import os
import numpy as np
from typing import Literal
from joblib import Parallel, delayed
from logging import CRITICAL, DEBUG, INFO, WARNING, ERROR, FATAL

from DensityFields import DensityField2D
from numpy2dataset import save_numpy_to_tf_dataset
from functools import partial

# %%
# Fixes an issues with threading on an HPC
# See: https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/ai-at-hpc-tips/joblib-example

# ## see current affinity
# import os
# os.sched_getaffinity(0)
# ## we see only one CPU
# # reset affinity - read about affinity in man taskset
# os.system("taskset -p 0xFFFFFFFF %d" % os.getpid())
# # check
# os.sched_getaffinity(0)

# thread_type: Literal['threads', 'processes', None] = None # 'threads'

# %% [markdown]
# ## Functions

# %%
def _make_map(fnl, seed, interp_kind, box_size, grid, cosmo_params, k_cut_low=None,k_cut_high=None):
    base_field = DensityField2D(box_size, grid, cosmo_params=cosmo_params, interp_kind=interp_kind, verbose=False)
    base_field.GenerateCAMBField(k_cut_low=k_cut_low,k_cut_high=k_cut_high,fnl=fnl,seed=seed)
    return base_field.r_delta / base_field.r_delta.std()

SaveTypes = Literal['npy', 'tf', 'both', 'none']
def _save(dir, filename, maps, start_idx, end_idx, save_type: SaveTypes = 'npy'):
    if not os.path.exists(dir): os.makedirs(dir)
    print(f'Saving {dir}/{filename} to {save_type}...', end=' ')
    # save as tfrecord, higher performance possible
    if save_type in ['tf', 'both']:
        print('Saving as tfrecord not supported right now...')
        # save_numpy_to_tf_dataset(np.stack(maps[start_idx:end_idx]), f'{dir}/{filename}.tfrecord')
    if save_type in ['npy', 'both']:
        with open(f'{dir}/{filename}.npy', 'wb') as f:
            np.save(f, np.stack(maps[start_idx:end_idx]))
    print('Done!')

def run_simulations(data_dir, name, num_sim, save_steps, box_size, grid, cosmo_params, fnls, 
                    force_fnl=None, n_jobs=-1, k_cut_low=None, k_cut_high=None, 
                    save_type: SaveTypes='npy', interp_kind='cubic', log_level=WARNING, verbose=False):
    for i in range(0, num_sim, save_steps):
        # base = DensityField2D(box_size, grid, n_threads=1, cosmo_params=cosmo_params, verbose=False, log_level=log_level)
        # make_map_with_base = partial(_make_map, base)
        if verbose: print(f'Generating ({interp_kind}) maps {i} to {i + save_steps} for {name}...')
        maps_chunk = Parallel(n_jobs=n_jobs, verbose=1)([
            delayed(_make_map)( #make_map_with_base)(
                fnl if force_fnl is None else force_fnl, seed, interp_kind, box_size, grid, cosmo_params=cosmo_params, k_cut_low=k_cut_low, k_cut_high=k_cut_high
            ) for seed, fnl in enumerate(fnls[i:i + save_steps])
        ])
        if verbose: print('Done!')
        if save_type != 'none': _save(f'{data_dir}/{interp_kind}', f'{name}_{i}-{i + save_steps}', maps_chunk, 0, len(maps_chunk), save_type=save_type)

def _bispectrum(BoxSize, grid, fnl,seed, kgrid, ls, qs, transfers, ells, d_A, cosmo, verbose=True):
    FFT_map = DensityField2D(BoxSize, grid, n_threads=1, d_A=d_A, ls=ls, qs=qs, transfers=transfers, ells=ells, cosmo_params=cosmo, kgrid=kgrid, log_level=ERROR, verbose=verbose)
    FFT_map.GenerateCAMBField(k_cut_high=None,fnl=fnl,seed=seed,verbose=verbose)
    BBB = FFT_map.Bk(2.5,3,13, 'All',verbose=verbose)
    return BBB

def get_bispectrum(base, num_bispectra, fnl):
    return np.array(Parallel(n_jobs=-1, verbose=1)([
        delayed(_bispectrum)(
        base.BoxSize, base.grid, fnl,i,base.kgrid, base.Ls, base.qs, base.transfers, base.ells,base.d_A, base.cosmo
        ) for i in range(num_bispectra)
    ]))

# %% [markdown]
# ## Test Generation

# %% [markdown]
# ### Test Settings

# %%
## Cosmology settings (LCDM)
cosmo_params = {
    'h': 0.6711,
    'r': 0,
    'As': 2.13e-09,
    'A': 2*np.pi*np.pi*2.13e-09,
    'ns': 0.9624,
    'kpivot': 0.05,
    'z_recomb': 1090.48,
    'ombh2': 0.02233,
    'omch2': 0.1198,
    'tau': 0.0561,
    'lmax': 11000,
    'accuracy_boost': 1,
}

num_bispectra = 100
fnl_range=(-1000, 1000)
num_threads = -1 #for all cores
BoxSize = 1000.                     # Size of the periodic box in Mpc/h
grid = 256                          # Size of the grid

# needed values
kF = 2*np.pi / BoxSize              # Fundamental mode of the box
kNyq = kF * grid / 2                # Nyquist frequency of the grid

# Should we only generate the data, skipping tests and fisher forcasts?
run_test = False 

# %%
if run_test:
    test_map = DensityField2D(BoxSize,grid,n_threads=1, cosmo_params=cosmo_params, log_level=DEBUG)
    test_field = test_map.GenerateCAMBField(fnl=1.,seed=0,verbose=True, debug_plots=True)

# %%
if run_test:
    FFT_map = DensityField2D(BoxSize,grid,n_threads=1, cosmo_params=cosmo_params)
    
    # Generate the same field but cut off at grid/3*kF (where FFT bispectrum measurements start to fail)
    FFT_map.GenerateCAMBField(0,fnl=1.,k_cut_high=grid/3*kF,seed=None,verbose=False)
    plt.imshow(FFT_map.r_delta)
    plt.title('density field with k_cut_high {}'.format(grid/3*kF))
    plt.show()

    kk, Pk, _ = FFT_map.Pk()
    plt.loglog(kk,Pk)
    plt.ylabel("$P(k)$ [Mpc/h]$^3$")
    plt.xlabel("$k$ [h/Mpc]")
    plt.show()

# %% [markdown]
# ## Fisher Forecast

# %% [markdown]
# Computing Bispectra can be done as follows

# %%
if run_test:
    # lets time how long to get the bispectrum
    df_base = DensityField2D(BoxSize, grid, n_threads=1, cosmo_params=cosmo_params)
    df_base.GenerateCAMBField(0,fnl=1.)
    
    # Compute the bispectrum in a given binning:
    BBB = FFT_map.Bk(2.5,3,13,'All',verbose=True)
    plt.semilogy(BBB[:,-2])
    plt.ylabel("$B(k)$ [Mpc/h]$^6$")
    plt.xlabel("triangle$_i$")
    print("kmax =",(BBB[-1,0]+1.5)*kF,grid/3*kF)

# %% [markdown]
# ### Perform a Fisher forecast

# %% [markdown]
# We compute many bispectra with fixed amounts of pnG

# %%
if run_test:
    # bf_base = DensityField2D(BoxSize,grid,n_threads=1, cosmo_params=cosmo_params)
    BispecP = get_bispectrum(df_base, num_bispectra, 100.)
    BispecM = get_bispectrum(df_base, num_bispectra, -100.)
    BispecG = get_bispectrum(df_base, num_bispectra,  0.)

# %% [markdown]
# We compute the covariance matrix and its inverse, corrected by the Hartlap factor

# %%
if run_test:
    Cov = np.cov(BispecG[:,:,-2].T)

    hartlapfactor = (len(BispecG) - len(Cov) - 2) / (len(BispecG) - 1)
    Cov_Inv = np.linalg.inv(Cov)
    Cov_Inv *= hartlapfactor

    Cov = np.diag(np.diag(Cov))
    # Cov_Inv = np.linalg.inv(Cov)

    plt.semilogy(np.diag(Cov))
    print(hartlapfactor)

# %% [markdown]
# We compute the derivative of the bispectra with respect to $f_{\rm NL}$

# %%
if run_test:
    dBdf = (BispecP.mean(0)[:,-2]-BispecM.mean(0)[:,-2])/200
    plt.semilogy(dBdf)
    plt.show()

# %% [markdown]
# ### Then the Fisher information is given by $$F = \sum_{TT'} \frac{\partial B_T}{f_{\rm NL}} C^{-1}_{TT'} \frac{\partial B_{T'}}{f_{\rm NL}}$$ and the measurement error is $\sigma_{f_{\rm NL}} = F^{-1/2}$

# %%
if run_test:
    FF = (dBdf.dot(Cov_Inv).dot(dBdf))
    sigma = FF**-.5
    print(sigma)

# %% [markdown]
# ### One can also estimate $f_{\rm NL}$ from the generated bispectra: $$\hat{f}_{\rm NL} = F^{-1}\sum{TT'}\frac{\partial B_T}{f_{\rm NL}}C^{-1}_{TT'} B_{T'}$$ where $B_{T'}$ is a measured bispectrum

# %%
if run_test:
    estimates_P = np.array([dBdf.dot(Cov_Inv).dot(BispecP[i,:,-2])/FF for i in tqdm(range(len(BispecP)))])
    estimates_M = np.array([dBdf.dot(Cov_Inv).dot(BispecM[i,:,-2])/FF for i in tqdm(range(len(BispecM)))])
    estimates_G = np.array([dBdf.dot(Cov_Inv).dot(BispecG[i,:,-2])/FF for i in tqdm(range(len(BispecG)))])

# %%
if run_test: 
    print(estimates_P.mean(), estimates_M.mean(), estimates_G.mean())

# %%
if run_test:
    print(estimates_P.std(), estimates_M.std(), estimates_G.std())

# %%
if run_test:
    plt.hist(estimates_P,bins=100)
    plt.hist(estimates_G,bins=100)
    plt.hist(estimates_M,bins=100)
    plt.show()

# %%
if run_test:
    plt.plot(BispecP.mean(0)[:,-2]*BispecP[0,:,:3].prod(1)*kF**3)
    plt.plot(BispecM.mean(0)[:,-2]*BispecM[0,:,:3].prod(1)*kF**3)
    plt.plot(BispecG.mean(0)[:,-2]*BispecG[0,:,:3].prod(1)*kF**3)

# %% [markdown]
# # Map Generation

# %% [markdown]
# Now we generate and save the corresponding maps.

# %% [markdown]
# ## Generator Settings

# %%
## Cosmology settings (LCDM)
cosmo_params = {
    'h': 0.6711,
    'r': 0,
    'As': 2.13e-09,
    'A': 2*np.pi*np.pi*2.13e-09,
    'ns': 0.9624,
    'kpivot': 0.05,
    'z_recomb': 1090.48,
    'ombh2': 0.02233,
    'omch2': 0.1198,
    'tau': 0.0561,
    'lmax': 11000,
    'accuracy_boost': 4,
}

## simulations settings
num_sim = 10**6 # number of files to generate
num_steps = 10**6 # Number of steps to use per file (keeps memory usage low)
fnl_range=(-1000, 1000)

BoxSize = 1000.                     # Size of the periodic box in Mpc/h
grid = 128                          # Size of the grid

# interp='linear'
interp='cubic'
save_type='npy' #tensorflow isnt working

# Number of threads to use
num_threads = -1 # for all cores

## Data Settings
base_name = f'{grid}x{num_sim//1000}k_fnl{fnl_range[0]}-{fnl_range[1]}'
data_file = f"{base_name}"
fnl_file = f"{base_name}-fnls"

data_dir=f'data/camb_3/{base_name}'
if not os.path.exists(data_dir): os.makedirs(data_dir)

# Generate extra maps such as fixed fnl maps
create_aux = False

# %%
print(f'Generating Field with {num_sim} runs of resolution {grid}, box size {BoxSize} Mpc/h, interpolation stragety {interp}')
print(f'base_name: {base_name}\ndir: {data_dir}\ndata file: {data_file}\nfnl file: {fnl_file}')

# %% [markdown]
# ## Generate / Load FNLs

# %% [markdown]
# Start by generating the fnls, loading them if possible 

# %%
# %%time
fnl_path = os.path.join(data_dir, fnl_file)
if os.path.exists(fnl_path + '.npy'):
    print(f'Loading existing fnls from {fnl_path}.npy...')
    fnls = np.load(fnl_path + '.npy')
    # np.random.shuffle(fnls) # just add randomness between runs
else:
    print(f'Generating new fnls and saving to {fnl_path}...')
    fnls = np.random.uniform(fnl_range[0], fnl_range[1], num_sim).astype(np.float32)
    _save(data_dir, fnl_file, fnls, 0, num_sim, save_type='both')

# %% [markdown]
# Now we can run our simulations

# %%
# %%time
run_simulations(data_dir, base_name, num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, n_jobs=num_threads, interp_kind=interp, save_type='npy') #, k_cut_high=0.25132741228718347)

# %% [markdown]
# ## Generate AUX maps

# %%
if create_aux:
    print('Creating auxillary files use for additional analysis...')
    run_simulations(data_dir, f"{base_name}_fnl-100", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, force_fnl=-100, n_jobs=num_threads, k_cut_high=0.25132741228718347)
    run_simulations(data_dir, f"{base_name}_fnl0", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, force_fnl=0, n_jobs=num_threads, k_cut_high=0.25132741228718347)
    run_simulations(data_dir, f"{base_name}_fnl100", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, force_fnl=100, n_jobs=num_threads, k_cut_high=0.25132741228718347)
    run_simulations(data_dir, f"{base_name}_highcut", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, n_jobs=num_threads, k_cut_high=1.1*kF)
    run_simulations(data_dir, f"{base_name}_lowcut", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, n_jobs=num_threads, k_cut_low=1.1*kF)


