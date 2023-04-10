import os
from logging import CRITICAL, ERROR
from typing import Literal

import numpy as np
from joblib import Parallel, delayed

from DensityFields import DensityField2D
from numpy2dataset import save_numpy_to_tf_dataset
from functools import partial

def make_map(base_field, fnl, seed, k_cut_low=None,k_cut_high=None):
    base_field.GenerateCAMBField(k_cut_low=k_cut_low,k_cut_high=k_cut_high,fnl=fnl,seed=seed)
    mapy = base_field.r_delta / base_field.r_delta.std()
    return mapy

def _save(dir, filename, maps, start_idx, end_idx, save_type: Literal['npy', 'tf', 'both'] = 'both'):
    print(f'Saving {dir}/{filename} to {save_type}...', end=' ')
    # save as tfrecord, higher performance possible
    if save_type in ['tf', 'both']:
        save_numpy_to_tf_dataset(np.stack(maps[start_idx:end_idx]), f'{dir}/{filename}.tfrecord')
    if save_type in ['npy', 'both']:
        with open(f'{dir}/{filename}.npy', 'wb') as f:
            np.save(f, np.stack(maps[start_idx:end_idx]))
    print('Done!')

def run_simulations(data_dir, name, num_sim, save_steps, box_size, grid, cosmo_params, fnls, 
                    force_fnl=None, n_jobs=-1, k_cut_low=None, k_cut_high=None, 
                    save_type: Literal['npy', 'tf', 'both']='both', interp_kind='linear', save=True, verbose=False):
    base = DensityField2D(box_size, grid, n_threads=1, cosmo_params=cosmo_params, verbose=False, log_level=ERROR, interp_kind=interp_kind)
    make_map_with_base = partial(make_map, base)
    
    for i in range(0, num_sim, save_steps):
        if verbose: print(f'Generating ({interp_kind}) maps {i} to {i + save_steps} for {name}...')
        maps_chunk = Parallel(n_jobs=n_jobs, verbose=1, prefer='threads')([
            delayed(make_map_with_base)(
                fnl if force_fnl is None else force_fnl, seed, k_cut_low=k_cut_low, k_cut_high=k_cut_high
            ) for seed, fnl in enumerate(fnls[i:i + save_steps])
        ])
        if verbose: print('Done!')
        if save: _save(data_dir, f'{interp_kind}/{name}_{i}-{i + save_steps}', maps_chunk, 0, len(maps_chunk), save_type=save_type)


zi = np.array([0, 3, 10, 30, 50, 100])
BoxSize = 1000.  # Size of the periodic box in Mpc/h
grid = 128  # Size of the grid
cell_size = BoxSize/grid  # Physical size of a cell
kF = 2*np.pi / BoxSize  # Fundamental mode of the box
kNyq = grid/2 * kF  # Nyquist frequency of the box
# Interpolation scheme for the power spectrum, either 'linear' or 'cubic'
interp_kind = 'linear'

# Cosmology settings
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

# simulations settings
num_sim = 10**7  # number of files to generate
num_steps = 10*5  # number of files to generate at a time
n_jobs = -1
fnl_range = (-1000, 1000)

base_name = f'{grid}x{num_sim//1000}k_fnl{fnl_range[0]}-{fnl_range[1]}'
data_dir = f'data/camb_3/{base_name}'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

data_file_pre = f"runs"
fnl_file_pre = f"fnls"

# Load / Generate fnls
fnl_path = os.path.join(data_dir, f"{fnl_file_pre}.npy")
if os.path.exists(fnl_path):
    print(f'Loading existing fnls from {fnl_path}...')
    fnls = np.load(fnl_path)
    np.random.shuffle(fnls)  # just add randomness between runs
else:
    print(f'Generating new fnls and saving to {fnl_path}...')
    fnls = np.random.uniform(
        fnl_range[0], fnl_range[1], num_sim).astype(np.float32)
    _save(data_dir, fnl_file_pre, fnls, 0, num_sim, save_type='both')

print(
    f'Generating {num_sim} maps of size {grid}x{grid} with fnl in range {fnl_range}...')
df_base = DensityField2D(BoxSize, grid, n_threads=1,
                        cosmo_params=cosmo_params, verbose=True)
print('Finished generating base map and fnls.')

print('Generating maps...')
run_simulations(data_dir, base_name, num_sim, num_steps, BoxSize, grid,
                cosmo_params, fnls, n_jobs=n_jobs, k_cut_high=0.25132741228718347)
print('Finished generating maps.')

# print('Generating aux maps...')
# run_simulations(data_dir, f"{base_name}_fnl-100", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, force_fnl=-100, n_jobs=n_jobs, k_cut_high=0.25132741228718347)
# run_simulations(data_dir, f"{base_name}_fnl0", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, force_fnl=0, n_jobs=n_jobs, k_cut_high=0.25132741228718347)
# run_simulations(data_dir, f"{base_name}_fnl100", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, force_fnl=100, n_jobs=n_jobs, k_cut_high=0.25132741228718347)
# run_simulations(data_dir, f"{base_name}_highcut", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, n_jobs=n_jobs, k_cut_high=1.1*kF)
# run_simulations(data_dir, f"{base_name}_lowcut", num_sim, num_steps, BoxSize, grid, cosmo_params, fnls, n_jobs=n_jobs, k_cut_low=1.1*kF)
# print('Finished generating aux maps.')
