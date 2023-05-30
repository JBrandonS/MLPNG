# %%
import os
import re
import resource
import psutil

import numpy as np
from numpy.random import randint, normal, uniform
import pylab as pl
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from joblib import Parallel, delayed
from functools import partial

from astropy import units as u

import healpy as hp
import camb

from ksw import Cosmology, Data
from ksw.radial_functional import radial_func

import h5py
from tqdm.auto import tqdm

# %matplotlib inline
# %load_ext line_profiler

# %%
def p_mem():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_info = py.memory_info()
    memory_use_in_bytes = memory_info.rss
    memory_use_in_gb = memory_use_in_bytes / (1024 ** 3)
    return f'{memory_use_in_gb:,.4f} GB'

# %% [markdown]
# # Data Generator
# 
# This code primamrly uses [KSW](https://github.com/AdriJD/ksw/tree/master/src), as a means to generate non-gaussian cmb maps. These get stored in a data file with the fnls, fullsky maps, and patches.
# 
# The fullsky maps are generated using the method discussed in [CMB lensing and primordial non-Gaussianity](https://arxiv.org/abs/0905.4732), where we find (eq. 6) 
# $$
# a_{\ell m} = a_{\ell m}^{{G}} + f_{NL}^X a_{\ell m}{^{NG}}
# $$
# and generated the full sky map by passing these $a_{\ell m}$ into healpy.
# 
# Most of this code is to calculate the term (eq. 27)
# 
# $$
# a_{\ell m}^{NG,loc'} = \int dr r^2 \left[ \alpha_\ell(r)\left(\int d^2 \hat{n} Y_{\ell m}^\star (\hat{n}) B(r,\hat{n})^2 \right)\right]
# $$
# 
# and
# 
# $$
# \alpha_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^2 \Delta_\ell^T(k) j_\ell(k r)
# $$
# 
# $$
# \beta_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^{-1} \Delta_\phi \Delta_\ell^T(k) j_\ell(k r)
# $$
# 
# $$
# B(r, \hat{n}) = \sum_{\ell,m} \frac{\beta_\ell (r)}{C_\ell} a_{\ell m} Y_{\ell m}
# $$
# 
# where $\Delta_\phi$ is pridormial normalizatiopn, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions
# 
# ---

# %% [markdown]
# ## Settings
# 
# See: https://camb.readthedocs.io/en/latest/model.html
# 
# Some values are forced during init of ksw.Cosmology. 
# For an example on how to modify these values after init: https://github.com/AdriJD/ksw/blob/fba3250cfe4c5145b5db12fa54bbc54bbcf5a56c/tests/python/test_cosmology.py#L65

# %%
cosmo_params = {
    'H0': 67.5,
    'r': 0,
    'As': 2.13e-09,
    'ns': 0.9624,
    'pivot_scalar': 0.05,
    'ombh2': 0.02233,
    'omch2': 0.1198,
    'mnu': 0.06,
    'tau': 0.0561,
    'TCMB': 2.7255,
    'max_l': 3000, # should be higher then lmax below, will get c_l_max > lmax error otherwise, for some reason
}

# %% [markdown]
# This lmax is the one that really gets used.
# lmax >= 300 is enforced by the ksw code due to errors with CAMB. 
# lmax needs to be somewhat smaller then max_l, if you get errors about c_ell change these.
# You will get warning messages if lmax > 4\* nside.  
# 
# nside should be of type 2\*\*n
# 
# num_patches should be even.
# 
# patch_side_deg \* num_patches \<\= 180, patch_side_deg \<\= 45; or you will overlap patches 
# 
# r_max is given in Mpc, we preform r_res slices over this volume to do the integral. Values have been picked by guestimation, with r_res being set low to help test. Will want to raise this for better calculations.
# 
# KSW only supports polarization values of 'T', 'E', ['T', 'E'].

# %%
lmax = 2500

fnl_range=(-1000, 1000)

nsims = 1000
npatches = 10
save_size = 10              # save every n sims, helps control memory usage

nside=1024
patch_side_deg = 10

r_min = 1                    # Mpc, min radius for the patch, >1e-6, but I've had issues below 1 with kernel crashes, check if this should be higher       
r_max = 14000                # Mpc, max radius for the patch, should be > SLS distance, check this value
r_res = 1000                 # number of slices in the radius, time is greatly impacted by this
r_batch = 100                # number of slices to compute per thread, memory usage is greatly impacted by this

# KSW only supports 'T' and 'E'
polarizations = ['T']#,'E']
# polarizations = ['T', 'E']

# FWHM of gaussian beam, use astropy units to make thing easy here
beam_width = 1 * u.arcmin # type: ignore

# noise settings, for the noise covariance matrix (without beam) in uK^2.
noise_loc = 0.
noise_scale = 0.1

# %%
# gotta keep this low
alm_parallel = Parallel(3, verbose=11)

parallel = Parallel(-1, verbose=11)

# %%
if isinstance(polarizations, str):
    chars_of_polarizations = polarizations
elif isinstance(polarizations, list):
    chars_of_polarizations = ''.join(polarizations)
else:
    raise TypeError("polarizations must be either a string or a list of strings")

base_name = f'{nside}_{nsims}x{npatches}_fnl{fnl_range[0]}-{fnl_range[1]}_r{r_res}_p{chars_of_polarizations}'
data_dir = f'data/ksw'
filename = f'{base_name}.hdf5.nc'
data_file = os.path.join(data_dir, filename)

if not os.path.exists(data_dir): 
    os.makedirs(data_dir)
    print(f'Created directory {data_dir}')
else:
    print(f'Reusing directory {data_dir}')
    # pattern = re.compile(f"{base_name}_\d+-\d+\.npy")
    pattern = re.compile(f"{base_name}_\d+-\d+\.hdf5(\.nc)?")
    for file in os.listdir(data_dir):
        if pattern.match(file):
            file_path = os.path.join(data_dir, file)
            os.remove(file_path)
            print(f"Deleted existing data file: {file_path}")

print(p_mem(), data_dir, filename, data_file)

# %%
npol = 1 if isinstance(polarizations, str) else len(polarizations)
pol = npol > 1
nell = lmax + 1

ls, ms = hp.Alm.getlm(lmax)

print(p_mem(), npol, pol, nell, ls.shape, ms.shape)

# %%
# np.split didnt do what I wanted
def split_into_batches(data, batch_size):
    batches = []
    indices = []
    
    # number of complete batches
    num_batches = len(data) // batch_size
    
    for i in range(num_batches):
        batches.append(data[i*batch_size:(i+1)*batch_size])
        indices.append(np.arange(i*batch_size, (i+1)*batch_size))
    
    # if there are leftover elements
    if len(data) % batch_size != 0:
        batches.append(data[num_batches*batch_size:])
        indices.append(np.arange(num_batches*batch_size, len(data)))
        
    return np.ascontiguousarray(indices), np.ascontiguousarray(batches)

radii = np.linspace(r_min, r_max, r_res)
dr = radii[1] - radii[0]

sim_idxs, sim_slices = split_into_batches(np.arange(nsims), save_size)
radii_idxs, radii_slices = split_into_batches(radii, r_batch)

print(p_mem(), sim_idxs.shape, sim_slices.shape, radii_idxs.shape, radii_slices.shape)

# %%
noise_ell = normal(noise_loc, noise_scale, (3, nell) if pol else (nell))
beam_ell = hp.gauss_beam(beam_width.to_value(u.radian), lmax, pol)

if pol:
    # KSW only supports 'T' and 'E', check this is what we are getting
    beam_ell = beam_ell[:,:2].swapaxes(0,1)

print(p_mem(), noise_ell.shape, beam_ell.shape)

# %% [markdown]
# ## Simulate the patches

# %%
camb_params_obj = camb.set_params(**cosmo_params)
cosmo = Cosmology(camb_params_obj, verbose=True)

cosmo.compute_transfer(cosmo_params['max_l'])
cosmo.compute_c_ell()

# %%
data = Data(lmax, noise_ell, beam_ell, polarizations, cosmo)

c_ells = data.cosmology.c_ell['unlensed_scalar']['ells'] # type: ignore
tr_ell_k = data.cosmology.transfer['tr_ell_k']
ells = data.cosmology.transfer['ells']
ks = data.cosmology.transfer['k']

alm = data.compute_alm_sim(lens_power=False)

print(p_mem(), alm.shape, tr_ell_k.shape, ells.shape, ks.shape, c_ells.shape)

# %%
def interpolate_ells(func, ells_sparse, ls):
    return CubicSpline(ells_sparse, func, axis=1)(ls)

# %% [markdown]
# Radial func computes
# 
# $$
# f_\ell^X(r) = \frac{2}{\pi} \int k^2 dk f(k) \Delta^{TX}_\ell(k) j_\ell(k r)
# $$
# 
# $$
# \alpha_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^2 \Delta_\ell^T(k) j_\ell(k r)
# $$

# %%
# f_k = np.swapaxes([ks], 0, 1) 
f_k = np.ones((len(ks),1))

alpha_ell = radial_func(f_k, tr_ell_k, ks, radii, ells).squeeze()
alpha_l = np.concatenate(np.array([interpolate_ells(alpha_ell, ells, np.arange(lmax))]))
alpha_l = np.ascontiguousarray(alpha_l)

# %% [markdown]
# $$
# \beta_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^{-1} \Delta_\phi \Delta_\ell^T(k) j_\ell(k r)
# $$

# %%
delta_phi = 2 * np.pi ** 2 * cosmo_params['As'] * (3 / 5)**2

f_k = (ks**-3 * delta_phi)[:, np.newaxis]
beta_ell = radial_func(f_k, tr_ell_k, ks, radii, ells).squeeze()

div = beta_ell / c_ells[np.newaxis, ells, np.newaxis]

bl_div_cl = np.concatenate(np.array([interpolate_ells(div, ells, np.arange(lmax))]))
bl_div_cl = np.ascontiguousarray(bl_div_cl)

print(p_mem(), alpha_l.shape, bl_div_cl.shape)

# %% [markdown]
# $$
# B(r, \hat{n}) = \sum_{\ell,m} \frac{\beta_\ell (r)}{C_\ell} a_{\ell m} Y_{\ell m}
# $$
# 
# $$
# a_{\ell m}^{NG,loc'} = \int dr r^2 \left[ \alpha_\ell(r)\left(\int d^2 \hat{n} Y_{\ell m}^\star (\hat{n}) B(r,\hat{n})^2 \right)\right]
# $$

# %%
def alm_ng_slice(dr, nside, lmax, pol, alm, bl_div_cl, alpha_l, radii):
    Balm = np.stack([ hp.almxfl(alm, bl_div_cl[i]) for i in range(len(radii)) ])

    B = hp.alm2map(Balm, nside=nside, lmax=lmax, mmax=None, pol=pol, pixwin=False, fwhm=0, sigma=None)

    inner = hp.map2alm(B**2, lmax=lmax, mmax=None, pol=pol, use_pixel_weights=True)
    
    alm_ng = np.sum([dr * radii[i]**2 * hp.almxfl(inner[i], alpha_l[i]) for i in range(len(radii))], axis=0) # type: ignore
    print('slice', p_mem(), alm_ng.shape)
    return alm_ng

def get_alm_ng_slice(bl_div_cl, alpha_l, radii_i, radii_s, p):
    for r_idx, r in zip(radii_i, radii_s):
        yield bl_div_cl[r_idx,:,p], alpha_l[r_idx,:,p], r

# %%
alm_ng = np.sum([alm_ng_slice(dr, nside, lmax, pol, alm[0], bl_div_cl, alpha_l, r) # type: ignore
    for bl_div_cl, alpha_l, r in get_alm_ng_slice(bl_div_cl, alpha_l, radii_idxs, radii_slices, 0)], axis=(0))

print('computed alm_ng', p_mem(), alm_ng.shape)

# %%
def cutSqPatches(fullsky_map, img_size, side_deg, num_patches):
    Tmap_datat = np.zeros((num_patches//2, int(img_size), int(img_size)))
    Tmap_datab = np.zeros((num_patches//2, int(img_size), int(img_size)))

    for counter in range(num_patches//2):
        Tmap_datat[counter] = np.ma.getdata(hp.cartview(fullsky_map, fig=0, xsize=img_size, ysize=img_size, rot=[0, 0], lonra=[side_deg*counter, side_deg*(counter+1)], latra=[0, side_deg],
                                                        title="CartView", unit="mK", format="%.2g", return_projected_map=True))
        Tmap_datab[counter] = np.ma.getdata(hp.cartview(fullsky_map, fig=1, xsize=img_size, ysize=img_size, rot=[0, 0], lonra=[side_deg*counter, side_deg*(counter+1)], latra=[-side_deg, 0],
                                                        title="CartView", unit="mK", format="%.2g", return_projected_map=True))
    
    return np.concatenate((Tmap_datat, Tmap_datab))

def get_sim_run(sims):
    for sim in sims:
        for p in range(npol):
                yield sim, p

def run_sim(nside, npatches, patch_side_deg, lmax, pol, alm, alm_ng, fnl):
    alm_prime = np.stack(alm + fnl * alm_ng)

    print('running sim', p_mem(), alm_prime.shape)
    maps  = hp.alm2map(alm_prime, nside, lmax=lmax, mmax=None, pol=pol, pixwin=False, fwhm=0, sigma=None)
    patches = cutSqPatches(maps, nside, patch_side_deg, npatches)
    
    return (maps, patches)

def append_to_hdf5(file_path, data_dict):
    with h5py.File(file_path, 'a') as f:
        for key, value in data_dict.items():
            # If dataset exists in file, append to it
            if key in f:
                f[key].resize((f[key].shape[0] + value.shape[0]), axis = 0)
                f[key][-value.shape[0]:] = value
            else:
                maxshape = (None,) + value.shape[1:]
                dataset = f.create_dataset(key, shape=value.shape, maxshape=maxshape, chunks=True)
                dataset[:] = value
                

# %%
fnls = uniform(fnl_range[0], fnl_range[1], nsims).astype(np.float32)

# %%
pl.ioff()
for sims in sim_slices:
    sim_data = parallel(delayed(run_sim)(nside, npatches, patch_side_deg, lmax, pol, alm[p], alm_ng, fnls[sim]) 
        for sim, p in get_sim_run(sims)) # type: ignore

    maps, patches = zip(*sim_data)
    maps = np.stack(maps)
    patches = np.stack(patches)
    
    data_dict = {'fnl': fnls, 'maps': maps, 'patches': patches}

    print('saving', p_mem(), maps.shape, patches.shape)
    append_to_hdf5(data_file, data_dict)
    
pl.close('all')
pl.ion();

# %%
# rename file to indicate that it is done
os.rename(data_file, data_file.replace('.hdf5.nc', '.hdf5'))

print('done', p_mem())

# %% [markdown]
# Done with generation

# %% [markdown]
# ---
# 
# # Tests

# %%
random_indices = [(0, randint(nsims), randint(npatches)) for _ in range(4)]

random_indices

# %%
for pol, s, p in random_indices:
    hp.mollview(maps[s], title=f'Sim {s}, patch {p}, pol {pol}, fnl {fnls[s]}', unit='mK')

# %%
for pol, s, p in random_indices:
    plt.figure()
    plt.imshow(patches[s, p])
    plt.title(f"patch {s*npatches + p} [fnl={fnls[s]}]")
    plt.show()

# %%
for pol, s, p in random_indices:
    cl = hp.anafast(maps[s], lmax=lmax)
    ell = np.arange(len(cl))
    plt.figure()
    plt.plot(ell, ell * (ell + 1) * cl)
    plt.title(f"sim {s} [fnl={fnls[s]}]")
    plt.show()

# %%
# TODO get KSW estimator for fnl

# %% [markdown]
# # Goodbye


