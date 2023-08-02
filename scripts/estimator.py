import os
import sys
import numpy as np

import healpy as hp
import camb
import datetime

from ksw import Shape, KSW, Cosmology, Data
from astropy import units as u

from utils import MemoryMonitor, load_data, save_data, get_radii, load_single_data
from config import SimConfig

# This will still crash on mainframe if MPI fails to start correctly....
# but try except doesn't work for some reason, and makes completion error
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Needs ~100 sim to reach ~1% accuracy,

config_file = sys.argv[1] if len(sys.argv) > 1 else 'settings/settings.json'
s = SimConfig(config_file, print_settings= rank == 0)

if s.debug:
    monitor = MemoryMonitor()

def vp(*args, **kwargs):
    if s.verbose:
        print(f'{rank} | {datetime.datetime.now()} =>', *args, **kwargs)

radii, drs = get_radii(s.settings['r_min'], s.settings['r_max'])
loc_shape = Shape.prim_local(ns=s.cosmo_params['ns'], pivot=s.cosmo_params['pivot_scalar'])

vp('Running camb', flush=True)
camb_params_obj = camb.set_params(**s.cosmo_params)
cosmo = Cosmology(camb_params_obj)
cosmo.compute_transfer(s.cosmo_params['max_l'], verbose=s.verbose)
cosmo.compute_c_ell()
cosmo.add_prim_reduced_bispectrum(loc_shape, radii)
vp('done', flush=True)

# TODO check
noise_ell, beam_ell = s.get_noise_beam()
data = Data(s.lmax, noise_ell, beam_ell, s.polarizations, cosmo)
icov = data.icov_diag_lensed if s.lensing else data.icov_diag_nonlensed

# TODO: double check this is correct
if s.disable_noise:
    beam = lambda alm: alm #hp.sphtfunc.smoothalm(alm, fwhm=0, pol=False) 
else:
    beam_width = s.settings['beam_width'] * u.arcmin
    beam_width_rad = beam_width.to_value(u.radian)
    beam = lambda alm: hp.sphtfunc.smoothalm(alm, fwhm=beam_width_rad)

vp('Starting KSW', flush=True)
ksw = KSW(cosmo.red_bispectra, 
          icov, 
          beam, 
          s.lmax, 
          s.polarizations,
          precision='double' if s.double_precision else 'single')
vp('done', flush=True)

def alm_loader(str_idx):
    vp('loading alms')
    idx = int(str_idx)
    alm = load_single_data(s.alm_final_file, 'alm', idx, verbose=s.verbose)
    almng = load_single_data(s.alm_final_file, 'almng', idx, verbose=s.verbose)
    fnl = load_single_data(s.data_file, 'fnls', idx, verbose=s.verbose)
    vp('Done')
    return alm + fnl * almng

alm_strs = np.arange(s.total_sims).astype(str)

theta_batch = 250 #nelem // 10000

# maybe put all in 1 file?
ksw_mc_file = os.path.join(s.data_dir, 'kswmc_'+s.data_str)
if os.path.exists(ksw_mc_file):
    vp('Loading from file:', ksw_mc_file, flush=True)
    ksw.start_from_read_state(ksw_mc_file, comm)
else:
    vp('starting setp_batch', flush=True)
    ksw.step_batch(alm_loader, alm_strs, comm, verbose=s.verbose, theta_batch=theta_batch)
    vp('finished setp_batch', flush=True)

    if rank == 0:
        ksw.write_state(ksw_mc_file, comm)

vp('Computing estimates', flush=True)
fisher = ksw.compute_fisher()
ests = ksw.compute_estimate_batch(alm_loader, alm_strs, comm, verbose=s.verbose, fisher=fisher, theta_batch=theta_batch)
vp('done', flush=True)

if s.debug:
    monitor.join_and_plot(s.plot_dir, f'{s.name}-est_memory_{rank}')

if rank == 0:
    sdata = {}
    sdata['fisher'] = np.atleast_1d(fisher)
    sdata['estimates'] = ests
    # # # sdata['errors'] = (ests - fnls) / fnls

    save_data(s.data_file, sdata, verbose=s.verbose)
    os.replace(s.data_file, s.data_final_file)

vp('Finished', rank, '!')
    