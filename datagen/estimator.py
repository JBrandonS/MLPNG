import os
import sys
import numpy as np

import healpy as hp
import camb

from ksw import Shape, KSW, Cosmology, Data
from astropy import units as u

from utils import MemoryMonitor, load_data, save_data, get_radii
from config import SimConfig

try: # This will still crash on mainframe if MPI fails to start correctly....
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    comm = None
    rank = 0

if rank == 0:
    monitor = MemoryMonitor()

# Needs ~100 sim to reach ~1% accuracy,

config_file = sys.argv[1] if len(sys.argv) > 1 else 'settings/settings.json'
s = SimConfig(config_file)

camb_params_obj = camb.set_params(**s.cosmo_params)
cosmo = Cosmology(camb_params_obj)
cosmo.compute_transfer(s.cosmo_params['max_l'], verbose=s.verbose)
cosmo.compute_c_ell()

# if rank == 0:
alms = load_data(s.alm_final_file, 'alm')
almngs = load_data(s.alm_final_file, 'almng')
fnls = load_data(s.data_final_file, 'fnls')

alm_prime = alms + fnls[:, np.newaxis, np.newaxis] * almngs
# else:
#     fnls = np.empty((nsims,), dtype=np.float64)
#     alm_prime = np.empty((nsims, npol, nelem), dtype=np.complex128)

# comm.bcast(fnls, root=0)
# comm.bcast(alm_prime, root=0)

radii, drs = get_radii(s.settings['r_min'], s.settings['r_max'])
loc_shape = Shape.prim_local(ns=s.cosmo_params['ns'], pivot=s.cosmo_params['pivot_scalar'])
cosmo.add_prim_reduced_bispectrum(loc_shape, radii)

# TODO check
noise_ell, beam_ell = s.get_noise_beam()
data = Data(s.lmax, noise_ell, beam_ell, s.polarizations, cosmo)
icov = data.icov_diag_lensed if s.lensing else data.icov_diag_nonlensed

# TODO: double check this is correct
if s.disable_noise:
    beam = lambda alm: hp.sphtfunc.smoothalm(alm, fwhm=0, pol=False) 
else:
    beam_width = s.settings['beam_width'] * u.arcmin
    beam = lambda alm: hp.sphtfunc.smoothalm(alm, fwhm=beam_width.to_value(u.radian), pol=False)

ksw = KSW(cosmo.red_bispectra, 
          icov, 
          beam, 
          s.lmax, 
          s.polarizations,
          precision='double' if s.double_precision else 'single')

def alm_loader(str_idx):
    return alm_prime[int(str_idx)]

alm_strs = np.arange(alm_prime.shape[0]).astype(str)

theta_batch = 250 #nelem // 10000

# maybe put all in 1 file?
ksw_mc_file = os.path.join(s.data_dir, 'ksw-'+s.data_str)
if os.path.exists(ksw_mc_file):
    if s.verbose: 
        print('Loading from file:', ksw_mc_file)
    ksw.start_from_read_state(ksw_mc_file, comm)
else:
    ksw.step_batch(alm_loader, alm_strs, comm, verbose=s.verbose, theta_batch=theta_batch)

    if rank == 0:
        ksw.write_state(ksw_mc_file, comm)

fisher = ksw.compute_fisher()
ests = ksw.compute_estimate_batch(alm_loader, alm_strs, comm, verbose=s.verbose, fisher=fisher, theta_batch=theta_batch)

if rank == 0:
    monitor.join_and_plot(s.plot_dir, f'est_memory_{rank}')
    # save_data(data_final_file, {'fisher': fisher})
    save_data(s.data_final_file, {'estimates': ests})

print('Finished', rank, '!')
    