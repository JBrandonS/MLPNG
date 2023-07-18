
from mpi4py import MPI
from ksw import Shape, KSW, Cosmology, Data
import camb

from utils import *
from config import *

monitor = MemoryMonitor()

# Needs ~100 sim to reach ~1% accuracy,
# We should be able to step through the sims and save the MC so prevent the need to do this for larger runs
# That is TODO

camb_params_obj = camb.set_params(**cosmo_params)
cosmo = Cosmology(camb_params_obj)
cosmo.compute_transfer(cosmo_params['max_l'])
cosmo.compute_c_ell()

# if rank == 0:
alms = load_data(alm_final_file, 'alm')
almngs = load_data(alm_final_file, 'almng')
fnls = load_data(data_final_file, 'fnls')

alm_prime = alms + fnls[:, np.newaxis, np.newaxis] * almngs
# else:
#     fnls = np.empty((nsims,), dtype=np.float64)
#     alm_prime = np.empty((nsims, npol, nelem), dtype=np.complex128)

# comm.bcast(fnls, root=0)
# comm.bcast(alm_prime, root=0)

loc_shape = Shape.prim_local(ns=cosmo_params['ns'], pivot=cosmo_params['pivot_scalar'])
cosmo.add_prim_reduced_bispectrum(loc_shape, radii)

# TODO check
noise_ell, beam_ell = get_noise_beam()
data = Data(lmax, noise_ell, beam_ell, polarizations, cosmo)
icov = data.icov_diag_lensed if do_lensing else data.icov_diag_nonlensed

# TODO: double check this is correct
if disable_noise:
    beam = lambda alm: hp.sphtfunc.smoothalm(alm, fwhm=0, pol=False) 
else:
    beam_width = settings['beam_width'] * u.arcmin
    beam = lambda alm: hp.sphtfunc.smoothalm(alm, fwhm=beam_width.to_value(u.radian), pol=False)

ksw = KSW(cosmo.red_bispectra, icov, beam, lmax, polarizations, precision='double')

def alm_loader(str_idx):
    print(f'MPI {rank}: sending data for', str_idx, ', true fnl: ', fnls[int(str_idx)])
    print(alm_prime.shape, alm_prime[int(str_idx)].shape)
    return alm_prime[int(str_idx)]

alm_strs = np.arange(alm_prime.shape[0]).astype(str)

theta_batch = 250 #nelem // 10000

# maybe put all in 1 file?
ksw_mc_file = os.path.join(data_dir, 'ksw-'+data_str)
if os.path.exists(ksw_mc_file):
    print('Loading from file:', ksw_mc_file)
    ksw.start_from_read_state(ksw_mc_file, comm)
else:
    ksw.step_batch(alm_loader, alm_strs, comm, verbose=True, theta_batch=theta_batch)

    if rank == 0:
        ksw.write_state(ksw_mc_file, comm)

if rank == 0:
    fisher = ksw.compute_fisher()
else:
    fisher = None
fisher = comm.bcast(fisher, root=0)
# print('after bcast fisher:', fisher)

ests = ksw.compute_estimate_batch(alm_loader, alm_strs, comm, verbose=True, fisher=fisher, theta_batch=theta_batch)

monitor.join_and_plot(plot_dir, f'est_memory_{rank}')

if rank == 0:
    # save_data(data_final_file, {'fisher': fisher})
    save_data(data_final_file, {'estimates': ests})

print('Finished', rank, '!')
    