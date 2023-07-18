import json
import time
import os
import healpy as hp

from astropy import units as u

from utils import *

try:
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    rank = 0
    size = 0
    comm = None

# kind of a bad idea to do things this way, but simplifies things and is easy to understand

starttime = time.time()

settings = json.load(open('settings.json', 'r'))

if not settings['silent'] and rank == 0:
    print(settings)

cosmo_params = settings['cosmo_params']
lmax = settings['cosmo_params']['lmax']
polarizations = settings['polarizations']

debug = settings['debug']
silent = settings['silent']
do_lensing = settings['do_lensing']

base_dir = settings['base_dir']
alm_cache_dir = settings['alm_cache_dir']
tmp_dir = settings['tmp_dir']
plot_dir = settings['plot_dir']
data_dir = base_dir + ('/lensed' if do_lensing else '/unlensed')

for s in [data_dir, plot_dir, alm_cache_dir, tmp_dir]:
    safe_makedirs(s)

nside = settings['nside']
nsims = settings['nsims']
npatches = settings['npatches']
narray = settings['narray']
npol = len(polarizations)

disable_noise = settings['disable_noise']

job_array_index = os.environ.get('SLURM_ARRAY_TASK_ID')
if job_array_index is not None:
    in_ja = True
    job_array_index = int(job_array_index)
    njobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT'))       # type: ignore
    job_array_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN'))  # type: ignore
    job_array_max = int(os.environ.get('SLURM_ARRAY_TASK_MAX'))  # type: ignore
    print('Running job array index', job_array_index)

    if job_array_index % 100 != 1:
        debug = False  # auto disable plots
        save_plots = False
else:
    in_ja = False
    njobs = 1

nell = lmax + 1
nelem = hp.Alm.getsize(lmax)
npix = hp.nside2npix(nside)
ells = np.arange(nell)
chars_of_polarizations = ''.join(polarizations)

nsims_str = str(nsims*narray)
nn_str = 'nn_' if disable_noise else ''

base_name = f'{nside}_{nn_str}{chars_of_polarizations}_{nsims_str}'

ja_str = '' if job_array_index is None else f'_{job_array_index}'

data_str = f'{base_name}x{settings["npatches"]}_fnl{settings["fnl_range"][0]}-{settings["fnl_range"][1]}{ja_str}'
data_file = os.path.join(data_dir, f'{data_str}.hdf5.nc')
data_final_file = os.path.join(data_dir, f'{data_str}.hdf5')

alm_str = f'{base_name}{ja_str}'
alm_file = os.path.join(alm_cache_dir, f'{alm_str}.alms.hdf5.nc')
alm_final_file = os.path.join(alm_cache_dir, f'{alm_str}.alms.hdf5')

def get_noise_beam():
    beam_width = settings['beam_width'] * u.arcmin
    noise_scale_tt = settings['noise_scale_tt'] * u.arcmin
    noise_scale_ee = settings['noise_scale_ee'] * u.arcmin
    noise_scale_te = settings['noise_scale_te'] * u.arcmin

    beam_ell_pre = hp.gauss_beam(
        beam_width.to_value(u.radian), lmax=lmax, pol=True)
    beam_ell_pre = np.swapaxes(beam_ell_pre, 0, 1)

    noise_ell = []
    beam_ell = []
    if 'T' in polarizations:
        noise = np.ones((nell)) * noise_scale_tt.to_value(u.radian)**2

        noise_ell.append(noise)
        beam_ell.append(beam_ell_pre[0])

    if 'E' in polarizations:
        noise = np.ones((nell)) * noise_scale_ee.to_value(u.radian)**2

        noise_ell.append(noise)
        beam_ell.append(beam_ell_pre[1])

    if polarizations == ['T', 'E']:
        noise = np.ones((nell)) * noise_scale_te.to_value(u.radian)**2
        noise_ell.append(noise)

    noise_ell = np.array(noise_ell).squeeze()
    beam_ell = np.array(beam_ell).squeeze()

    if disable_noise:
        noise_ell = noise_ell * 10**-12
        beam_ell = np.ones_like(beam_ell)
    return noise_ell, beam_ell

radii, drs = get_radii(settings['r_min'], settings['r_max'])
