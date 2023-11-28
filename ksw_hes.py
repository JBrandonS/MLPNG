import camb
import healpy as hp
import numpy as np
from astropy import units as u
from ksw import KSW, Cosmology, Data, Shape
from mpi4py import MPI

from scripts.config import SimConfig
from scripts.utils import get_radii, save_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config_file = "settings/scraped.json"
s = SimConfig(config_file, print_settings=(rank == 0))

t_scale = s.cosmo_params["TCMB"] * 10 ** (6)
lhdus = (1, 2, 3) if s.npol > 1 else 1

def remove_mono_dipole(alm):
    """
    Remove the monopole and dipole terms from the alms.
    """
    lmax = hp.Alm.getlmax(len(alm))
    alm[hp.Alm.getidx(lmax, 0, 0)] = 0.0  # Remove monopole
    alm[hp.Alm.getidx(lmax, 1, 0)] = 0.0  # Remove dipole
    alm[hp.Alm.getidx(lmax, 1, 1)] = 0.0  # Remove dipole
    return alm

fnls = []
alms = []
alm_l = []

print('loading alms', end=' ')
for i in np.arange(1, 11):
    idx = str(i).zfill(4)
    base1 = f"scraped/alm_l_{idx}_v3.fits"
    base2 = f"scraped/alm_nl_{idx}_v3.fits"

    alm_heidelberg_l = hp.read_alm(base1, hdu=lhdus)
    alm_heidelberg_nl = hp.read_alm(base2, hdu=lhdus)

    # print('alm_heidelberg_l nside', hp.get_nside(alm_heidelberg_l))

    alm_h_l = remove_mono_dipole(alm_heidelberg_l)
    alm_h_nl = remove_mono_dipole(alm_heidelberg_nl)

    # rng = np.random.default_rng()
    fnl = np.random.uniform(-10, 10)
    fnls.append(fnl)

    alms.append((alm_h_l + fnl * alm_h_nl)*t_scale)
    alm_l.append(alm_h_l*t_scale)
print('done')

camb_params_obj = camb.set_params(**s.cosmo_params)
cosmo = Cosmology(camb_params_obj)
cosmo.compute_transfer(s.cosmo_params["max_l"], verbose=s.verbose)
cosmo.compute_c_ell()

radii, drs = get_radii(s.settings["r_min"], s.settings["r_max"])
loc_shape = Shape.prim_local(
    ns=s.cosmo_params["ns"], pivot=s.cosmo_params["pivot_scalar"]
)
cosmo.add_prim_reduced_bispectrum(loc_shape, radii)

noise_ell, beam_ell = s.get_noise_beam()
data = Data(s.lmax, noise_ell, beam_ell, s.polarizations, cosmo)
icov = data.icov_diag_lensed if s.lensing else data.icov_diag_nonlensed

if s.disable_noise:

    def beam(alm):
        return alm  # hp.sphtfunc.smoothalm(alm, fwhm=0, pol=False)

else:
    beam_width = s.settings["beam_width"] * u.arcmin
    beam_width_rad = beam_width.to_value(u.radian)

    def beam(alm):
        return hp.sphtfunc.smoothalm(alm, fwhm=beam_width_rad, inplace=False)


ksw = KSW(
    cosmo.red_bispectra,
    icov,
    beam,
    s.lmax,
    s.polarizations,
    precision="double" if s.double_precision else "single",
)

alm_strs = np.arange(0, 10).astype(str)
alm_step_strs = np.arange(0, 1000).astype(str) #np.random.choice(alm_strs, size=100, replace=False)

# def alm_step_loader(idx):
#     return alm_l[int(idx)].copy()

def alm_step_loader_2(idx):
    # needs a gaussian realization of signal + noise
    return data.compute_alm_sim(lens_power=s.lensing)

def alm_loader(idx):
    print('loading', idx, 'fnl', fnls[int(idx)])
    return alms[int(idx)]

def compute_icov_ell(N, b):
    S_ell = cosmo._camb_data.get_cmb_power_spectra(
        cosmo.camb_params, lmax=s.lmax, raw_cl=True, CMB_unit="muK"
    )["total"][:, 0]
    b_inv = 1/b
    return (1 / (S_ell + b_inv * N * b_inv))[None, :]

icov_ell = compute_icov_ell(noise_ell, beam_ell)
fiso = ksw.compute_fisher_isotropic(icov_ell, comm=comm)
print('fisher iso', fiso)

ksw.step_batch(
        alm_step_loader_2, alm_step_strs, comm, verbose=True
    )

# fisher = 1000
# alm_step_strs = np.arange(0, 50).astype(str) # just controls the amount per loop, nums are not actually used
# while fisher > 1e-6:
#     ksw.step_batch(
#         alm_step_loader_2, alm_step_strs, comm, verbose=False
#     )

fisher = ksw.compute_fisher()
print('fisher', fisher)

estimates = ksw.compute_estimate_batch(
    alm_loader,
    alm_strs,
    comm,
    verbose=s.verbose,
    fisher=fisher,
)

if rank == 0:
    sdata = {}
    sdata["settings"] = s.settings
    sdata["fisher"] = np.atleast_1d(fisher)
    sdata["fisher_iso"] = np.atleast_1d(fiso)

    sdata["estimates"] = estimates
    sdata["fnls"] = np.atleast_1d(fnls)

    snr = (estimates - fnls) * np.sqrt(fisher)
    sdata["errors"] = snr
    print("error_var", np.var(snr))
    sdata["errors_var"] = np.atleast_1d(np.var(snr))

    save_data("ksw_test.hdf5", sdata, verbose=s.verbose)
    # os.replace(s.data_file_nc, s.data_file_complete)
