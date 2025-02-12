"""..."""
import gc
import logging
import os
import sys
from itertools import product
from tkinter import S
import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from tqdm.auto import trange

import camb
from ksw import Cosmology, Shape, KSW

import lenspyx
from lenspyx import utils_hp

from ksw.radial_functional import radial_func

from . import Core
from .utils import save_data, setup_logging, make_alm_plots, remove_mono_dipole

logger = setup_logging("mlpng.generator", level=logging.DEBUG)

def integrand(alm, alpha_l, bl_div_cl, lmax, nside, upscale=False):
    """
    Calculates the Outer integral of Hansen 2010 eq 27. That is:

    int dr r^2 [ alpha_ell(r) ( int d^2 hat{n} Y^*_{ell m}(hat{n}) B(r, hat{n})^2 ) ]

    Parameters:
        alm (ndarray): The input spherical harmonic coefficients.
        alpha_l (ndarray): The alpha_l coefficients.
        bl_div_cl (ndarray): The division of beta_ell function and C_l.
        lmax (int): The maximum l value.
        nside (int): The nside value.
        upscale (bool, optional): If True, the nside will be increased by 1 increment to improve calculations.

    Returns:
        ndarray: The calculated integrand.
    """
    if upscale:
        if nside >= 4096:
            raise ValueError(f"Nside {nside} is too large for upscaling")

        # valid nsides for use_pixel_weights
        nsides = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
        nside = nsides[nsides > nside][0]

    Balm = np.array([hp.almxfl(alm[p], bl_div_cl[..., p]) for p in range(alm.shape[0])])
    B = hp.alm2map(Balm, nside, pol=False)
    inner = np.array(hp.map2alm(B**2, lmax, use_pixel_weights=True, pol=False), ndmin=2)
    solution = [hp.almxfl(inner[p], alpha_l[..., p]) for p in range(alpha_l.shape[-1])]
    return np.array(solution)

def trap_generator(generator, radii):
    """
    Perform trapezoidal integration using a generator. This will consume the memory as possible to help
    with memory management, this becomes needed for nside >= 512.

    Parameters:
        generator: A generator yielding the function values to integrate.
        radii: The radii values, should match the size of the generator

    Returns:
        The integral computed using Simpson's rule.
    """
    integral = 0.0
    y_prev = radii[0] ** 2 * next(generator)
    for i in range(1, len(radii)):
        y_curr = radii[i] ** 2 * next(generator)
        integral += (radii[i] - radii[i - 1]) * (y_prev + y_curr) / 2.0
        y_prev = y_curr

    return integral
    
class Generator(Core):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        camb_params = camb.set_params(**self.cosmo_params, verbose=False)

        self.cosmo: Cosmology = Cosmology(camb_params, False)

        # we need at least lmax of 300 for this transfer code
        camb_lmax = max(self.lmax + self.lmax_buffer, 300)
        self.cosmo.compute_transfer(camb_lmax, False)
        self.cosmo.compute_c_ell()

        if self.lensing:
            c_ell_lensed = self.cosmo.c_ell["lensed_scalar"]["c_ell"][: self.nell].T
            self.c_ell_lensed = c_ell_lensed.astype(self.r_dtype)

        c_ell = self.cosmo.c_ell["unlensed_scalar"]["c_ell"][: self.nell].T
        self.c_ell = c_ell.astype(self.r_dtype)

        ns = self.cosmo_params["ns"]
        ps = self.cosmo_params["pivot_scalar"]
        shape = Shape.prim_local(ns, pivot=ps)

        self.cosmo.add_prim_reduced_bispectrum(shape, self.radii)

        pols = self.pol_idxs()
        self.cov = self.b_ell**2 * self.c_ell + self.n_ell

        inoise = np.full(self.n_ell.shape, 1e-16)
        inoise[pols, self.lmin :] = 1 / self.n_ell[pols, self.lmin :]

        # icov = np.zeros_like(cov)
        # icov[pols, self.lmin :] = 1 / self.c_ell[pols, self.lmin :]
        # icov = get_itotcov_ell(icov, inoise, self.b_ell)
        # self.icov = icov[pols, pols]

        cov = self.b_ell**2 * self.c_ell + self.n_ell
        self.icov = np.zeros_like(cov)
        self.icov[pols, self.lmin :] = 1 / cov[pols, self.lmin :]
        # self.icov = self.icov2[pols]

        # icov should be  x^icov = S^{-1} (S^{-1} + P^H N^{-1} P)^{-1} P^H N^{-1} P s,
        # where data = P s + n, where s are the spherical harmonic coefficients
        # of the signal. P = M Y B, where B is the beam, Y is spherical harmonic
        # synthesis (alm2map) and M is the pixel mask and any custom filters.
        # N^{-1} and S^{-1} are the inverse noise and signal covariance matrices,
        # respectively. ^H denotes the Hermitian transpose.
        self.estimator: KSW = KSW(
            self.cosmo.red_bispectra,
            self.icov_func,
            self.lmax,
            self.pols,
            self.precision,
        )

    def icov_func(self, alm):
        ret = np.zeros_like(alm)
        for pol in range(ret.shape[0]):
            ret[pol] = hp.almxfl(alm[pol], self.icov[pol])
        return ret

    def generate_alm(self, nsims=None, cls=None):
        """
        generates the gaussian alms using the given core.

        Parameters:
            core: The core object containing necessary parameters and data.
            nsims: The number of simulations to generate. If not provided will use the core's nsims.
            cls: The cls to use for the generation. If not provided will use the core's (unlensed) c_ell.

        Returns:
            sims: The generated alms. in TT, EE, BB, TE order.
        """
        if nsims is None:
            nsims = self.nsims

        if cls is None:
            cls = self.cov

        sims = [hp.synalm(cls, lmax=self.lmax, new=True) for _ in range(nsims)]
        sims = np.ascontiguousarray(sims) # ensure contiguous memory
        return sims

    def generate_alm_nl(self, alms):
        """
        This function calculates the non-gaussian alms using the given core and the gaussian alms.

        Parameters:
            core: The core object containing necessary parameters and data.
            alms: The input alm array.

        Returns:
            alm_nl: The calculated almng array.
        """

        pol_idxs = self.pol_idxs()
        logger.debug("Using pol_idxs: %s", pol_idxs)

        # transfer = compute_transfer(core, core.lmax, verbose=True)
        transfer = self.cosmo.transfer

        # get the transfer functions with tr_ell_k being \Delta_\ell(k)
        tr_ells = transfer["ells"]
        tr_k = transfer["k"]
        tr_ell_k = transfer["tr_ell_k"]  # this is T, E, PHI
        tr_ell_k = tr_ell_k[..., pol_idxs]

        # camb returns transfers in zeta, need to convert to phi
        tr_ell_k *= 5 / 3

        # this will be the f(k) value to be placed in the radial function
        # first will be for alpha_ell, second will be beta_ell
        # see komatsu 2003 eq 5 and 6
        f_k = np.ones((len(tr_k), 2), dtype=self.r_dtype)

        # we need to multiply the f_k by the delta_phi term, and convert from k^2 to k^-1
        # see creminelii 2006, eq 16
        # for delta_phi = A: (taken from adri's ksw)
        # Planck defines A as <phi_k1 phi_k2> = (2pi)^3 delta(k12) A / k^3.
        # CAMB defines As as <zeta_k2 zeta_k2> = (2pi)^3 delta(k12) 2 * pi^2 As / k^3.
        # So A = (3/5)^2 * 2 * pi^2 As.
        Pk = self.cosmo.camb_params.primordial_power(tr_k, 0)
        delta_phi = 2 * np.pi**2 * Pk * (3 / 5) ** 2
        f_k[:, 1] = delta_phi * tr_k ** (-3)
        rad = radial_func(f_k, tr_ell_k, tr_k, self.radii, tr_ells)

        alpha_ell = rad[..., 0]
        alpha_l = interp1d(
            tr_ells,
            alpha_ell,
            "cubic",
            1,
            bounds_error=False,
            fill_value=0,
        )(self.ells)

        beta_ell = rad[..., 1]
        beta_l = interp1d(
            tr_ells,
            beta_ell,
            "cubic",
            1,
            bounds_error=False,
            fill_value=0,
        )(self.ells)

        bl_div_cl = np.zeros_like(beta_l)
        bl_div_cl[:, self.lmin :] = beta_l[:, self.lmin :] * self.icov.T[None, self.lmin :, pol_idxs]

        # ensure all arrays are c contiguous, they wont be since we are using the interpolator which returns f contiguous
        alpha_l = np.ascontiguousarray(alpha_l)
        bl_div_cl = np.ascontiguousarray(bl_div_cl)

        # This uses joblib.parallel to generate the patches in parallel
        # by default (temp_folder=None) this will use a ram disk /dev/shm
        # if the data files are larger than the available memory, it will error
        # so we give it a temp folder to use, which wont have that problem
        temp_folder = os.environ.get("SCRATCH", None)
        logger.debug("Using temp folder for alm_nl generation: %s", temp_folder)
        parallel = Parallel(self.slurm.n_cpus, return_as="generator", temp_folder=temp_folder)
        alm_nl = np.zeros_like(alms[:, pol_idxs])

        logger.debug("Starting...")
        for sim in trange(self.nsims, desc="alm_nl"):
            generator = parallel(
                delayed(integrand)(
                    alms[sim, pol_idxs],
                    alpha_l[r],
                    bl_div_cl[r],
                    self.lmax,
                    self.nside,
                )
                for r in range(len(self.radii))
            )

            # here we use our function to calculate the integral
            alm_nl[sim] = trap_generator(generator, self.radii)

        return alm_nl


    def lens_alms(self, alms_to_lens):
        # make a copy of the alms for the lensing since they will modify them
        alm = alms_to_lens.copy()

        if not self.use_t:
            alm = np.concatenate(
                (np.zeros((self.nsims, self.ndups, 1, self.nelem)), alm), axis=2
            )
        if self.use_e:
            # need to add a zero for the spin-2 component
            alm = np.concatenate(
                (alm, np.zeros((self.nsims, self.ndups, 1, self.nelem))), axis=2
            )
        alm_lensed = np.zeros_like(alm)

        # PP PT PE
        cl_phi = self.cosmo._camb_data.get_lens_potential_cls(self.lmax, "muK", True)
        cl_phi *= self.phi_scale
        plm = utils_hp.synalm(cl_phi[:, 0], self.lmax, mmax=None)
        # phi_map = hp.alm2map(plm, nside=self.nside, pol=False)

        # transform the lensing potential into spin-1 deflection field
        fl = np.sqrt(np.arange(self.nell) * np.arange(1, self.nell + 1))
        dlm = utils_hp.almxfl(plm, fl, mmax=None, inplace=False)

        geom_info = ("healpix", {"nside": self.nside})
        geom = lenspyx.get_geom(geom_info)

        for sim, dup in product(range(self.nsims), range(self.ndups)):
            lenmap = lenspyx.alm2lenmap(
                alm[sim, dup], dlm, geometry=geom_info, nthreads=self.slurm.n_cpus
            )
            lenmap = np.ascontiguousarray(lenmap)  # convert from tuple to array

            if self.use_t:
                alm_lensed[sim, dup, 0] = geom.map2alm(
                    lenmap[0].copy(), self.lmax, self.lmax, nthreads=self.slurm.n_cpus
                )

            if self.use_e:
                alm_lensed[sim, dup, 1:] = geom.map2alm_spin(
                    lenmap[1:].copy(), 2, self.lmax, self.lmax, nthreads=self.slurm.n_cpus
                )
        return alm_lensed


    def get_fisher_iso(self, lensed=False):
        fisher = self.estimator.compute_fisher_isotropic(self.icov[self.pol_idxs()], comm=None)
        logger.debug("%s iso fisher: %s", "lensed" if lensed else "unlensed", fisher)
        return np.array([fisher]).astype(self.r_dtype)

    def run(self):
        r"""
        This code generates the alms

        $$a_{\ell m} = a_{\ell m}^{{G}} + f_{NL}^X a_{\ell m}^{NG}$$

        with

        $a_{\ell m}^{NG,loc'} = \int dr r^2 \left[ \alpha_\ell(r)\left(\int d^2 \hat{n} Y_{\ell m}^\star (\hat{n}) B(r,\hat{n})^2 \right)\right]$

        and

        $\alpha_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^2 \Delta_\ell^T(k) j_\ell(k r)$

        $\beta_\ell(r)=\frac{2}{\pi} \int_0^\infty dk k^{-1} \Delta_\phi \Delta_\ell^T(k) j_\ell(k r)$

        $B(r, \hat{n}) = \sum_{\ell,m} \frac{\beta_\ell (r)}{C_\ell} a_{\ell m} Y_{\ell m}$

        where $\Delta_\phi$ is primordial normalization, $\Delta_\ell^T(k)$ is the transfer function, $j_\ell(k r)$ are the spherical bessel functions.
        It then will optionally lens the alms and finally it will cut them into patches
        """
        pol_idxs = self.pol_idxs()

        alm_l = self.generate_alm()
        alm_nl = self.generate_alm_nl(alm_l)
        fnls = self.rng.uniform(self.fnl_min, self.fnl_max, (self.nsims, self.ndups, 1))

        # Add the duplicate dimension to the alms
        alm_l = alm_l[:, None, pol_idxs]
        alm_nl = alm_nl[:, None]

        # finally combine into the full alms and remove the monopole and dipole terms
        alms = alm_l + fnls[..., None] * alm_nl
        alms = remove_mono_dipole(alms, inplace=True)
        alms = np.ascontiguousarray(alms)

        if self.should_plot():
            # lets make a few plots for the alms
            make_alm_plots(self, alm_l[:, 0], alm_nl[:, 0], alms[:, 0])

        logger.info("Getting maps in nest ordering")
        maps = np.zeros(self.map_shape, dtype=self.r_dtype)
        for sim, dup in product(range(self.nsims), range(self.ndups)):
            rmap = hp.alm2map(alms[sim, dup].copy(), nside=self.nside, pol=self.use_pols)
            maps[sim, dup] = hp.reorder(rmap, r2n=True)

        # we go ahead and save some data so it can be freed, needed for nside > 512 to have good memory usage
        sdata = {}
        if self.save_alms:
            sdata["alm"] = alms.astype(self.c_dtype)
            sdata["alm_l"] = alm_l.astype(self.c_dtype)
            sdata["alm_nl"] = alm_nl.astype(self.c_dtype)
        # we transpose to get into a channels-last format, better for AI
        sdata["map"] = np.transpose(maps, (0, 1, 3, 2)).astype(self.r_dtype)
        save_data(self.file, sdata)
        del maps, alm_l, alm_nl  # actually force the garbage collection

        alm_lensed = maps_lensed = None
        if self.lensing:
            logger.info("Starting lensing")
            alm_lensed = self.lens_alms(alms)

            maps_lensed = np.zeros(self.map_shape, dtype=self.r_dtype)
            for sim, dup in product(range(self.nsims), range(self.ndups)):
                rmap = hp.alm2map(alm_lensed[sim, dup], nside=self.nside, pol=self.use_pols)
                maps_lensed[sim, dup] = hp.reorder(rmap, r2n=True)

            if self.should_plot():
                make_alm_plots(self, alms=alm_lensed[:, 0, pol_idxs], lensed=True)

        # note: we save as requested dtype but do calculations in complex128 since healpy needs it for alm2map
        sdata = {}
        sdata["fnl"] = fnls.astype(self.r_dtype)
        if self.slurm.is_main:
            # only save 1 copy of the fnls, so only on main
            sdata["fisher_iso"] = self.get_fisher_iso(False)
            if self.lensing:
                sdata["fisher_iso_lensed"] = self.get_fisher_iso(True)
        if self.lensing:
            if self.save_alms:
                sdata["alm_lensed"] = alm_lensed[:, :, pol_idxs].astype(self.c_dtype)
                
            sdata["map_lensed"] = np.transpose(maps_lensed[:, :, pol_idxs], (0, 1, 3, 2)).astype(self.r_dtype)
            # sdata["phi_map"] = phi_map.astype(core.r_dtype)
            
        save_data(self.file, sdata)
        logger.info("Finished %s!", self.slurm.job)


if __name__ == "__main__":
    generator = Generator()
    generator.run()
