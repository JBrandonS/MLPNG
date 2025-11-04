"""..."""

import logging
import os

import camb
import healpy as hp
import lenspyx
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from ksw import KSW, Cosmology, ReducedBispectrum, Shape
from ksw.radial_functional import radial_func
from lenspyx import utils_hp
from scipy.interpolate import interp1d
from tqdm.auto import trange

from . import Core
from .utils import (
    make_alm_plots,
    plot_cl,
    plot_cl_alm,
    plot_cl_vs,
    plot_histogram,
    plot_map_alm,
    plot_predictions,
    print_errors,
    remove_mono_dipole,
    save_data,
    setup_logging,
)


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


def trap_generator(gen, radii):
    """
    Perform trapezoidal integration using a generator. This will consume the memory as possible to help
    with memory management, this becomes needed for nside >= 512.

    Parameters:
        gen: A generator yielding the function values to integrate.
        radii: The radii values, should match the size of the generator

    Returns:
        The integral computed using Simpson's rule.
    """
    integral = 0.0
    y_prev = radii[0] ** 2 * next(gen)
    for i in range(1, len(radii)):
        y_curr = radii[i] ** 2 * next(gen)
        integral += (radii[i] - radii[i - 1]) * (y_prev + y_curr) / 2.0
        y_prev = y_curr

    return integral


class Generator(Core):
    """..."""

    def __init__(self, argv=None, log_level=logging.DEBUG, verbose=False):
        super().__init__(argv, log_level=log_level)
        self.logger = setup_logging("mlpng.generator", level=log_level)

        self.theta_batch = int(np.floor(1.5 * self.lmax + 1)) // self.slurm.n_cpus

        camb_params = camb.set_params(**self.cosmo_params, verbose=verbose)
        self.cosmo: Cosmology = Cosmology(camb_params, verbose=verbose)

        # need to run camb to get the transfer functions
        # we need at least lmax of 300 for this transfer code
        camb_lmax = max(self.lmax + self.lmax_buffer, 300)
        self.cosmo.compute_transfer(camb_lmax, verbose=verbose)
        self.cosmo.compute_c_ell(lmax=camb_lmax)

        c_ell = self.cosmo.c_ell["unlensed_scalar"]["c_ell"][: self.nell]
        self.c_ell = c_ell.T.astype(self.r_dtype)

        cov = self.b_ell**2 * self.c_ell + self.n_ell
        self.cov = remove_mono_dipole(cov)

        pols = self.pol_idxs()
        self.icov = np.zeros_like(self.cov[pols])
        self.icov[:, self.lmin :] = 1 / self.cov[pols, self.lmin :]

        if self.lensing:
            c_ell_lens = self.cosmo.c_ell["lensed_scalar"]["c_ell"][: self.nell]
            self.c_ell_lens = c_ell_lens.T.astype(self.r_dtype)

            cov_lens = self.b_ell**2 * self.c_ell_lens + self.n_ell
            self.cov_lens = remove_mono_dipole(cov_lens)

            self.icov_lens = np.zeros_like(self.cov_lens[pols])
            self.icov_lens[:, self.lmin :] = 1 / self.cov_lens[pols, self.lmin :]

            self.cl_phi = self.cosmo.c_ell["lenspotential"]["c_ell"]
            self.cl_phi = self.cl_phi[:, 0].astype(self.r_dtype)

    def get_ksw(
        self,
        shape_str,
        step=True,
        step_alms=None,
        n_steps=None,
        c_ells=None,
        lensed=False,
    ):
        """
        This function returns a KSW estimator for the given shape and parameters.

        Parameters:
            shape_str: The shape to use for the KSW estimator, one of ["local", "equilateral", "orthogonal"].
            step: If True, will initialize the KSW with a step batch.
            step_alms: The alms to use for the step batch. If None, will generate new alms.
            n_steps: The number of steps to use for the step batch. If None, will use the core's mc_steps. If step_alms is provided,
                the smaller value between n_steps and the number of step_alms is used.
            icov: The inverse covariance matrix to use. If None, will use the core's icov.
        Returns:
            ksw: The KSW estimator for the given shape and parameters.
        """
        ns = self.cosmo_params["ns"]
        ps = self.cosmo_params["pivot_scalar"]

        if c_ells is not None:
            icov = np.zeros_like(c_ells)
            icov[..., self.lmin :] = 1 / c_ells[..., self.lmin :]
        else:
            icov = None

        match shape_str:
            case "local":
                shape = Shape.prim_local(ns, pivot=ps)
            case "equilateral":
                shape = Shape.prim_equilateral(ns, pivot=ps)
            case "orthogonal":
                shape = Shape.prim_orthogonal(ns, pivot=ps)
            case _:
                raise ValueError(f"Unknown shape {shape_str}")

        #  hack to remove the previous bispectrum, if there is one
        self.cosmo.red_bispectra = []
        self.cosmo.add_prim_reduced_bispectrum(shape, self.radii)

        # icov should be  x^icov = S^{-1} (S^{-1} + P^H N^{-1} P)^{-1} P^H N^{-1} P s,
        # where data = P s + n, where s are the spherical harmonic coefficients
        # of the signal. P = M Y B, where B is the beam, Y is spherical harmonic
        # synthesis (alm2map) and M is the pixel mask and any custom filters.
        # N^{-1} and S^{-1} are the inverse noise and signal covariance matrices,
        # respectively. ^H denotes the Hermitian transpose.
        ksw = KSW(
            self.cosmo.red_bispectra,
            lambda a: self.icov_func(a, icov, lensed),
            self.lmax,
            self.pols,
            self.precision,
        )
        ksw.start_from_read_state(self.mc_file)
        return ksw

    def icov_func(self, alm, icov=None, lensed=False):
        """
        This function applies the inverse covariance to the alms for use with the KSW estimator.

        Parameters:
            alm: The input alm array.
            icov: The inverse covariance matrix to use. If None, will use the core's icov.
        Returns:
            ret: The alms after applying the inverse covariance.
        """
        if icov is None:
            icov = self.icov_lens if lensed else self.icov

        ret = np.zeros_like(alm)
        for pol in range(ret.shape[0]):
            ret[pol] = hp.almxfl(alm[pol], icov[pol])
        return ret

    def generate_alm(self, nsims=None, c_ells=None, lensed=False) -> np.ndarray:
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

        if c_ells is None:
            c_ells = self.cov_lens if lensed else self.cov

        sims = [hp.synalm(c_ells, lmax=self.lmax, new=True) for _ in range(nsims)]
        sims = remove_mono_dipole(np.array(sims))
        return np.ascontiguousarray(sims)  # ensure contiguous memory

    def generate_alm_nl(self, alms, icov=None, lensed=False):
        """
        This function calculates the non-gaussian alms using the given core and the gaussian alms.

        Parameters:
            core: The core object containing necessary parameters and data.
            alms: The input alm array.

        Returns:
            alm_nl: The calculated almng array.
        """

        if icov is None:
            icov = self.icov_lens if lensed else self.icov

        pol_idxs = self.pol_idxs()
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
        bl_div_cl[:, self.lmin :] = beta_l[:, self.lmin :] * icov.T[None, self.lmin :]

        # ensure all arrays are c contiguous, they wont be since we are using the interpolator which returns f contiguous
        alpha_l = np.ascontiguousarray(alpha_l)
        bl_div_cl = np.ascontiguousarray(bl_div_cl)

        # This uses joblib.parallel to generate the patches in parallel
        # by default (temp_folder=None) this will use a ram disk /dev/shm
        # if the data files are larger than the available memory, it will error
        # so we give it a temp folder to use, which wont have that problem
        temp_folder = os.environ.get("SCRATCH", None)
        self.logger.debug("Using temp folder for alm_nl generation: %s", temp_folder)
        parallel = Parallel(
            self.slurm.n_cpus,
            return_as="generator",
            temp_folder=temp_folder,
        )
        alm_nl = np.zeros_like(alms[:, pol_idxs])

        self.logger.debug("Starting...")
        for sim in trange(self.nsims, desc="alm_nl"):
            gen = parallel(
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
            alm_nl[sim] = trap_generator(gen, self.radii)

        self.logger.debug("done")
        return alm_nl

    def lens_alms(self, alm, verbose=False):
        """This function lenses the alms using the lenspyx library."""
        lmax = self.lmax + self.lmax_buffer
        fl = np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2))

        cl_phi = self.cl_phi * self.phi_scale
        alm_phi = [
            hp.synalm(cl_phi, new=True, verbose=verbose) for _ in range(self.nsims)
        ]

        geom_info = ("healpix", {"nside": self.nside})
        geom = lenspyx.get_geom(geom_info)

        # we need to generate the lensed alm shape, adding a pol for E if needed
        # note: E is not fully supported yet, and im out of time so we will not support it
        alm_shape = alm.shape
        lensed_shape = (alm_shape[0], 3 if self.use_e else 1, alm_shape[2])

        alm_lensed = np.zeros(lensed_shape, alm.dtype)
        for sim in range(self.nsims):
            dlm = hp.almxfl(alm_phi[sim], fl)

            # we have to split the lensing into the T and EB components
            # not really supported right now but maybe not needed
            if self.use_t:
                lenmap = lenspyx.alm2lenmap(
                    alm[sim, 0],
                    dlm,
                    geometry=geom_info,
                    nthreads=self.slurm.n_cpus,
                )
                lenmap = np.ascontiguousarray(lenmap)

                alm_lensed[sim, 0] = geom.map2alm(
                    lenmap,
                    self.lmax,
                    self.lmax,
                    nthreads=self.slurm.n_cpus,
                )

            if self.use_e:  # not really supported yet
                lenmap = lenspyx.alm2lenmap_spin(
                    alm[sim, 1:],
                    dlm,
                    2,
                    geometry=geom_info,
                    nthreads=self.slurm.n_cpus,
                )
                lenmap = np.ascontiguousarray(lenmap)  # convert from tuple to array

                alm_lensed[sim, 1:] = geom.map2alm_spin(
                    lenmap,
                    2,
                    self.lmax,
                    self.lmax,
                    nthreads=self.slurm.n_cpus,
                )

        return alm_lensed, alm_phi

    def generate_alm_nl_shape(self, alms, shape, ksw=None, icov=None, lensed=False):
        """
        This function generates the non-gaussian alms for a given shape using SZ MC method via KSW

        Parameters:
            alms: The input alm array.
            shape: The shape to use for the non-gaussian alms.
            lensed: If True, will use the lensed cov in the ksw code.
        Returns:
            alm_ng: The calculated alm_ng array for the given shape.
        """
        pols = self.pol_idxs()

        if ksw is None:
            ksw = self.get_ksw(shape, step_alms=alms, lensed=lensed)

        alm_ng = np.zeros_like(alms[:, pols])
        for i in trange(self.nsims, desc=f"alm_ng {shape}"):
            alm_icov = self.icov_func(alms[i, pols], icov=icov, lensed=lensed)
            alm_ng[i] = ksw.compute_ng_sim(alm_icov, theta_batch=self.theta_batch)

        alm_ng = remove_mono_dipole(alm_ng)
        return np.ascontiguousarray(alm_ng)

    def compute_fisher_shapes(self, shapes, lensed=False):
        """
        This function computes the fisher matrix for the given shapes.

        Parameters:
            shapes: A list of shapes to compute the fisher matrix for.
            lensed: If True, will use the lensed inverse covariance.
        Returns:
            fisher: The computed fisher matrix for the given shapes.
        """

        # shape here doesn't matter, we dont step so this is fast
        estimator = self.get_ksw(shapes[0], step=False, lensed=lensed)

        # if we only are using 1 shape we just compute the fisher for that shape
        # if len(shapes) == 1:
        #     return [estimator.compute_fisher_isotropic(icov_)]

        # get a lot of standard parameters from the cosmology
        tr_ell_k = self.cosmo.transfer["tr_ell_k"]
        k = self.cosmo.transfer["k"]
        ells_sparse = self.cosmo.transfer["ells"]
        ps = self.cosmo_params["pivot_scalar"]
        ns = self.cosmo_params["ns"]
        As = self.cosmo.camb_params.InitPower.As

        red_bispectra = []
        amp_factor = 2 * (2 * np.pi**2 * As) ** 2 * (3 / 5)

        for shape in shapes:
            match shape:
                case "local":
                    shape = Shape.prim_local(ns, pivot=ps)
                case "equilateral":
                    shape = Shape.prim_equilateral(ns, pivot=ps)
                case "orthogonal":
                    shape = Shape.prim_orthogonal(ns, pivot=ps)
                case _:
                    raise ValueError(f"Unknown shape {shape}")

            # this is cosmology.add_prim_reduced_bispectrum
            f_k = shape.get_f_k(k)
            amps = np.asarray(shape.amps).copy()
            amps *= amp_factor

            red_bisp = radial_func(f_k, tr_ell_k, k, self.radii, ells_sparse)
            factors, rule, weights = self.cosmo._parse_prim_reduced_bispec(
                red_bisp, self.radii, shape.rule, amps
            )
            red_bispectra.append(
                ReducedBispectrum(factors, rule, weights, ells_sparse, shape.name)
            )

        icov = self.icov_lens if lensed else self.icov
        return estimator.compute_fisher_multi(icov, red_bispectra)

    def run(self, verbose=False):
        """This function runs the generator, generating the alms and calculating the non-gaussian alms."""

        pol_idxs = self.pol_idxs()

        # get the unlensed alms, go ahead and save them and then run the full calculations
        alm_l = self.generate_alm()
        sdata = {"alm_l": {"unlensed": alm_l[:, pol_idxs].astype(self.c_dtype)}}
        save_data(self.file, sdata, verbose=verbose)

        self._run(alm_l, False, verbose=verbose)

        if self.lensing:
            # now we lens the alms and save them
            self.logger.debug("Lensing alms...")
            alm_lens, alm_phi = self.lens_alms(alm_l)

            # go ahead and save the data here
            sdata = {
                "alm_l": {"lensed": alm_lens[:, pol_idxs].astype(self.c_dtype)},
                "alm_phi": np.array(alm_phi).astype(self.c_dtype),
            }
            save_data(self.file, sdata, verbose=verbose)

            self._run(alm_lens, lensed=True, verbose=verbose)

    def _run(self, alm_l, lensed, verbose=False):
        """
        This function runs the generator for a given set of alms, calculating the non-gaussian alms and fisher matrices.

        Parameters:
            alm_l: The input alm array.
            lensed: If True, will use the lensed cov in the ksw code.
        """
        c_ells = self.c_ell_lens if lensed else self.c_ell
        icov = self.icov_lens if lensed else self.icov
        l_str = "lensed" if lensed else "unlensed"
        pol_idxs = self.pol_idxs()

        if self.slurm.is_main:
            # we grab the fisher matrix here to save it, and calculate the marginal likelihoods
            self.logger.debug(
                "Computing fisher matrix for %s shapes: %s", l_str, self.shapes
            )

        fisher_mat = np.array(self.compute_fisher_shapes(self.shapes, lensed))
        if len(self.shapes) > 1:
            marg_likes = np.sqrt(np.diag(np.linalg.inv(fisher_mat)))
        else:
            marg_likes = np.sqrt(1 / fisher_mat)
        self.logger.debug("Marginal likelihoods: %s", marg_likes)

        sdata = {
            "fisher_matrix": {l_str: fisher_mat.astype(self.r_dtype)},
            "marginal_likelihoods": {l_str: marg_likes.astype(self.r_dtype)},
        }
        save_data(self.file, sdata, verbose=verbose)

        for shape in self.shapes:
            self.logger.debug("Starting %s %s", l_str, shape)
            ksw = self.get_ksw(shape, c_ells=c_ells, lensed=lensed)

            fisher = ksw.compute_fisher()
            self.logger.debug("fisher: %s, std div: %s", fisher, 1 / np.sqrt(fisher))

            self.logger.debug("Computing %s alm_nl for shape %s", l_str, shape)
            if shape == "local":
                # use hanson method for local shape
                alm_nl = self.generate_alm_nl(alm_l, icov, lensed)
            else:
                # for the shapes we just use the KSW method
                alm_nl = self.generate_alm_nl_shape(alm_l, shape, ksw, icov, lensed)

            sdata = {
                "alm_nl": {l_str: {shape: alm_nl.astype(self.c_dtype)}},
                "fisher": {l_str: {shape: [fisher.astype(self.r_dtype)]}},
            }
            save_data(self.file, sdata, verbose=verbose)

            if self.should_plot():
                self.logger.debug("Making alm plots for %s", shape)
                make_alm_plots(
                    self,
                    shape,
                    alm_l[:, pol_idxs],
                    alm_nl[:, pol_idxs],
                    lensed=lensed,
                )

            if self.slurm.is_main and self.estimate:
                self.logger.debug("Computing estimates for %s %s", l_str, shape)

                # make our alms
                fnl = self.rng.uniform(self.fnl_min, self.fnl_max, (self.nsims, 1, 1))
                alm = alm_l + fnl * alm_nl

                n_estimates = min(self.nsims, self.num_estimates)
                estimates, _, _, _ = ksw.compute_estimate_batch(
                    lambda idx: self.icov_func(alm[idx, pol_idxs], icov, lensed),
                    range(n_estimates),
                    fisher=fisher,
                )
                estimates = estimates.T

                print_errors(fnl, estimates, fisher)
                if self.should_plot():
                    base = f"est_{shape}_{l_str}"
                    plot_dir = os.path.join(self.dirs["plot"], self.name, "estimates")
                    plot_predictions(
                        fnl,
                        estimates,
                        sigma=1 / np.sqrt(fisher),  # type: ignore
                        save_file=self.get_plot_file(f"{base}_preds", plot_dir),
                    )
                    plot_histogram(
                        fnl,
                        estimates,
                        save_file=self.get_plot_file(f"{base}_hist", plot_dir),
                    )

            self.logger.info("Finished %s %s!", l_str, shape)


if __name__ == "__main__":
    generator = Generator()

    # check if we actually need to run / clean files
    generator.check_existing_data_file()

    generator.run(verbose=False)
