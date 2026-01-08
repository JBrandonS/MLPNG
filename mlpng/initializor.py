"""..."""

import logging
import os
import sys
import warnings

import camb
import healpy as hp
import numpy as np
from ksw import KSW, Cosmology, Shape
from mpi4py import MPI

from . import Core
from .utils import remove_mono_dipole, setup_logging

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0


class Initializor(Core):
    """..."""

    def __init__(self, argv=None, log_level=None, verbose=False):
        if log_level is None:
            log_level = logging.DEBUG if mpi_root else logging.ERROR

        warnings.filterwarnings("ignore", message=".*power_spectra_from_transfer.*")

        super().__init__(argv)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # quick check before doing any calculations
        self.check_existing_data_file()

        self.logger.debug("MPI Rank: %s", mpi_rank)
        self.logger.debug("MPI Size: %s", mpi_size)
        self.logger.debug("MPI Root: %s", mpi_root)

        self.theta_batch = int(np.floor(1.5 * self.lmax + 1)) // self.slurm.n_cpus

        camb_params = camb.set_params(**self.cosmo_params, verbose=verbose)
        self.cosmo: Cosmology = Cosmology(camb_params, verbose=verbose)

        # need to run camb to get the transfer functions
        # we need at least lmax of 300 for this transfer code
        camb_lmax = max(self.lmax + self.lmax_buffer, 300)

        self.logger.debug("Computing transfers")
        self.cosmo.compute_transfer(camb_lmax, verbose=verbose)

        self.logger.debug("Computing C_ells")
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

    def generate_alm(self, lensed=False) -> np.ndarray:
        if lensed:
            cov = self.cov_lens
        else:
            cov = self.cov

        sims = hp.synalm(cov, lmax=self.lmax, new=True)
        return remove_mono_dipole(sims, True)

    def get_ksw(
        self,
        shape_str,
        n_steps=None,
        lensed=False,
    ):
        """
        This function returns a KSW estimator for the given shape and parameters.

        Parameters:
            shape_str: The shape to use for the KSW estimator, one of ["local", "equilateral", "orthogonal"].
            step_alms: The alms to use for the step batch. If None, will generate new alms.
            n_steps: The number of steps to use for the step batch. If None, will use the core's mc_steps. If step_alms is provided,
                the smaller value between n_steps and the number of step_alms is used.
        Returns:
            ksw: The KSW estimator for the given shape and parameters.
        """
        ns = self.cosmo_params["ns"]
        ps = self.cosmo_params["pivot_scalar"]

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
            lambda a: self.icov_func(a, lensed=lensed),
            self.lmax,
            self.pols,
            self.precision,
        )

        nsteps = n_steps if n_steps is not None else self.mc_steps

        self.logger.debug(
            "Initalizing KSW for %s %s with %s steps",
            "lensed" if lensed else "unlensed",
            shape_str,
            nsteps,
        )

        def get_step_alm(idx: int):
            alm = self.generate_alm()[self.pol_idxs()]
            return self.icov_func(alm, lensed=lensed)

        ksw.step_batch(
            get_step_alm,
            range(nsteps),
            comm=mpi_comm,
            theta_batch=self.theta_batch,
        )

        return ksw

    def icov_func(self, alm, icov=None, lensed=False):
        """
        This function applies the inverse covariance to the alms for use with the KSW estimator.

        Parameters:
            alm: The input alm array.
            icov: The inverse covariance matrix to use. If None, will use the core's icov.
            lensed: Whether to use the lensed inverse covariance, only used if icov is None.
        Returns:
            ret: The alms after applying the inverse covariance.
        """
        if icov is None:
            icov = self.icov_lens if lensed else self.icov

        ret = np.zeros_like(alm)
        for pol in range(ret.shape[0]):
            ret[pol] = hp.almxfl(alm[pol], icov[pol])
        return ret

    def check_existing_data_file(self):
        """Check if MC files already exist and handle them based on the `force_gen` setting.
        If files exist and `force_gen` is True, they are removed."""
        should_exit = False
        existing_files = []
        missing_files = []
        opts = [False, True] if self.lensing else [False]

        if mpi_root:
            for lensed in opts:
                for shape in self.shapes:
                    mc_file = self.get_mc_file(shape, lensed=lensed)
                    if os.path.exists(mc_file):
                        existing_files.append(mc_file)
                    else:
                        missing_files.append(mc_file)

            # Check if only some files exist
            if existing_files and missing_files and not self.force_gen:
                self.logger.error(
                    "Partial MC files found! Some exist, some don't. Cannot proceed without --force_generation."
                )
                self.logger.error("Existing files: %s", existing_files)
                self.logger.error("Missing files: %s", missing_files)
                should_exit = True
            elif existing_files and not self.force_gen:
                self.logger.info("MC files already exist, exiting")
                should_exit = True
            elif existing_files and self.force_gen:
                for mc_file in existing_files:
                    self.logger.info("Removing existing MC file '%s'", mc_file)
                    os.remove(mc_file)

        mpi_comm.Barrier()

        should_exit = mpi_comm.bcast(should_exit, root=0)
        if should_exit:
            mpi_comm.Abort()
            sys.exit(0)

    def run(self):
        opts = [False, True] if self.lensing else [False]
        for lensed in opts:
            l_str = "lensed" if lensed else "unlensed"

            for shape in self.shapes:
                self.logger.debug("Starting %s %s", l_str, shape)
                ksw = self.get_ksw(shape, lensed=lensed)
                mc_file = self.get_mc_file(shape, lensed=lensed)
                ksw.write_state(mc_file, comm=mpi_comm)
                self.logger.info("Finished %s %s! Saved to %s", l_str, shape, mc_file)

                # Wait for all processes to reach this point, not sure if needed
                mpi_comm.Barrier()


if __name__ == "__main__":
    setup_logging(__name__, level=logging.DEBUG if mpi_root else logging.ERROR)

    init = Initializor()
    init.run()
