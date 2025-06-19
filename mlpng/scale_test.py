import os
import logging
import h5py
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import healpy as hp
from itertools import product

from mlpng.utils import *
from mlpng.generator import Generator
from mlpng.utils.utils import *
from mlpng.utils.plots import plot_cl, plot_map, plot_map_alm

import lenspyx
from lenspyx import utils_hp

mpi_comm = None

# disable a root logger error from CAMB
logging.getLogger("root").setLevel(logging.ERROR)

for nside in [32, 64, 128, 256]:
    print("Nside:", nside)
    for phi in [1.0, 1.1, 2, 3, 5, 10, 30, 100]:
        generator = Generator(
            [
                f"settings/n{nside}.json",
                "--nsims",
                "1",
                "--pols",
                "T",
                "--phi_scale",
                str(phi),
                "--shape",
                "local",
                "--no-noise",
                # "--double_precision",
            ],
            logging.ERROR,
        )

        pol_idxs = generator.pol_idxs()

        sims = [
            hp.synalm(generator.c_ell, lmax=generator.lmax, new=True)
            for _ in range(generator.nsims)
        ]
        sims = remove_mono_dipole(np.array(sims))
        alm_l = np.ascontiguousarray(sims)[0, 0]  # ensure contiguous memory

        # we use a buffer for the lmax to ensure we can compute the lensing potential without error
        lmax = generator.lmax + generator.lmax_buffer

        # our conversion factor = sqrt(l * (l + 1))
        fl = np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2))

        # cl_phi is PP PT PE, we only want the phi phi part
        cl_phi = generator.cosmo._camb_data.get_lens_potential_cls(
            lmax,
            "muK",
            raw_cl=True,
        )
        cl_phi = cl_phi[..., 0]

        alm_phi = hp.synalm(
            cl_phi,
            lmax,
            lmax,
            new=True,
            verbose=False,
        )
        alm_phi *= generator.phi_scale  # scale the lensing potential by phi_scale

        dlm = hp.almxfl(alm_phi, fl)

        lenmap = lenspyx.alm2lenmap(
            alm_l,
            dlm,
            geometry=("healpix", {"nside": generator.nside}),
            nthreads=generator.slurm.n_cpus,
        )
        lenmap = remove_mono_dipole(lenmap)

        lens_cls = generator.cosmo.c_ell["lensed_scalar"]["c_ell"][: generator.nell, 0]
        lens_cls *= generator.phi_scale

        icov = 1 / lens_cls
        icov[: generator.lmin] = 0.0
        icov = np.array(icov)

        fisher_mat = np.array(generator.compute_fisher_shapes(generator.shapes))
        if len(generator.shapes) > 1:
            marg_likes = np.sqrt(np.diag(np.linalg.inv(fisher_mat)))
        else:
            marg_likes = np.sqrt(1 / fisher_mat)

        fisher_mat = generator.compute_fisher_shapes(generator.shapes, icov=icov)
        if len(generator.shapes) > 1:
            marg_likes_lens = np.sqrt(np.diag(np.linalg.inv(fisher_mat)))
        else:
            marg_likes_lens = np.sqrt(1 / fisher_mat)

        diff = marg_likes_lens - marg_likes
        ratio = marg_likes_lens / marg_likes

        print(
            f"| {generator.phi_scale} | {float(marg_likes)} | {float(marg_likes_lens)} | {float(diff)} | {float(ratio)} |"
        )
