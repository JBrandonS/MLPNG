from decimal import Overflow
import logging
import os
import sys
from copy import deepcopy
from typing import Literal, Optional

import camb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyfftw
from matplotlib.colors import LogNorm, Normalize
from numba import njit, prange, set_num_threads
from scipy.interpolate import interp1d
from tqdm import tqdm


class DensityField2D():
    SaveType = Literal['npy', 'tf', 'both', 'none']
    InterpType = Literal['linear', 'cubic']
    # General \Lambda CDM parameters
    default_cosmo_params = {
        'h': 0.6711,
        'r': 0,
        'As': 2.13e-09,
        'A': 2*np.pi*np.pi*2.13e-09,
        'ns': 0.9624,
        'kpivot': 0.05,
        'z_recomb': 1090.48,
        'ombh2': 0.02233,
        'omch2': 0.1198,
        'tau': 0.0561,
        'lmax': 10000,
        'accuracy_boost': 1,
    }

    def __init__(self, BoxSize, grid,
                n_threads=1, FFTW_WISDOM='FFTW_ESTIMATE', dtype=np.float64,
                cosmo_params=default_cosmo_params, camb_params_obj=None, init_camb=True, interp_kind: InterpType = 'cubic',
                qs=None, transfers=None, ls=None, ells=None, d_A=None, kgrid=None,
                log_level=logging.WARNING, verbose=False):
        logging.basicConfig(format='[%(asctime)s | %(name)s -%(funcName)20s()] | %(levelname)s : %(message)s', stream=sys.stdout)
        self.log = logging.getLogger(__name__)
        self.log.setLevel(log_level)
        
        self.log.debug('Initializing CMBField...')
        assert grid % 2 == 0, f"choose an even grid size. Got: {grid}"

        # Size of periodic box e.g. in Mpc/h
        self.BoxSize = BoxSize
        # Number of grid-points per dimension
        self.grid = grid
        # Physical size of grid-cell
        self.cell_size = self.BoxSize/self.grid
        # Fundamental mode of the box
        self.kF = 2*np.pi / self.BoxSize
        # Nyquist frequency of the grid
        self.kNyq = self.kF * self.grid / 2

        # Number of threads to be used
        self.n_threads = n_threads
        # FFTW WISDOM to be used (default FFTW_ESTIMATE)
        self.FFTW_WISDOM = FFTW_WISDOM

        # Numerical precision either in np.floatXX or 'floatXX' format
        self.r_dtype = dtype
        if self.r_dtype == np.float32 or self.r_dtype == 'float32':
            self.c_dtype = np.complex64
        elif self.r_dtype == np.float64 or self.r_dtype == 'float64':
            self.c_dtype = np.complex128

        # Setup FFTW grids and transforms
        self.rshape = np.array([self.grid, self.grid])
        self.cshape = np.array([self.grid, self.grid//2 + 1])
        self.r_fftgrid = pyfftw.empty_aligned(self.rshape, dtype=self.r_dtype)
        self.c_fftgrid = pyfftw.empty_aligned(self.cshape, dtype=self.c_dtype)

        self.fft_r2c = pyfftw.FFTW(self.r_fftgrid, self.c_fftgrid, axes=tuple(range(2)),
                                direction="FFTW_FORWARD", threads=self.n_threads, flags=[self.FFTW_WISDOM])

        self.fft_c2r = pyfftw.FFTW(self.c_fftgrid, self.r_fftgrid, axes=tuple(range(2)),
                                direction="FFTW_BACKWARD", threads=self.n_threads, flags=[self.FFTW_WISDOM])

        # Allocate array for real and complex grids to be saved and in case given load density and FFT
        self.r_delta = np.zeros_like(self.r_fftgrid)
        self.c_delta = np.zeros_like(self.c_fftgrid)

        if kgrid is None:
            # Setup mesh and k-space grid
            self.kx = 2 * np.pi * np.fft.fftfreq(self.grid, self.cell_size)
            # Note rfft.
            self.ky = 2 * np.pi * np.fft.rfftfreq(self.grid, self.cell_size)
            self.kmesh = np.meshgrid(self.kx, self.ky, indexing="ij")
            self.kgrid = np.sqrt(self.kmesh[0]**2 + self.kmesh[1]**2)
        else:
            self.kx = self.ky = self.kmesh = None
            self.kgrid = kgrid

        # Setup Camb parameters and run camb if given
        self.cosmo = cosmo_params
        self.h = cosmo_params['h']

        if camb_params_obj is None and init_camb:
            if verbose:
                self.log.info('Setting CAMB parameters based on provided cosmology:\n{}'.format(self.cosmo))
            self.camb_params = camb.CAMBparams()
            self.camb_params.set_cosmology(
                H0=self.cosmo['h']*100, ombh2=self.cosmo['ombh2'], omch2=self.cosmo['omch2'], tau=self.cosmo['tau'])
            self.camb_params.InitPower.set_params(
                As=self.cosmo['As'], ns=self.cosmo['ns'], r=self.cosmo['r'], pivot_scalar=self.cosmo['kpivot'])  # type: ignore
            self.camb_params.set_for_lmax(
                self.cosmo['lmax'], lens_potential_accuracy=self.cosmo['accuracy_boost'])
            self.camb_params.set_accuracy(
                AccuracyBoost=self.cosmo['accuracy_boost'])
            self.camb_params.Want_CMB = True
            self.camb_params.WantTransfer = True
        else:
            self.camb_params = camb_params_obj

        self.interp_kind = interp_kind
        if init_camb:
            self.run_camb()
        else:
            self.results = self.transfer_func = None
            self.Ls = ls
            self.qs = qs
            self.transfers = transfers
            self.ells = ells
            self.d_A = d_A

        # Set numba threads for parallelization (grid>512)
        set_num_threads(n_threads)
        if verbose:
            self.log.info(f'Finished generating CMBField with resolution {self.grid}, box size {self.BoxSize} Mpc/h, and cell size {self.cell_size}')
    
    def run_camb(self, params=None, interp_kind: InterpType='cubic'):
        if params is None:
            self.log.debug('Running CAMB with global parameters...')
            params = self.camb_params
        else:
            self.log.debug('Running CAMB with provided parameters {}...')

        if interp_kind is None:
            interp_kind = self.interp_kind

        self.results = camb.get_results(params)
        self.d_A = self.results.angular_diameter_distance(
            self.cosmo['z_recomb']) * 1000

        self.ells = np.round(self.kgrid * self.h * self.d_A).astype(np.int64)
        self.Ls, self.qs, self.transfer_func = self.results.get_cmb_transfer_data().get_transfer()
        if interp_kind == 'cubic':
            self.log.debug(
                'Interpolating transfer functions with cubic splines.')
            self.transfers = [interp1d(
                self.Ls, self.transfer_func[:, i], kind='cubic') for i in range(len(self.qs))]
        elif interp_kind == 'linear':
            self.transfers = np.array([np.interp(
                self.Ls, self.Ls, self.transfer_func[:, i]) for i in range(len(self.qs))], dtype=object)
        else:
            assert False, self.log.fatal('Invalid interpolation kind: {interp_kind}')
        self.log.debug('Finished running CAMB.')

    def Load_r2c(self, delta_r):
        assert (delta_r.shape == self.r_delta.shape), "mismatching grid sizes"
        self.r_delta = delta_r.copy()
        np.copyto(self.r_fftgrid, delta_r)
        self.fft_r2c()
        self.c_delta = self.c_fftgrid.copy()

    def Load_c2r(self, delta_c):
        assert (delta_c.shape == self.c_delta.shape), "mismatching grid sizes"
        self.c_delta = delta_c.copy()
        np.copyto(self.c_fftgrid, delta_c)
        self.fft_c2r()
        self.r_delta = self.r_fftgrid.copy()

    def Pk(self, kmax=None):
        if kmax is None:
            kmax = self.kNyq

        kmax_n = np.int64((np.ceil(kmax/self.kF)))
        ks = np.zeros(kmax_n-1)
        ns = np.zeros(kmax_n-1)
        Pks = np.zeros(kmax_n-1)

        for kxi in range(self.cshape[0]):
            for kyi in range(self.cshape[1]):
                if kxi == 0 and kyi == 0:
                    continue

                k = self.kgrid[kxi, kyi]
                if k >= kmax:
                    continue
                k_index = np.int64((np.floor(k/self.kF)))

                delta_r = self.c_delta[kxi, kyi].real
                delta_i = self.c_delta[kxi, kyi].imag
                delta2 = delta_r**2 + delta_i**2

                Pks[k_index-1] += delta2
                ks[k_index-1] += k
                ns[k_index-1] += 1.

        ks /= ns
        Pks *= 1/ns * self.BoxSize**3 / self.grid**4
        return ks, Pks, ns

    def mask_c2r(self, delta_c, k_low, k_high):
        np.copyto(self.c_fftgrid, delta_c)
        self.c_fftgrid[self.kgrid >= k_high] = 0.+0.j
        self.c_fftgrid[self.kgrid < k_low] = 0.+0.j
        self.fft_c2r()
        return self.r_fftgrid

    def _Bk_counts(self, fc, dk, NBmax, triangle_type, verbose=False, data_dir='data/static'):
        file_name = f"{data_dir}/FFTest2D_BkCounts_LBox{self.BoxSize}_Grid{self.grid}_Binning{dk}kF_fc{fc}_NBins{NBmax}_TriangleType{triangle_type}.npy"
        if os.path.exists(file_name):
            if verbose:
                self.log.info(f"Loading Counts from {file_name}")
            counts = np.load(file_name, allow_pickle=True).item()
            if verbose:
                self.log.info(
                    f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type})")
            return counts

        counts = {}
        counts['counts_P'] = np.zeros(NBmax)

        if triangle_type == 'All':
            counts['bin_centers'] = np.array([(i, j, l)
                                              for i in fc+np.arange(0, (NBmax))*dk
                                              for j in np.arange(fc, i+1, dk)
                                              for l in np.arange(fc, j+1, dk) if i <= j+l+dk])

        elif triangle_type == 'Squeezed':
            counts['bin_centers'] = np.array([(i, i, j)
                                              for ji, j in enumerate(fc + np.arange(0, NBmax)*dk)
                                              for i in fc+np.arange(ji+1, NBmax)*dk])
        elif triangle_type == 'Equilateral':
            counts['bin_centers'] = np.array([(i, i, i)
                                              for i in fc+np.arange(0, NBmax)*dk])
        if verbose:
            self.log.info(
                f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type})")

        if verbose:
            self.log.info(f"Creating Grids for Counts...")
        c_ones = np.ones_like(self.c_fftgrid)
        r_ones_shells = np.zeros(
            (NBmax, self.rshape[0], self.rshape[1]), dtype=self.r_dtype)
        for i in tqdm(range(NBmax), disable=not verbose):
            k_low = self.kF * (fc + dk * i - dk/2)
            k_high = self.kF * (fc + dk * i + dk/2)
            r_ones_shells[i] = self.mask_c2r(c_ones, k_low, k_high)

        if verbose:
            self.log.info("Computing Powerspectrum Counts...")
        counts['counts_P'] = _Pk_shells(r_ones_shells) * self.grid**2
        if verbose:
            self.log.info("done!")

        if verbose:
            self.log.info("Computing Triangle Counts...")
        bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)
        counts['counts_B'] = _Bk_shells(
            r_ones_shells, bin_indices) * self.grid**4
        if verbose:
            self.log.info("done!")

        np.save(file_name, counts)  # type: ignore
        if verbose:
            self.log.info(f"Saved Triangle Counts to {file_name}")

        return counts

    def Bk(self, fc, dk, NBmax, triangle_type='All', verbose=False):
        # Computes binned bispectrum of field for given binning and triangles

        # fc: center of first bin in units of the fundamental mode
        # dk: width of the bin in units of the fundamental mode
        # NBmax: total number of momentum bins
        # Such that bins are given by kf*[(fc + i)±dk/2 for i in range(NBmax)]
        # triangle_type:
        # 'All': include all shapes of triangles
        # 'Squeezed': only triangles k_1 > k_2 = k_3
        # 'Equilateral': include only triangles k_1 = k_2 = k_3
        # verbose: self.log.info progress statements

        counts = self._Bk_counts(fc, dk, NBmax, triangle_type, verbose=verbose)

        r_delta_shells = np.zeros(
            (NBmax, self.rshape[0], self.rshape[1]), dtype=self.r_dtype)

        if verbose:
            self.log.info("Creating Grids for Measurements...")
        for i in tqdm(range(NBmax), disable=not verbose):
            k_low = self.kF * (fc + dk * i - dk/2)
            k_high = self.kF * (fc + dk * i + dk/2)
            r_delta_shells[i] = self.mask_c2r(self.c_delta, k_low, k_high)

        if verbose:
            self.log.info("Computing Powerspectrum...")
        P = _Pk_shells(r_delta_shells) * self.BoxSize**3 / \
            counts['counts_P'] / self.grid**2
        if verbose:
            self.log.info("done!")

        if verbose:
            self.log.info("Computing Bispectrum...")
        bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)
        B = _Bk_shells(r_delta_shells, bin_indices) * \
            self.BoxSize**6 / self.grid**2
        if verbose:
            self.log.info("done! \n")

        result = np.ones((len(counts['bin_centers']), 8))
        result[:, :3] = counts['bin_centers']
        result[:, 3:6] = P[bin_indices]
        result[:, 6] = B/counts['counts_B']
        result[:, 7] = counts['counts_B']

        return result

    def GenerateLinearField(self, zi, k_cut_low=None, k_cut_high=None, fnl=0., seed=0, verbose=False):

        # zi: Index of desired redshift z={0,3,10,30,50,100}
        # k_cut_low: cut off density field at k < k_cut_low
        # k_cut_high: cut off density field at k >= k_cut_high
        # fnl: parameter for amount of local primordial non-Gaussianity
        # seed: random seed for realization
        # verbose: self.log.infos progress statements

        if verbose: self.log.info(f"Generating linear field with seed {seed}")
        kLin, TFLin = np.loadtxt("LinearTransfer.dat")[:, [0, 1+zi]].T

        if verbose:
            self.log.info("Making white noise field...")
        white_noise(self.c_fftgrid, seed)
        if verbose:
            self.log.info("done!")

        if verbose:
            self.log.info("Applying primordial power spectrum...")
        PP = self.calculate_primordial_power(self.kgrid*self.h)
        self.c_fftgrid *= np.sqrt(PP) * self.grid**2/(self.BoxSize/self.h)**1.5
        if verbose:
            self.log.info("done!")

        # Cut-off beyond Nyquist Frequency
        self.c_fftgrid[self.kgrid > self.kNyq] = 0.+0.j

        # If f_nl, transform to real-space for squaring
        if np.abs(fnl) > 0.:
            if verbose:
                self.log.info(f"Making map non-Gaussian with fnl={fnl}...")
            self.fft_c2r()
            self.r_fftgrid += 5/3 * fnl * self.r_fftgrid**2
            self.fft_r2c()
            if verbose:
                self.log.info("done!")

        if verbose:
            self.log.info("Applying linear transfer function...")
        apply_linear_power_or_transfer(self.c_fftgrid, self.kgrid, kLin, TFLin, self.BoxSize, self.grid)
        if verbose:
            self.log.info("done!")

        if k_cut_low is not None:
            if verbose:
                self.log.info(f"Cutting lower k-space at {k_cut_low}...")
            self.c_fftgrid[self.kgrid < k_cut_low] = 0.+0.j
            if verbose:
                self.log.info("done!")

        if k_cut_high is not None:
            if verbose:
                self.log.info(f"Cutting higher k-space at {k_cut_high}...")
            self.c_fftgrid[self.kgrid >= k_cut_high] = 0.+0.j
            if verbose:
                self.log.info("done!")

        self.c_delta = self.c_fftgrid.copy()
        if verbose:
            self.log.info("Fourier transforming to real space...")
        self.fft_c2r()
        self.r_delta = self.r_fftgrid.copy()
        if verbose:
            self.log.info("done!")

    def GenerateCAMBField(self, k_cut_low=None, k_cut_high=None, fnl=0., seed: Optional[int] = 0, verbose=False, debug_plots=False, interp_kind=None):
        if seed is None:
            seed = np.random.randint(0, 2**32-1)
        if verbose:
            self.log.info(f"Generating CAMB field with seed {seed}")

        if verbose:
            self.log.info("Making white noise field...")
        white_noise(self.c_fftgrid, seed)
        if verbose:
            self.log.info("done!")
        if debug_plots:
            plot_fields(np.abs(self.c_fftgrid), 'C =white_noise')

        if verbose:
            self.log.info("Applying primordial power spectrum...")
        PP = self.calculate_primordial_power(self.kgrid*self.h)
        self.c_fftgrid *= np.sqrt(PP) * self.grid**2/(self.BoxSize/self.h)**1.5
        if verbose:
            self.log.info("done!")
        if debug_plots:
            plot_fields(np.abs(self.c_fftgrid),
                        'C +primordial', norm=LogNorm())

        # Cut-off beyond Nyquist Frequency
        if verbose:
            self.log.info("Cutting at Nyquest Frequency...")
        self.c_fftgrid[self.kgrid > self.kNyq] = 0.+0.j
        if verbose:
            self.log.info("Done!")
        if debug_plots:
            plot_fields(np.abs(self.c_fftgrid), 'C -Nyquist', norm=LogNorm())

        # If f_nl, transform to real-space for squaring
        if np.abs(fnl) > 0.:
            if verbose: self.log.info(f"Making map non-Gaussian with fnl {fnl}...")
            self.fft_c2r()
            if debug_plots: plot_fields(self.r_fftgrid, 'R Pre-NG')  # type: ignore
            apply_ng(self.r_fftgrid, fnl)
            if debug_plots: plot_fields(self.r_fftgrid, 'R Post-NG')  # type: ignore
            self.fft_r2c()
            if debug_plots: plot_fields(np.abs(self.c_fftgrid), 'C +NG', norm=LogNorm())  # type: ignore
            if verbose: self.log.info("done!")

        if verbose:
            self.log.info("Applying camb transfer function...")
        if interp_kind is None:
            interp_kind = self.interp_kind
        if interp_kind == 'cubic':
            apply_camb_transfer_cubic(self.c_fftgrid, self.kgrid, self.Ls,
                                self.qs, self.transfers, self.ells, self.h)  # type: ignore
        elif interp_kind == 'linear':
            apply_camb_transfer_linear(self.c_fftgrid, self.kgrid, self.Ls,
                                    self.qs, self.transfers, self.ells, self.h)  # type: ignore
        else:
            self.log.fatal("Interpolation kind %s not supported!", interp_kind)
            sys.exit(6)
        if verbose:
            self.log.info("done!")
        if debug_plots:
            plot_fields(np.abs(self.c_fftgrid), 'C +CAMB', norm=LogNorm())

        if k_cut_low is not None:
            if verbose:
                self.log.info(f"Cutting lower k-space at {k_cut_low}...")
            self.c_fftgrid[self.kgrid < k_cut_low] = 0.+0.j
            if debug_plots:
                plot_fields(np.abs(self.c_fftgrid),
                            'C -k_cut_low', norm=LogNorm())
            if verbose:
                self.log.info("done!")

        if k_cut_high is not None:
            if verbose:
                self.log.info(f"Cutting higher k-space at {k_cut_high}...")
            self.c_fftgrid[self.kgrid >= k_cut_high] = 0.+0.j
            if debug_plots:
                plot_fields(np.abs(self.c_fftgrid),
                            'C -k_cut_high', norm=LogNorm())
            if verbose:
                self.log.info("done!")

        if debug_plots:
            plot_fields(np.abs(self.c_fftgrid), 'C Final', norm=LogNorm())
        self.c_delta = self.c_fftgrid.copy()
        if verbose:
            self.log.info("Fourier transforming to real space...")
        self.fft_c2r()
        self.r_delta = self.r_fftgrid.copy()
        if debug_plots:
            plot_fields(self.r_fftgrid, 'R Final')
        if verbose:
            self.log.info("done!")
        return self.r_delta

    def calculate_primordial_power(self, kgrid=None, cosmo=None, seed=None):
        """
        Calculate the primordial power spectrum
        """
        if kgrid is None:
            kgrid = self.kgrid
        if cosmo is None:
            cosmo = self.cosmo

        pk = np.zeros_like(kgrid)
        idx = np.where(kgrid > 0)
        pfactor = cosmo['A']*cosmo['kpivot']**(1.-cosmo['ns'])
        pk[idx] = np.power(kgrid[idx], cosmo['ns']-4.)*pfactor
        return pk


###################################
## Helper functions
###################################

@njit(parallel=True)
def _Pk_shells(r_delta_shells):
    # Helperfunction to compute powerspectrum of binned real density fields
    P_measured = np.zeros(len(r_delta_shells))
    for i in prange(len(r_delta_shells)):
        P_measured[i] = np.sum(r_delta_shells[i]**2)
    return P_measured

@njit(parallel=True)
def _Bk_shells(r_delta_shells, bin_indices):
    # Helperfunction to compute bispectrum of binned real density fields
    B_measured = np.zeros(len(bin_indices))
    for bin_i in prange(len(bin_indices)):
        bin_index = bin_indices[bin_i]
        B_measured[bin_i] = np.sum(r_delta_shells[bin_index[0]]
                                   * r_delta_shells[bin_index[1]]
                                   * r_delta_shells[bin_index[2]])
    return B_measured

@njit(parallel=True)
def fast_copy(target, source):
    # Helperfunction to copy arrays in parallel (only faster than np.copyto() in 3D)
    assert target.shape == source.shape
    for i in prange(target.shape[0]):
        for j in range(target.shape[1]):
            target[i, j] = source[i, j]

@njit(parallel=True)
def white_noise(fftgrid, seed):
    # Helperfunction to generate white noise in parallel
    if seed is not None:
        np.random.seed(seed)
    for i in prange(fftgrid.shape[0]):
        for j in range(fftgrid.shape[1]):
            fftgrid[i, j] = (np.random.normal(0, 1)+1j * np.random.normal(0, 1))/np.sqrt(2)

@njit(parallel=True)
def make_kgrid(x, y):
    # Helperfunction to generate kgrid in parallel
    kgrid = np.empty((x.size, y.size), dtype=np.float64)
    for j in prange(y.size):
        for k in range(x.size):
            kgrid[k, j] = np.sqrt(x[k]**2 + y[j]**2)
    return kgrid

@njit(parallel=True)
def apply_ng(r_grid, fnl):
    for i in prange(r_grid.shape[0]):
        r_grid[i] += 5/3 * fnl * r_grid[i]**2

@njit(parallel=True)
def apply_linear_power_or_transfer(delta_c, kgrid, kLin, PLin_or_TFLin, BoxSize, grid):
    # Helperfunction to apply interpolated powerspectrum or transferfunction in parallel
    for i in prange(kgrid.shape[0]):
        delta_c[i] *= np.interp(kgrid[i], kLin, PLin_or_TFLin)
    delta_c[0, 0] = 0

# This function cannot be parallelized with njit, because it uses scipy.interpolate.interp1d
def apply_camb_transfer_cubic(delta_c, kgrid, ls, qs, transfers, ells, h):
    for i in prange(kgrid.shape[0]):
        ks = kgrid[i] * h
        idx = find_closest_index(ks, qs)
        mask = (idx > 0) & (ells[i] >= 2)

        # NOTE: This does not modify masked values at all, should we set to 0?
        delta_c[i][mask] *= [transfers[k](ell) for k, ell in zip(idx[mask], ells[i][mask])]
    delta_c[0, 0] = 0

@njit(parallel=True)
def apply_camb_transfer_linear(delta_c, kgrid, ls, qs, transfers, ells, h):
    for i in prange(kgrid.shape[0]):
        ks = kgrid[i] * h
        idx = find_closest_index(ks, qs)
        mask = (idx > 0) & (ells[i] >= 2)
        delta_c[i][mask] *= transfers[i][mask]
    delta_c[0, 0] = 0

@njit(parallel=True)
def find_closest_index(k, qs):
    results = np.empty(k.shape, dtype=np.int32)
    for i, k_value in np.ndenumerate(k):
        results[i] = np.argmin(np.abs(qs - k_value))
    return results.flatten()

def plot_fields(field, title=None, cmap='viridis', norm: Optional[Normalize] = None):
    plt.figure()  # ensures new figure (prevents issues with interactive jupter notebook plots)
    plt.imshow(field, cmap=cmap, norm=norm)  # type: ignore
    plt.colorbar()
    if title:
        plt.title(title)
    plt.show()