import logging
import sys
import os
from re import M

import camb
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, NoNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


class CMBMap:
    # General \Lambda CDM parameters
    default_cosmo_params = {
        'h': 0.6711,
        'r': 0,
        'As': 2.13e-09,
        'ns': 0.9624,
        'kpivot': 0.05,
        'z_recomb': 1090.48,
        'ombh2': 0.02233,
        'omch2': 0.1198,
        'tau': 0.0561,
        'lmax': 2500,
        'accuracy_boost': 1,
        't_cmb': 2.7255,
    }
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-picklable attributes
        del state['camb_params']
        del state['camb_results']
        del state['transfer_interp']
        del state['transfer_func']
        del state['log']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        
        # Recreate non-picklable attributes
        self.init_camb()

        
        # self.run_camb()
            
    def __reduce__(self):
        # Define a tuple of picklable attributes
        picklable_state = (self.box_size, self.grid, None, self.cosmo, False, self.transfers, logging.INFO)

        # Return a tuple with the class as the callable and the picklable attributes as its arguments
        return (CMBMap, picklable_state)
    
    def __init__(self, 
                 box_size,
                 grid,  
                 camb_params=None, # type: ignore
                 cosmo=default_cosmo_params, 
                 run_camb=True, 
                 transfers=None,
                 log_level=logging.INFO): #logging.WARNING):
        logging.basicConfig(format='[ %(name)s - %(funcName)20s() ] | %(levelname)s : %(message)s', level=log_level, stream=sys.stdout)
        self.log = logging.getLogger(__name__)
        assert grid%2 == 0, self.log.fatal("choose an even grid size. Got: {}".format(grid))
                
        #Number of grid-points per dimension
        self.grid = grid
        self.cosmo = cosmo
        self.h = cosmo['h']
        
        self.box_size = box_size            # Size of periodic box e.g. in Mpc/h
        self.cell_size = box_size / grid    # Physical size of grid-cell
        self.kF = 2*np.pi / box_size        # Fundamental mode of the box
        self.kNyq = self.kF * grid / 2      # Nyquist frequency of the grid

        self.rshape = np.array([self.grid, self.grid])
        self.cshape = np.array([self.grid, self.grid//2 + 1])
        
        #Setup mesh and k-space grid
        self.log.debug('grid size: {}, box_size: {}, cell_size: {}'.format(self.grid, self.box_size, self.cell_size))
        kx = 2 * np.pi * np.fft.fftfreq( grid, self.cell_size)
        ky = 2 * np.pi * np.fft.rfftfreq(grid, self.cell_size) # Note rfft.
        self.kmesh = np.meshgrid(kx, ky, indexing="ij")
        self.kgrid = np.sqrt(self.kmesh[0]**2 + self.kmesh[1]**2)
        self.khgrid = self.kgrid * self.h

        self.camb_params = camb_params
        self.transfers = transfers
        if run_camb: 
            self.run_camb(self.cosmo, camb_params)

    def init_camb(self, cosmo=None):
        if cosmo is None:
            cosmo = self.cosmo
            
        params = camb.CAMBparams()
        self.log.info('Setting CAMB parameters based on cosmology:\n{}'.format(cosmo))
        
        params.set_cosmology(H0=cosmo['h']*100, ombh2=cosmo['ombh2'], omch2=cosmo['omch2'], tau=cosmo['tau'])
        params.InitPower.set_params(As=cosmo['As'], ns=cosmo['ns'], r=cosmo['r'], pivot_scalar=cosmo['kpivot']) # type: ignore
        params.set_for_lmax(cosmo['lmax'], lens_potential_accuracy=cosmo['accuracy_boost'])
        params.set_accuracy(AccuracyBoost=cosmo['accuracy_boost'])
        params.Want_CMB = True
        params.WantTransfer = True
        
        self.camb_params = params
        self.log.debug(params)
        return params

    def run_camb(self, cosmo=None, params=None):
        if cosmo is None:
            cosmo = self.cosmo
            
        if params is None: 
            if self.camb_params is not None:
                params = self.camb_params
            else:
                params = self.init_camb(cosmo)
        
        self.camb_results = camb.get_results(params)
        self.d_A = self.camb_results.angular_diameter_distance(cosmo['z_recomb']) * (1 + cosmo['z_recomb'])#* 1000
        
        self.Ls, self.qs, self.transfer_func = self.camb_results.get_cmb_transfer_data().get_transfer()
        self.transfer_interp = [interp1d(self.Ls, self.transfer_func[:, q], kind='cubic') for q in range(len(self.qs))]

        self.ellgrid = self.kgrid * self.d_A
        self.ellhgrid = self.khgrid * self.d_A

        self.pix_size = self.cell_size / self.d_A * self.h / self.d_A * 60. * 180. / np.pi # in arcmins
        
        self.log.debug('Finished setting up CAMB.')

    def GenerateField(self, f_nl=1., k_cut_low=None, k_cut_high=None, seed=0):       
        # Start with gaussian white noise
        field = self.calc_white_noise(self.kgrid.shape, seed=seed, dbg=False) 
        
        field *=  self.calc_primordial_power(self.khgrid, self.cosmo)
        field *= self.grid**2/(self.box_size/self.h)**1.5
        
        # Cut-off beyond Nyquist Frequency
        field[self.kgrid > self.kNyq] = 0.+0.j

        # Move to real space to add NG terms
        if f_nl > 0:
            real_field = np.fft.irfft2(field)
            real_field += self.calc_non_gaussian(f_nl, real_field)
            field = np.fft.rfft2(real_field)

        if self.transfers is None:
            self.transfers = self.calc_transfer_field()
        field *= self.transfers

        if k_cut_low is not None:
            field[self.kgrid < k_cut_low] = 0.+0.j
            self.log.debug("done!")

        if k_cut_high is not None:
            field[self.kgrid >= k_cut_high] = 0.+0.j

        final_field = np.fft.irfft2(field)
        
        self.c_field = field.copy()
        self.r_field = final_field.copy()
        return  final_field
    
    # @njit(parallel=True)
    def calc_white_noise(self, shape, loc=0, scale=1, seed=0, dbg=False):
        """
        Calculate white noise for a given 2D grid using Gaussian random variables with specified mean and standard deviation.
        """
        if dbg: 
            return np.ones(shape, dtype=np.complex128)
        np.random.seed(seed)
        real_part = np.random.normal(loc, scale, shape)
        imag_part = np.random.normal(loc, scale, shape)       
        return (real_part + 1j * imag_part) / np.sqrt(2)

    # @njit(parallel=True)
    def calc_primordial_power(self, khgrid, cosmo):
        """
        Calculate the primordial power spectrum
        """
        pk = np.zeros_like(khgrid)
        pfactor = 2 * np.pi**2 * cosmo['As']*cosmo['kpivot']**(1.-cosmo['ns'])
        for i in range(khgrid.shape[0]):
            mask = ( khgrid[i] > 0 )
            pk[i][mask] = np.power(khgrid[i][mask], cosmo['ns']-4.)*pfactor
        pk[0,0] = 0
        return pk
    
    def calc_non_gaussian(self, f_nl, field):
        return 5/3 * f_nl * field**2
    
    def calc_transfer_field(self, lmin=2, lmax=2500):
        khgrid = self.kgrid
        ellhgrid = self.ellgrid 

        # We only want to use the transfer function for modes that are within the range of the transfer function
        mask = (ellhgrid >= lmin) & (ellhgrid <= lmax) & (khgrid >= self.kF) & (khgrid < self.kNyq)

        k_idx = find_closest_index(khgrid[mask], self.qs)
        ell_vals = ellhgrid[mask]

        # Note the use of ones and not zeros, T(k -> 0) -> 1
        transfers = np.ones(khgrid.shape)
        transfers[mask] = np.array([self.transfer_interp[k](ell) for k, ell in zip(k_idx, ell_vals)])

        # Set transfer function to zero for modes that are outside the range of the transfer function
        # T(k -> inf) -> 0
        m2 = (ellhgrid > lmax) & (khgrid >= self.kNyq)
        transfers[m2] = 0
        
        # implot(transfers, title='Transfer function')
        return transfers

    def get_map(self, zero_mean=True):
        val = self.r_field
        val = (val - val.mean()) #/ val.std()
        return val

    def calculate_cls(self, map=None, lmin=2, lmax=2500, raw_cls=False, nbins=249):
        self.log.debug('Calculating C_ls...')
        if map is None:
            map = self.get_map()
        
        ellhgrid = self.ellhgrid

        FMap = np.fft.rfft2(map)
        PSMap = np.real(np.conj(FMap)*FMap)
        self.log.info("PSMap shape: %s, PSMap mean: %s, PSMap rms: %s min: %s max: %s", PSMap.shape, np.mean(PSMap), np.std(PSMap), np.min(PSMap), np.max(PSMap))
        
        cls = np.zeros(nbins)
        counts = np.zeros(nbins)
        ls = np.zeros(nbins)
        
        bins = np.linspace(lmin, lmax, nbins + 1)
        for i in range(nbins):
            idx = (ellhgrid >= bins[i]) & (ellhgrid < bins[i+1])           
            counts[i] = np.sum(idx)
            if counts[i] > 0:       
                ls[i] = np.mean(ellhgrid[idx])      
                cls[i] += np.mean( PSMap[idx] ) #/ ( 2 * ls[i] + 1 )
                if not raw_cls:
                    cls[i] *= ls[i] * (ls[i] + 1) / (2 * np.pi)
        
        cmb_unit = self.cosmo['t_cmb'] * 10**6 # convert to muK^2, might want to multiply by t_cmb anyways
        cls *= cmb_unit**2

        cls *= np.sqrt(self.pix_size /60.* np.pi/180.)*2.
            
        self.log.debug('Done calculating C_ls.')
        return ls[counts > 0], cls[counts > 0]
        
    def plot_cls(self, title=None, lmin=2, lmax=2500, plot_theory=True, theory_spectra='unlensed_scalar'):               
        if plot_theory:
            theory = self.camb_results.get_cmb_power_spectra(CMB_unit='muK', spectra=[theory_spectra])[theory_spectra][lmin:lmax]
            plt.loglog(np.arange(lmin, lmax), theory[:, 0], label='Theory (CAMB)')
            
        ls, cls = self.calculate_cls(lmin=lmin, lmax=lmax)
            
        mask = (ls >= lmin) & (ls <= lmax) & (cls > 0)
        plt.loglog(ls[mask], cls[mask], label=r'$C_{\ell}$')
        
        if title is not None:
            plt.title(title)
        plt.xlabel("$\ell$")    # type: ignore
        plt.ylabel("$C_\ell$")  # type: ignore
        plt.legend()
        plt.show()
    
    def plot_cmb(self, c_min=-400.,c_max=400., X_width=10., Y_width=10.):
        rmap = self.get_map()
        self.log.info(f"map mean: {np.mean(rmap)} map rms: {np.std(rmap)}")
        
        plt.gcf().set_size_inches(10, 10)
        im = plt.imshow(rmap, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r) # type: ignore
        # im.set_clim(c_min,c_max)
        
        ax=plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(im, cax=cax)
        im.set_extent([0,X_width,0,Y_width])
        plt.ylabel('angle $[^\circ]$') # type: ignore
        plt.xlabel('angle $[^\circ]$') # type: ignore
        cbar.set_label('temperature [uK]', rotation=270)
        plt.show()

    def mask_c2r(self, delta_c, k_low, k_high):
        self.c_field[self.kgrid >= k_high] = 0.+0.j
        self.c_field[self.kgrid < k_low] = 0.+0.j
        self.r_field = np.fft.irfft2(self.c_field)
        return self.r_field
    
    def Pk(self, kmax=None):
            if kmax is None:
                kmax = self.kNyq

            kmax_n = np.int64((np.ceil(kmax/self.kF)))
            ks = np.zeros(kmax_n-1)
            ns = np.zeros(kmax_n-1)
            Pks = np.zeros(kmax_n-1)
            map = self.c_field

            for kxi in range(self.cshape[0]):
                for kyi in range(self.cshape[1]):
                    if kxi == 0 and kyi == 0:
                        continue

                    k = self.kgrid[kxi, kyi]
                    if k >= kmax:
                        continue
                    k_index = np.int64((np.floor(k/self.kF)))

                    delta_r = map[kxi, kyi].real
                    delta_i = map[kxi, kyi].imag
                    delta2 = delta_r**2 + delta_i**2

                    Pks[k_index-1] += delta2
                    ks[k_index-1] += k
                    ns[k_index-1] += 1.

            ks /= ns
            Pks *= 1/ns * self.box_size**3 / self.grid**4
            return ks, Pks, ns
        
    def _Bk_counts(self, fc, dk, NBmax, triangle_type, data_dir='data/static'):
        file_name = f"{data_dir}/FFTest2D_BkCounts_LBox{self.box_size}_Grid{self.grid}_Binning{dk}kF_fc{fc}_NBins{NBmax}_TriangleType{triangle_type}.npy"
        if os.path.exists(file_name):
            self.log.debug(f"Loading Counts from {file_name}")
            counts = np.load(file_name, allow_pickle=True).item()
            self.log.debug(f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type})")
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
            counts['bin_centers'] = np.array([(i, i, i) for i in fc+np.arange(0, NBmax)*dk])
        self.log.debug(f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type})")

        self.log.debug(f"Creating Grids for Counts...")
        c_ones = np.ones_like(self.c_field)
        r_ones_shells = np.zeros((NBmax, self.grid, self.grid), dtype=np.float64)
        for i in tqdm(range(NBmax), disable=True):
            k_low = self.kF * (fc + dk * i - dk/2)
            k_high = self.kF * (fc + dk * i + dk/2)
            r_ones_shells[i] = self.mask_c2r(c_ones, k_low, k_high)

        self.log.debug("Computing Powerspectrum Counts...")
        counts['counts_P'] = _Pk_shells(r_ones_shells) * self.grid**2
        self.log.debug("done!")

        self.log.debug("Computing Triangle Counts...")
        bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)
        counts['counts_B'] = _Bk_shells( r_ones_shells, bin_indices) * self.grid**4
        self.log.debug("done!")

        np.save(file_name, counts)  # type: ignore
        self.log.debug(f"Saved Triangle Counts to {file_name}")
        return counts
        
    def Bk(self, fc, dk, NBmax, triangle_type='All'):
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

        counts = self._Bk_counts(fc, dk, NBmax, triangle_type)

        r_delta_shells = np.zeros((NBmax, self.grid, self.grid), dtype=np.float64)

        self.log.debug("Creating Grids for Measurements...")
        for i in tqdm(range(NBmax), disable=True):
            k_low = self.kF * (fc + dk * i - dk/2)
            k_high = self.kF * (fc + dk * i + dk/2)
            r_delta_shells[i] = self.mask_c2r(self.c_field, k_low, k_high)

        self.log.debug("Computing Powerspectrum...")
        P = _Pk_shells(r_delta_shells) * self.box_size**3 / counts['counts_P'] / self.grid**2
        self.log.debug("done!")

        self.log.debug("Computing Bispectrum...")
        bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)
        B = _Bk_shells(r_delta_shells, bin_indices) * self.box_size**6 / self.grid**2
        self.log.debug("done! \n")

        result = np.ones((len(counts['bin_centers']), 8))
        result[:, :3] = counts['bin_centers']
        result[:, 3:6] = P[bin_indices]
        result[:, 6] = B/counts['counts_B']
        result[:, 7] = counts['counts_B']

        return result

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
     
def find_closest_index(index, grid_vals):
    return np.array([np.abs(grid_vals - k_val).argmin() for k_val in index.flatten()]).reshape(index.shape)

def implot(field, title=None, cmap='viridis', norm=None):
    plt.figure() # ensures new figure
    plt.imshow(field, cmap=cmap, norm=norm) # type: ignore
    plt.colorbar()
    if title: plt.title(title)
    plt.show()
    