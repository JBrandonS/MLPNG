from ctypes.wintypes import LGRPID
from re import M
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import camb
from scipy.interpolate import interp1d
import logging
import sys
from matplotlib.colors import LogNorm, NoNorm, Normalize

from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import fftconvolve
import matplotlib.cm as cm

from numba import prange

class CMBField:
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
        'lmax': 5000,
        'accuracy_boost': 4,
    }
    
    def __init__(self, 
                 box_size,
                 grid,  
                 camb_params=None, # type: ignore
                 cosmo_params=default_cosmo_params, 
                 run_camb=True, 
                 seed=None, 
                 logger=None, 
                 log_level=logging.INFO): #logging.WARNING):
        if logger is not None:
            self.log = logger 
        else: 
            logging.basicConfig(format='[ %(name)s - %(funcName)20s() ] | %(levelname)s : %(message)s', level=log_level, stream=sys.stdout)
            self.log = logging.getLogger(__name__)
            
        self.log.debug('Initializing CMBField...')
        assert grid%2 == 0, self.log.fatal("choose an even resolution size. Got: {}".format(grid))
        
        #Number of grid-points per dimension
        self.grid = grid
        self.cosmo = cosmo_params
        self.h = self.cosmo['h']
    
        self.seed = np.random.randint(0,2**32-1) if seed is None else seed
        self.log.debug('Master seed set to {}'.format(self.seed))
        
        #Size of periodic box e.g. in Mpc/h
        self.box_size = box_size 
        #Physical size of grid-cell
        self.cell_size = self.box_size / self.grid 
        #Fundamental mode of the box
        self.kF = 2*np.pi / self.box_size
        #Nyquist frequency of the grid
        self.kNyq = self.kF * self.grid / 2
        
        #Setup mesh and k-space grid
        self.log.debug('resolution: {}, box_size: {}, cell_size: {}'.format(self.grid, self.box_size, self.cell_size))
        self.kx = 2 * np.pi * np.fft.fftfreq( self.grid, self.cell_size)
        self.ky = 2 * np.pi * np.fft.rfftfreq(self.grid, self.cell_size) # Note rfft.
        self.kmesh = np.meshgrid(self.kx,self.ky,indexing="ij")
        self.kgrid = np.sqrt(self.kmesh[0]**2 + self.kmesh[1]**2)
        self.khgrid = self.kgrid * self.h

        self.camb_params = camb_params if camb_params is not None else camb.CAMBparams()
        if camb_params is None and run_camb:
                # Set cosmological parameters
                self.log.info('Setting CAMB parameters based on provided cosmology:\n{}'.format(self.cosmo))
                self.camb_params.set_cosmology(H0=self.cosmo['h']*100, ombh2=self.cosmo['ombh2'], omch2=self.cosmo['omch2'], tau=self.cosmo['tau'])
                self.camb_params.InitPower.set_params(As=self.cosmo['As'], ns=self.cosmo['ns'], r=self.cosmo['r'], pivot_scalar=self.cosmo['kpivot']) # type: ignore
                self.camb_params.set_for_lmax(self.cosmo['lmax'], lens_potential_accuracy=self.cosmo['accuracy_boost'])
                self.camb_params.set_accuracy(AccuracyBoost=self.cosmo['accuracy_boost'])
                self.camb_params.Want_CMB = True
                self.camb_params.WantTransfer = True
            
        if run_camb: 
            self.init_camb()
        else: 
            self.log.warning('Not running CAMB. No results will be available.\nYou will need to set values [results, d_A, Ls, qs, transfer_func] or run init_camb manually.')
            
        self.log.info('Finished generating CMBField with resolution {}, box size {} Mpc/h, cell size {}'.format(self.grid, self.box_size, self.cell_size)) 


    def init_camb(self, params=None):
        if params is None: 
            self.log.debug('Running CAMB...')
            params = self.camb_params
        else: 
            self.log.debug('Running CAMB with provided parameters {}...'.format(params))
        
        self.results = camb.get_results(params)
        self.d_A = self.results.angular_diameter_distance(self.cosmo['z_recomb']) * (1 + self.cosmo['z_recomb'])
        
        self.Ls, self.qs, self.transfer_func = self.results.get_cmb_transfer_data().get_transfer()
        self.transfer_interp = [interp1d(self.Ls, self.transfer_func[:, q], kind='cubic') for q in range(len(self.qs))]
        
        self.ellgrid = self.khgrid * self.d_A
        self.log.debug(f'ell gird: {self.ellgrid.shape} {self.ellgrid}')
        
        self.theta = (self.box_size*self.h) / self.d_A * 60 * 180 / np.pi #Field of view in arcminutes
        self.pix_size = self.theta / self.grid #Size of pixel in arcminutes
        self.pix_size_rad = self.pix_size / 60 * np.pi / 180 #Size of pixel in radians
        
        self.log.info('d_A: {}, angular size: {}'.format(self.d_A, self.theta))
        self.log.debug('Finished running CAMB.')
        

    def GenerateField(self, f_nl=0., seed=None, debug_plots=False):
        seed = self.seed if seed is None else seed
        
        # Start with gaussian white noise
        density_field = self.calc_white_noise(seed=seed)
        if debug_plots: plot_fields(np.abs(density_field), "FFT Field (white noise)")
        
        # Multiply by sqrt(Primordial Power Spectrum) to apply initial conditions
        ppk = self.calc_primordial_power()
        density_field *=  np.sqrt( ppk ) * self.grid**2/(self.box_size/self.h)**1.5
        if debug_plots: plot_fields(np.abs(density_field), "FFT Field (+ Primordial Power Spectrum)", norm=LogNorm())
        
        # Cut-off beyond Nyquist Frequency
        density_field[self.kgrid > self.kNyq] = 0.+0.j

        # Move to real space to add NG terms
        real_field = np.fft.irfft2(density_field).real
        real_field += self.calc_non_gaussian(f_nl, real_field)
        density_field = np.fft.rfft2(real_field)
        if debug_plots: plot_fields(np.abs(density_field), "FFT Field (+ Non-Gaussianity)", norm=LogNorm())

        self.transfers = self.calc_transfer_field()
        density_field *= self.transfers #* self.grid**2/(self.box_size/self.h)**1.5
        if debug_plots: plot_fields(np.abs(density_field), "FFT Field (+ transfer function)", norm=LogNorm())

        # Go into real space to return the real density field
        self.c_density_field = density_field.copy()
        final_field = np.fft.irfft2(density_field).real
        if debug_plots: plot_fields(final_field, "Complete Density Field")
        
        self.log.debug('Finished generating density field with seed {}'.format(seed))
        self.r_density_field = final_field.copy()
        return  self.r_density_field


    def calc_white_noise(self, kgrid=None, loc=0, scale=1, seed=None):
        """
        Calculate white noise for a given 2D grid using Gaussian random variables with specified mean and standard deviation.
        """
        if seed is None: seed = self.seed 
        if kgrid is None: kgrid = self.kgrid
        np.random.seed(seed)
        
        self.log.debug('Calculating white noise with seed {}, loc {}, scale {}'.format(seed, loc, scale))
        white_noise = np.zeros(kgrid.shape, dtype=complex)
        for i in range(kgrid.shape[0]):
            for j in range(kgrid.shape[1]):
                white_noise[i,j] = (np.random.normal(loc,scale)+1j*np.random.normal(loc,scale))/np.sqrt(2)
        return white_noise
    
                
    def calc_primordial_power(self, kgrid=None, cosmo=None, seed=None):
        """
        Calculate the primordial power spectrum
        """
        if kgrid is None: kgrid = self.khgrid
        if cosmo is None: cosmo = self.cosmo
        
        pk = np.zeros_like(kgrid)
        idx = np.where(kgrid > 0)
        
        pfactor = cosmo['A']*cosmo['kpivot']**(1.-cosmo['ns'])
        pk[idx] = np.power(kgrid[idx], cosmo['ns']-4.)*pfactor
        pk[0,0] = 0
        return pk


    def calc_non_gaussian(self, F_nl, field):
        return 5/3 * F_nl * field**2
    
    def calc_transfer_field(self, kgrid=None, ellgrid=None):
        if kgrid is None: kgrid = self.kgrid
        if ellgrid is None: ellgrid = self.ellgrid    
        ## Indexing should be fast since numpy can be optimized, but a bit hard to read
        # Calculate the indices for each element in the kgrid array
        indices = np.array([self.find_closest_index(k, self.qs) for k in kgrid.ravel()]).reshape(kgrid.shape)
        # print(indices.shape, indices)
        
        # Create a mask for the conditions (0 < k <= kNyq or self.ells < 2)
        mask = (indices > 0) & (ellgrid >= 2) & (kgrid <= self.kNyq) & (ellgrid < 5000)
        self.log.debug('Number of elements in mask: {}'.format(np.sum(mask)))
        
        # Apply the transfer function using the mask and indices
        transfers = np.zeros(kgrid.shape)
        # transfers[mask] = np.array([self.transfer_interp[k](ell) for k, ell in zip(indices[mask], ellgrid[mask])])
        
        self.log.info('Calculating transfer function... that might take a while...')
        for x in range(ellgrid.shape[0]):
            valid_indices = np.where(mask[x])
            transfers[x, valid_indices] += np.mean([self.transfer_interp[k](ellgrid[x, valid_indices]) for k in range(len(self.qs))], axis=0)
        self.log.info('Done!')
        return transfers

    def find_closest_index(self, k, qs):
        return np.abs(qs - k).argmin()
    
    def get_pk(self, field=None, kgrid=None, kmax=None):
        field = self.c_density_field if field is None else field
        kgrid = self.khgrid if kgrid is None else kgrid
        kmax = self.kNyq if kmax is None else kmax

        kmax_n = np.int64(np.ceil(kmax / self.kF))
        ks = np.zeros(kmax_n - 1)
        ns = np.zeros(kmax_n - 1)
        Pks = np.zeros(kmax_n - 1)

        k_indices = np.floor(kgrid / self.kF).astype(np.int64)
        delta2 = np.abs(field) ** 2
        valid_indices = (k_indices > 0) & (k_indices < kmax_n) & (kgrid < kmax)
        k_indices -= 1

        kxi_indices, kyi_indices = np.indices(kgrid.shape)
        for k_index in range(kmax_n - 1):
            indices = valid_indices & (k_indices == k_index)
            kxi_filtered, kyi_filtered = kxi_indices[indices], kyi_indices[indices]
            Pks[k_index] = np.sum(delta2[kxi_filtered, kyi_filtered])
            ks[k_index] = np.sum(self.kgrid[kxi_filtered, kyi_filtered])
            ns[k_index] = np.sum(indices)

        ks /= ns
        Pks *= 1 / ns * self.box_size ** 3 / self.grid ** 4
        return ks, Pks, ns
    
    def get_cmap(self, norm=True):
        val = self.c_density_field 
        if norm:
            val -= val.mean() 
            val /= self.c_density_field.std()
        return val

    def get_rmap(self, norm=True):
        val = self.r_density_field 
        if norm: 
            val -= val.mean()
            val /= self.r_density_field.std()
        return val

    def _fillgrid(self, c_fftgrid, d_type=complex):
        if c_fftgrid.shape[0] == c_fftgrid.shape[1]:
            self.log.debug('Grid is already full.')
            return c_fftgrid
        
        full_c_fftgrid = np.empty((self.grid, self.grid), dtype=d_type)
        full_c_fftgrid[:, :self.grid // 2 + 1] = c_fftgrid
        for i in range(self.grid):
            for j in range(1, self.grid // 2):
                full_c_fftgrid[i, self.grid - j] = np.conj(full_c_fftgrid[i, j])
        return full_c_fftgrid

    _get_camb_cmb_pk_value = None
    def get_camb_cmb_pk(self, theory_spectra='unlensed_scalar', force=False):
        if self._get_camb_cmb_pk_value is None or force:
            self._get_camb_cmb_pk_value = self.results.get_cmb_power_spectra(CMB_unit='muK', spectra=[theory_spectra])[theory_spectra]
        return self._get_camb_cmb_pk_value

    def cosine_window(self, N):
        "makes a cosine window for apodizing to avoid edges effects in the 2d FFT" 
        # make a 2d coordinate system
        N=int(N) 
        ones = np.ones(N)
        inds  = (np.arange(N)+.5 - N/2.)/N * np.pi ## eg runs from -pi/2 to pi/2
        X = np.outer(ones,inds)
        Y = np.transpose(X)
    
        # make a window map
        window_map = np.cos(X) * np.cos(Y)
        return(window_map)

    def calculate_2d_spectrum(self,Map,delta_ell,ell_max, kgrid=None):
        "calcualtes the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"
        grid=self.grid
        
        # make a 2d ell coordinate system
        if kgrid is None:
            ones = np.ones(grid)
            inds  = (np.arange(grid)+.5 - grid/2.) /(grid-1.)
            kX = np.outer(ones,inds) / self.pix_size_rad
            kY = np.transpose(kX)
            K = np.sqrt(kX**2. + kY**2.)
            ell_scale_factor = 2*np.pi
        else:
            K = kgrid
            ell_scale_factor = self.d_A
            
        ell2d = K * ell_scale_factor
        self.log.info(f'ell2d min: {ell2d.min()}, max: {ell2d.max()}')

        # make an array to hold the power spectrum results
        N_bins = int(ell_max/delta_ell)
        ell_array = np.arange(N_bins)
        CL_array = np.zeros(N_bins)
        
        # get the 2d fourier transform of the map
        FMap1 = np.fft.ifft2(np.fft.fftshift(Map))
        FMap2 = np.fft.ifft2(np.fft.fftshift(Map))
        PSMap = np.fft.fftshift(np.real(np.conj(FMap1) * FMap2))
        
        # fill out the spectra
        i = 0
        while (i < N_bins):
            ell_array[i] = (i + 0.5) * delta_ell
            inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
            if np.sum(inds_in_bin) != 0:
                CL_array[i] = np.mean(PSMap[inds_in_bin])
            # print(i, ell_array[i], inds_in_bin, CL_array[i])
            i = i + 1
    
        # return the power spectrum and ell bins
        return(ell_array, CL_array*np.sqrt(self.pix_size_rad)*2.)
    
    def calculate_cls(self, map=None, lmin=2, lmax=5000, raw_cls=False, nbins=100):
        self.log.debug('Calculating C_ls...')
        if map is None:
            map = self.get_rmap()
        
        lgrid = self._fillgrid(self.ellgrid, d_type=float)
        lmin = np.max([lmin, np.min(lgrid), self.kF * self.h * self.d_A])
        lmax = np.min([lmax, np.max(lgrid), self.kNyq * self.h * self.d_A])
        self.log.info(f'lmin: {lmin}, lmax: {lmax}')
        
        bins = np.linspace(lmin, lmax, nbins+1)
        bin_indices = np.digitize(lgrid, bins)
        
        FMap = np.fft.ifft2(np.fft.fftshift(map))
        PSMap = np.real(np.conj(FMap) * FMap)
        
        cls = np.zeros(nbins)
        counts = np.zeros(nbins)
        ls = np.zeros(nbins)
        for i in range(nbins):
            idx = (bin_indices == i + 1)
            counts[i] = np.sum(idx)
            if counts[i] != 0: 
                cls[i] = np.mean(PSMap[idx])
                ls[i] = np.mean(lgrid[idx])
                
        self.log.debug(f'cls {cls.shape} ls {ls}')
        
        cls *= 2*np.sqrt(self.pix_size_rad)
        if not raw_cls:
            cls *= ls * (ls + 1) / (2 * np.pi**2)
            
        self.log.debug('Done calculating C_ls.')
        return ls, cls, counts

    def Plot_CMB_Map(self, Map_to_Plot,c_min,c_max,X_width,Y_width):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        print("map mean:",np.mean(Map_to_Plot),"map rms:",np.std(Map_to_Plot))
        plt.gcf().set_size_inches(10, 10)
        im = plt.imshow(Map_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r)
        # im.set_clim(c_min,c_max)
        ax=plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(im, cax=cax)
        # im.set_extent([0,X_width,0,Y_width])
        plt.ylabel('angle $[^\circ]$')
        plt.xlabel('angle $[^\circ]$')
        cbar.set_label('temperature [uK]', rotation=270)
        plt.show()
    
    def plot_cls(self, kgrid=None, title=None, plot_theory=True, theory_spectra='unlensed_scalar'):               
        if plot_theory:
            theory = self.get_camb_cmb_pk(theory_spectra)[2:5000]
            plt.loglog(np.arange(len(theory)), theory[:, 0], label='Theory')

        # map = self.get_rmap()
        # window = (self.cosine_window(self.grid))
        # map *= window
        
        # ls, cls = self.calculate_2d_spectrum(map, 1, 5000, kgrid)
        # # cls *= ls * (ls + 1) / (2 * np.pi**2)
        # plt.loglog(ls, cls*10**6, label='2D Spectrum')
            
        ls, dls, counts = self.calculate_cls()
        m = dls > 0
        plt.loglog(ls[m], dls[m]*10**6, label=r'$D_{\ell}$')
        
        if title is not None: 
            plt.title(title)
        plt.xlabel('$\ell$')
        plt.ylabel('$D_\ell$')
        plt.legend()
        plt.show()
        


def plot_fields(field, title=None, cmap='viridis', norm=None):
    plt.figure() # ensures new figure
    plt.imshow(field, cmap=cmap, norm=norm) # type: ignore
    plt.colorbar()
    if title: plt.title(title)
    plt.show()

# @njit(parallel=True)
def apply_linear_power_or_transfer(delta_c,kgrid,kLin,PLin_or_TFLin,BoxSize,grid):
    # Helperfunction to apply interpolated powerspectrum or transferfunction in parallel
    for i in prange(kgrid.shape[0]):
        delta_c[i] *= np.interp(kgrid[i],kLin,PLin_or_TFLin)
    delta_c[0,0] = 0