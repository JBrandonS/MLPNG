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

from numba import prange, njit

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
        
        #Size of periodic box e.g. in Mpc/h
        self.box_size = box_size 
        #Physical size of grid-cell
        self.cell_size = self.box_size / self.grid 
        #Fundamental mode of the box
        self.kF = 2*np.pi / self.box_size
        #Nyquist frequency of the grid
        self.kNyq = self.kF * self.grid / 2

        self.pix_size = self.cell_size * 60. * 180. / np.pi # in arcmins
        
        #Setup mesh and k-space grid
        self.log.debug('resolution: {}, box_size: {}, cell_size: {}'.format(self.grid, self.box_size, self.cell_size))
        self.kx = 2 * np.pi * np.fft.fftfreq( self.grid, self.cell_size)
        self.ky = 2 * np.pi * np.fft.rfftfreq(self.grid, self.cell_size) # Note rfft.
        self.kmesh = np.meshgrid(self.kx,self.ky,indexing="ij")
        self.kgrid = np.sqrt(self.kmesh[0]**2 + self.kmesh[1]**2)
        self.khgrid = self.kgrid * self.h

        # onesvec = np.ones(grid)
        # inds  = (np.arange(grid)+.5 - grid/2.) /(grid-1.) # create an array of size N between -0.5 and +0.5
        # # compute the outer product matrix: X[i, j] = onesvec[i] * inds[j] for i,j 
        # # in range(N), which is just N rows copies of inds - for the x dimension
        # X = np.outer(onesvec,inds) 
        # # compute the transpose for the y dimension
        # Y = np.transpose(X)
        # # radial component R
        # self.kgrid = np.sqrt(X**2. + Y**2.)
        # self.khgrid = self.kgrid * self.h

        self.camb_params = camb_params
        if run_camb: 
            self.run_camb(self.cosmo, camb_params)

    def run_camb(self, cosmo=None, params=None):
        if cosmo is None:
            cosmo = self.cosmo
            
        if params is None: 
            if self.camb_params is not None:
                params = self.camb_params
            else:
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
        
        self.camb_results = camb.get_results(params)
        self.d_A = self.camb_results.angular_diameter_distance(cosmo['z_recomb']) * (1 + cosmo['z_recomb'])
        
        self.Ls, self.qs, self.transfer_func = self.camb_results.get_cmb_transfer_data().get_transfer()
        self.transfer_interp = [interp1d(self.Ls, self.transfer_func[:, q], kind='cubic') for q in range(len(self.qs))]

        self.ellgrid = self.kgrid * self.d_A
        self.ellhgrid = self.khgrid * self.d_A

        # pix_to_rad = (self.pix_size/60. * np.pi/180.) # going from pix_size in arcmins to degrees and then degrees to radians
        # ell_scale_factor = 2. * np.pi * self.d_A / pix_to_rad  # now relating the angular size in radians to multipoles
        # self.ellgrid = self.kgrid * ell_scale_factor
        # self.ellhgrid = self.ellgrid * self.h
        
        self.log.debug('Finished running CAMB.')

    def make_CMB_T_map(self, N, pix_size):
        FT_2d = self.GenerateField() # we take the sqrt since the power spectrum is T^2
        # plt.imshow(np.abs(FT_2d))
        # plt.show()
        
        # move back from ell space to real space
        CMB_T = FT_2d #np.fft.ifft2(np.fft.fftshift(FT_2d)) 
        # move back to pixel space for the map
        CMB_T = CMB_T/(pix_size /60.* np.pi/180.)
        # we only want to plot the real component
        CMB_T = np.real(CMB_T)

        ## return the map
        return(CMB_T)

    def GenerateField(self, f_nl=1., seed=0):       
        # Start with gaussian white noise
        field = calc_white_noise(self.kgrid.shape, seed=seed)
        
        # Multiply by sqrt(Primordial Power Spectrum) to apply initial conditions
        ppk = calc_primordial_power(self.khgrid, self.cosmo)
        field *=  np.sqrt(ppk) * self.grid**2/(self.box_size/self.h)**1.5
        
        # Cut-off beyond Nyquist Frequency
        field[self.kgrid > self.kNyq] = 0.+0.j

        # Move to real space to add NG terms
        real_field = np.fft.irfft2(np.fft.fftshift(field))
        real_field += self.calc_non_gaussian(f_nl, real_field)
        field = np.fft.fftshift(np.fft.rfft2(real_field))

        field *= self.calc_transfer_field()

        # Go into real space to return the real density field
        final_field = np.fft.irfft2(field)
        self.r_field = final_field.copy()
        return  final_field

    def calc_non_gaussian(self, F_nl, field):
        return 5/3 * F_nl * field**2
    
    def calc_transfer_field(self, khgrid=None, ellhgrid=None, lmin=2, lmax=5000):
        if khgrid is None: khgrid = self.khgrid
        if ellhgrid is None: ellhgrid = self.ellhgrid 

        k_indices = np.array([find_closest_index(k, self.qs) for k in khgrid.flatten()]).reshape(khgrid.shape)
        mask = (k_indices > 0) & (ellhgrid >= lmin) & (ellhgrid < lmax) & (khgrid >= self.kF) & (khgrid < self.kNyq)
        
        k_idx = k_indices[mask]
        ell_vals = ellhgrid[mask]
        
        transfers = np.zeros(khgrid.shape)
        transfers[mask] = np.array([self.transfer_interp[k](ell) for k, ell in zip(k_idx, ell_vals)])
        implot(transfers, title='Transfer Function')

        # k_idx, ell_vals = make_monotonic(k_idx, ell_vals)

        # plt.figure()
        # fig, axs = plt.subplots(2,2, figsize=(12,8), sharex = True)
        # for ix, ax in zip([3, 20, 40, 60], axs.reshape(-1)):
        #     vals = np.array([self.transfer_interp[k](ix) for k in k_idx])
        #     ax.plot(self.qs[k_idx], vals)
        #     ax.set_title(r'$\ell = %s$'%self.Ls[ix])
        #     if ix>1: ax.set_xlabel(r'$k \rm{Mpc}$')
        # plt.show()
        
        return transfers

    def get_rmap(self, zero_mean=True):
        val = self.r_field
        if zero_mean: 
            val -= val.mean()
        return val

    def calculate_cls(self, map=None, lmin=2, lmax=5000, raw_cls=False, nbins=1000, muk_units=True):
        self.log.debug('Calculating C_ls...')
        if map is None:
            map = self.get_rmap()
        
        lhgrid = self.ellhgrid
        khgrid = self.khgrid
        
        bins = np.linspace(lmin, lmax, nbins+1)
        bin_size = (lmax - lmin) / nbins
        indices = np.array([find_closest_index(k, self.qs) for k in khgrid.flatten()]).reshape(khgrid.shape)   
        
        FMap = np.fft.rfft2(map)
        PSMap = np.real(np.conj(FMap)*FMap)
        
        cls = np.zeros(nbins)
        counts = np.zeros(nbins)
        ls = np.zeros(nbins)
        for i in range(nbins):
            idx = (lhgrid >= lmin + i * bin_size) & (lhgrid <= lmin + (i + 1) * bin_size)
            counts[i] = np.sum(idx)
            if counts[i] > 0: 
                cls[i] = np.mean(PSMap[idx])
                ls[i] = np.mean(lhgrid[idx])

        if not raw_cls:
            cls *= ls * (ls + 1) / (2 * np.pi**2)

        if muk_units:
            cmb_unit = 1e6 * 2.7
            cls *= cmb_unit**2
            
        # cls *= np.sqrt(self.pix_size /60.* np.pi/180.)*2.
        self.log.debug('Done calculating C_ls.')
        return ls, cls, counts
    
def Plot_CMB_Map(map, c_min=-400,c_max=400, X_width=10, Y_width=10):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # print("map mean:",np.mean(map),"map rms:",np.std(map))
    
    plt.gcf().set_size_inches(10, 10)
    im = plt.imshow(map, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r) # type: ignore
    # im.set_clim(c_min,c_max)
    ax=plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    im.set_extent([0,X_width,0,Y_width])
    plt.ylabel('angle $[^\circ]$')
    plt.xlabel('angle $[^\circ]$')
    cbar.set_label('temperature [uK]', rotation=270)
    plt.show()
    
def plot_cls(cmbmap, title=None, lmin=2, lmax=5000, plot_theory=True, theory_spectra='unlensed_scalar', muk_unit=False):               
    if plot_theory:
        theory = cmbmap.camb_results.get_cmb_power_spectra(CMB_unit='muK', spectra=[theory_spectra])[theory_spectra][lmin:lmax]
        plt.semilogy(np.arange(len(theory)), theory[:, 0], label='Theory (CAMB)')
        
    ls, cls, counts = cmbmap.calculate_cls()
    if muk_unit:
        cmb_unit = 1e6 * 2.7
        cls *= cmb_unit**2
        
    mask = (cls > 0) & (ls > lmin) & (ls < lmax)
    plt.semilogy(ls[mask], cls[mask], label=r'$C_{\ell}$')
    
    if title is not None: 
        plt.title(title)
    plt.xlabel("$\ell$")
    plt.ylabel("$C_\ell$")
    plt.legend()
    plt.show()

# @njit(parallel=True)
def calc_white_noise(shape=(128, 128), loc=0, scale=1, seed=0):
    """
    Calculate white noise for a given 2D grid using Gaussian random variables with specified mean and standard deviation.
    """
    np.random.seed(seed)
    white_noise = np.zeros(shape, dtype=complex)
    for i in prange(shape[0]):
        for j in range(shape[1]):
            white_noise[i,j] = (np.random.normal(loc,scale)+1j*np.random.normal(loc,scale))/np.sqrt(2)
    return white_noise

# @njit(parallel=True)
def calc_primordial_power(khgrid, cosmo):
    """
    Calculate the primordial power spectrum
    """
    pk = np.zeros_like(khgrid)
    pfactor = 2 * np.pi **2 * cosmo['As']*cosmo['kpivot']**(1.-cosmo['ns'])
    for i in prange(khgrid.shape[0]):
        mask = np.where(khgrid[i] > 0)[0]
        pk[i][mask] = np.power(khgrid[i][mask], cosmo['ns']-4.)*pfactor
    pk[0,0] = 0
    return pk

@njit(parallel=True)
def complete_grid(c_fftgrid, width, d_type=complex):
    if c_fftgrid.shape[0] == c_fftgrid.shape[1]:
        return c_fftgrid
    
    full_c_fftgrid = np.empty(width, dtype=d_type)
    full_c_fftgrid[:, :width // 2 + 1] = c_fftgrid
    for i in prange(width):
        for j in range(1, width // 2):
            full_c_fftgrid[i, width - j] = np.conj(full_c_fftgrid[i, j])
    return full_c_fftgrid

def find_closest_index(k, qs):
    return np.abs(qs - k).argmin()

def implot(field, title=None, cmap='viridis', norm=None):
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

def is_monotonic(array):
    return (np.diff(array) >= 0).all() or (np.diff(array) <= 0).all()

def make_monotonic(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Both input arrays should have the same length")

    new_arr1 = [arr1[0]]
    new_arr2 = [arr2[0]]

    for i in range(1, len(arr1)):
        if arr1[i] > new_arr1[-1]:
            new_arr1.append(arr1[i])
            new_arr2.append(arr2[i])

    return np.array(new_arr1), np.array(new_arr2)