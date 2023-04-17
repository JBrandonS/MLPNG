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

## Probably want to change np.fft to pyfftw

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
        'lmax': 2500,
        'accuracy_boost': 4,
    }
    
    def __init__(self, 
                 box_size,
                 resolution,  
                 camb_params=None, # type: ignore
                 cosmo_params=default_cosmo_params, 
                 run_camb=True, 
                 seed=None, 
                 logger=None, 
                 log_level=logging.INFO): #logging.WARNING):
        """
        Initialize a CMBField object with the given parameters and configurations.

        Parameters
        ----------
        resolution : int
            The number of grid points per dimension. Must be even.
        box_size : float
            Size of the periodic box in Mpc/h.
        F_nl : float
            Non-Gaussianity parameter.
        camb_params : camb.model.CAMBparams, optional
            CAMB parameters to be used. If None, default parameters will be set based on the provided cosmology.
        cosmo_params : dict, optional
            Cosmological parameters. Defaults to default_cosmo_params.
        run_camb : bool, optional
            If True, run CAMB to compute the CMB transfer function. Defaults to True.
        seed : int, optional
            Seed for the random number generator. If None, a random seed will be generated.
        logger : logging.Logger, optional
            A logger instance. If None, a new logger will be created.
        log_level : int, optional
            Logging level. Defaults to logging.INFO.

        Attributes
        ----------
        log : logging.Logger
            Logger instance for logging messages.
        resolution : int
            Number of grid points per dimension.
        cosmo : dict
            Dictionary containing cosmological parameters.
        F_nl : float
            Non-Gaussianity parameter.
        h : float
            Hubble constant (h).
        seed : int
            Seed for the random number generator.
        box_size : float
            Size of the periodic box in Mpc/h.
        cell_size : float
            Physical size of a grid cell.
        kF : float
            Fundamental mode of the box.
        kNyq : float
            Nyquist frequency of the grid.
        kx, ky : np.ndarray
            Arrays containing the wavenumbers in x and y directions.
        kmesh : list of np.ndarray
            Meshgrid of wavenumbers in x and y directions.
        kgrid : np.ndarray
            Grid of wavenumber magnitudes.
        kmin, kmax : float
            Minimum and maximum wavenumbers.
        camb_params : camb.model.CAMBparams
            CAMB parameters used for the calculation.
        results : camb.results.CAMBResults, optional
            The CAMB results object, only available if run_camb is True.
        d_A : float, optional
            The angular diameter distance to the recombination redshift in Mpc, only available if run_camb is True.
        Ls, qs : np.ndarray, optional
            Arrays of multipoles (ell) and wavenumbers (k) in the transfer function, only available if run_camb is True.
        transfer_func : np.ndarray, optional
            The transfer function data, only available if run_camb is True.
        transfer : list of scipy.interpolate.interpolate.interp1d, optional
            List of cubic interpolating functions for each wavenumber in qs, only available if run_camb is True.
        ells : np.ndarray, optional
            The grid of multipoles (ell) corresponding to the kgrid of the object, only available if run_camb is True.

        """
        if logger is not None:
            self.log = logger 
        else: 
            logging.basicConfig(format='[ %(name)s - %(funcName)20s() ] | %(levelname)s : %(message)s', level=log_level, stream=sys.stdout)
            self.log = logging.getLogger(__name__)
            
        self.log.debug('Initializing CMBField...')
        assert resolution%2 == 0, self.log.fatal("choose an even resolution size. Got: {}".format(resolution))
        
        #Number of grid-points per dimension
        self.resolution = resolution
        self.cosmo = cosmo_params
        self.h = self.cosmo['h']
    
        self.seed = np.random.randint(0,2**32-1) if seed is None else seed
        self.log.debug('Master seed set to {}'.format(self.seed))
        
        #Size of periodic box e.g. in Mpc/h
        self.box_size = box_size 
        #Physical size of grid-cell
        self.cell_size = self.box_size / self.resolution 
        #Fundamental mode of the box
        self.kF = 2*np.pi / self.box_size
        #Nyquist frequency of the grid
        self.kNyq = self.kF * self.resolution / 2
        self.pix_scale = self.resolution**2/(self.box_size/self.h)**1.5
        
        #Setup mesh and k-space grid
        self.log.debug('resolution: {}, box_size: {}, cell_size: {}'.format(self.resolution, self.box_size, self.cell_size))
        self.kx = 2 * np.pi * np.fft.fftfreq( self.resolution, self.cell_size)
        self.ky = 2 * np.pi * np.fft.rfftfreq(self.resolution, self.cell_size) # Note rfft.
        self.kmesh = np.meshgrid(self.kx,self.ky,indexing="ij")
        self.k_hgrid = np.sqrt(self.kmesh[0]**2 + self.kmesh[1]**2)
        self.kgrid = self.k_hgrid * self.h
        
        self.kmin = np.min(self.k_hgrid)
        self.kmax = np.max(self.k_hgrid)

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
            
        self.log.info('Finished generating CMBField with resolution {}, box size {} Mpc/h, and cell size {}'.format(self.resolution, self.box_size, self.cell_size)) 
        self.log.info('min k: {}, max k: {}. kF: {}, KNyq: {}'.format(self.kmin, self.kmax, self.kF, self.kNyq))


    def init_camb(self, params=None):
        """
        Initialize CAMB with the provided parameters and compute the CMB transfer function.

        Parameters
        ----------
        params : camb.model.CAMBparams, optional
            CAMB parameters to be used. If None, use the object's CAMB parameters.

        Attributes Set
        --------------
        results : camb.results.CAMBResults
            The CAMB results object.
        d_A : float
            The angular diameter distance to the recombination redshift in Mpc.
        Ls : np.ndarray
            The array of multipoles (ell) in the transfer function.
        qs : np.ndarray
            The array of wavenumbers (k) in the transfer function.
        transfer_func : np.ndarray
            The transfer function data.
        transfer : list of scipy.interpolate.interpolate.interp1d
            List of cubic interpolating functions for each wavenumber in qs.
        ells : np.ndarray
            The grid of multipoles (ell) corresponding to the kgrid of the object.

        """
        if params is None: 
            self.log.debug('Running CAMB...')
            params = self.camb_params
        else: self.log.debug('Running CAMB with provided parameters {}...'.format(params))
        
        self.results = camb.get_results(params)
        self.d_A = self.results.angular_diameter_distance(self.cosmo['z_recomb']) * 1000
        
        self.Ls, self.qs, self.transfer_func = self.results.get_cmb_transfer_data().get_transfer()
        self.transfer = [interp1d(self.Ls, self.transfer_func[:, i], kind='cubic') for i in range(len(self.qs))]
        self.ellgrid = np.round(self.k_hgrid * self.h * self.d_A).astype(np.int64)
        
        self.log.debug('d_A: {}'.format(self.d_A))
        self.log.debug('Finished running CAMB.')
        

    def GenerateField(self, f_nl=0., seed=None, debug_plots=False, linear=False):
        """
        Generate a density field by applying the CAMB transfer function, primordial power spectrum,
        and non-Gaussian terms. Saves output to densitfy_field attribute and returns it.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. If None, use the seed of the object.
        plot : bool, optional
            If True, create and display diagnostic plots of the density field at different stages
            of the generation process. Defaults to False.

        Returns
        -------
        final_field : np.ndarray
            The generated density field in real space.

        """
        seed = self.seed if seed is None else seed
        
        # Start with gaussian white noise
        density_field = self.calc_white_noise(seed=seed)
        if debug_plots: plot_fields(np.abs(density_field), "FFT Field (white noise)")
        
        # Multiply by sqrt(Primordial Power Spectrum) to apply initial conditions
        ppk = self.calc_primordial_power()
        density_field *=  np.sqrt( ppk ) * self.pix_scale
        if debug_plots: plot_fields(np.abs(density_field), "FFT Field (+ Primordial Power Spectrum)", norm=LogNorm())
        
        # Cut-off beyond Nyquist Frequency
        density_field[self.k_hgrid > self.kNyq] = 0.+0.j

        # Move to real space to add NG terms
        real_field = np.fft.irfft2(density_field).real
        real_field += self.calc_non_gaussian(f_nl, real_field)
        if debug_plots: plot_fields(real_field, "Real Field (+ Non-Gaussianity)")
        
        # Move back to kspace
        density_field = np.fft.rfft2(real_field)
        if debug_plots: plot_fields(np.abs(density_field), "FFT Field (+ Non-Gaussianity)", norm=LogNorm())

        self.transfers = self.calc_transfer_field()
        density_field *= self.transfers * self.pix_scale
        if debug_plots: plot_fields(np.abs(density_field), "FFT Field (+ transfer function)", norm=LogNorm())

        # Go into real space to return the real density field
        self.c_density_field = density_field.copy()
        final_field = np.fft.irfft2(density_field).real
        if debug_plots: plot_fields(np.abs(final_field), "Complete Density Field")
        
        self.log.debug('Finished generating density field with seed {}'.format(seed))
        self.r_density_field = np.abs(final_field.copy()).real
        return  self.r_density_field


    def calc_white_noise(self, kgrid=None, loc=0, scale=1, seed=None):
        """
        Calculate white noise for a given 2D grid using Gaussian random variables with specified mean and standard deviation.
        
        Parameters
        ----------
        kgrid : np.ndarray, optional
            A 2D numpy array representing the grid on which to calculate the white noise. Default is None, in which case
            the instance attribute 'kgrid' will be used.
        loc : float, optional
            The mean (location parameter) of the Gaussian random variables used to generate the white noise. Default is 0.
        scale : float, optional
            The standard deviation (scale parameter) of the Gaussian random variables used to generate the white noise.
            Default is 1.
        seed : int, optional
            A seed for the random number generator to enable reproducibility. Default is None, in which case the instance
            attribute 'seed' will be used.
        
        Returns
        -------
        white_noise : np.ndarray
            A 2D numpy array of the same shape as 'kgrid' containing the generated white noise values.
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
        if kgrid is None: kgrid = self.kgrid
        if cosmo is None: cosmo = self.cosmo
        
        pk = np.zeros_like(kgrid)
        idx = np.where(kgrid > 0)
        
        pfactor = cosmo['A']*cosmo['kpivot']**(1.-cosmo['ns'])
        pk[idx] = np.power(kgrid[idx], cosmo['ns']-4.)*pfactor
        return pk


    def calc_non_gaussian(self, F_nl, real_field):
        """
        Calculate the non-gaussianity term
        """
        return 5/3 * F_nl * real_field**2
    
    def calc_transfer_field(self, kgrid=None, ellgrid=None):
        """
        Calculate the CMB transfer function and apply it to a given density field.

        Parameters
        ----------
        kgrid : np.ndarray, optional
            Grid of wavenumbers (k) for which the transfer function will be calculated.
            If None, use the kgrid of the object.
        ells : np.ndarray, optional
            Grid of multipoles (ell) for which the transfer function will be calculated.
            If None, use the ells of the object.
        seed : int, optional
            Seed for the random number generator. Not used in the current implementation.
        cmb_units : bool, optional
            If True, multiply the final density field by a conversion factor to get
            values in CMB units. Defaults to False.
        debug_plots : bool, optional
            If True, create and display diagnostic plots of the transfer functions.
            Defaults to True.

        Returns
        -------
        final_density_field : np.ndarray
            The density field after applying the CMB transfer function.

        """
        if kgrid is None: kgrid = self.kgrid
        if ellgrid is None: ellgrid = self.ellgrid    
        ## Indexing should be fast since numpy can be optimized, but a bit hard to read
        # Calculate the indices for each element in the kgrid array
        indices = np.array([self.find_closest_index(k, self.qs) for k in kgrid.ravel()]).reshape(kgrid.shape)
        
        # Create a mask for the conditions (0 < k <= kNyq or self.ells < 2)
        mask = (indices > 0) & (ellgrid >= 2)
        self.log.debug('Number of elements in mask: {}'.format(np.sum(mask)))
        
        # Apply the transfer function using the mask and indices
        transfers = np.zeros(kgrid.shape)
        transfers[mask] = np.array([self.transfer[k](ell) for k, ell in zip(indices[mask], ellgrid[mask])])

        return transfers 


    def find_closest_index(self, k, qs):
        return np.abs(qs - k).argmin()
    
    
    def get_pk(self, field=None, kgrid=None, kmax=None):
        """
        Compute the power spectrum of a given density field on a specified k-grid up to a maximum wavenumber kmax.

        Parameters
        ----------
        field : np.ndarray, optional
            The input density field. If None, the instance's density_field attribute will be used. Default is None.
        kgrid : np.ndarray, optional
            The input k-grid. If None, the instance's kgrid attribute will be used. Default is None.
        kmax : float, optional
            The maximum wavenumber to compute the power spectrum up to. If None, the instance's kNyq attribute will be used. Default is None.

        Returns
        -------
        ks : np.ndarray
            Array of wavenumbers (k) of the computed power spectrum.
        Pks : np.ndarray
            Array of power spectrum values (P(k)) corresponding to the wavenumbers.
        ns : np.ndarray
            Array of counts of the number of modes at each wavenumber.

        Notes
        -----
        The function computes the power spectrum as P(k) = 1/n * V * <|delta(k)|^2>, where <|delta(k)|^2> is the average
        squared amplitude of the density field in Fourier space, n is the number of modes at each wavenumber, and V is the
        volume of the simulation box.
        """
        field = self.c_density_field if field is None else field
        kgrid = self.kgrid if kgrid is None else kgrid
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
            ks[k_index] = np.sum(self.k_hgrid[kxi_filtered, kyi_filtered])
            ns[k_index] = np.sum(indices)

        ks /= ns
        Pks *= 1 / ns * self.box_size ** 3 / self.resolution ** 4
        return ks, Pks, ns
    
    def get_cmap(self):
        return self.c_density_field / self.c_density_field.std()

    def _fillgrid(self, c_fftgrid, d_type=complex):
        if c_fftgrid.shape[0] == c_fftgrid.shape[1]:
            self.log.debug('Grid is already full.')
            return c_fftgrid
        
        full_c_fftgrid = np.empty((self.resolution, self.resolution), dtype=d_type)
        full_c_fftgrid[:, :self.resolution // 2 + 1] = c_fftgrid
        for i in range(self.resolution):
            for j in range(1, self.resolution // 2):
                full_c_fftgrid[i, self.resolution - j] = np.conj(full_c_fftgrid[i, j])
        return full_c_fftgrid

    _get_camb_cmb_pk_value = None
    def get_camb_cmb_pk(self, theory_spectra='unlensed_scalar', force=False):
        if self._get_camb_cmb_pk_value is None or force:
            self._get_camb_cmb_pk_value = self.results.get_cmb_power_spectra(CMB_unit='muK', spectra=[theory_spectra])[theory_spectra]
        return self._get_camb_cmb_pk_value

    def calculate_cls(self, lmin=2, lmax=2500, raw_cls=False, nbins=100):
        self.log.debug('Calculating C_ls...')
        kgrid = self._fillgrid(self.kgrid, d_type=np.float64)
        
        # lgrid = np.zeros(kgrid.shape)
        # for x in range(self.kx.shape[0]):
        #     for y in range(self.ky.shape[0]):
        #         lgrid[x,y] = (self.kx[x] * self.h * self.d_A)**2 + (self.ky[y] * self.h * self.d_A)**2 #* self.h
        # print(lgrid)
        lgrid = kgrid * self.d_A

        lmin = np.max([lmin, np.min(lgrid)])
        lmax = np.min([lmax, np.max(lgrid), self.kNyq * self.h * self.d_A])

        self.log.debug(f'lmin: {lmin}, lmax: {lmax}')
        
        log_bins = np.linspace(lmin, lmax, nbins, dtype=np.float64)
        bin_indices = np.digitize(lgrid, log_bins)

        tmap = np.abs(self._fillgrid(self.get_cmap()))**2
        delta2 = tmap * (kgrid**3 / 2 / np.pi**2) #* ( self.box_size ** 3 / self.resolution ** 4 )
        # pkf = self._fillgrid(self.calc_primordial_power())
        # delta2 = pkf**2 * kgrid**3 / 2 / np.pi**2 * self.box_size ** 3 / self.resolution ** 4
        
        cls = np.zeros(nbins)
        counts = np.zeros(nbins)
        ls = np.zeros(nbins)
        for i in range(nbins):
            idx = (bin_indices == i + 1)
            counts[i] = np.sum(idx)
            if counts[i] != 0: 
                # self.log.debug('Calculating C_l for bin %d/%d' % (i + 1, nbins))
                cls[i] = 4 * np.pi * np.trapz( delta2[idx], kgrid[idx]) #/ counts[i]
                ls[i] = np.sum(lgrid[idx]) / counts[i]
                
        if raw_cls is False:
            cls *= ls * (ls + 1) / (2 * np.pi**2)
        self.log.debug('Done calculating C_ls.')
        print(cls)
        return ls, cls, counts
        
    def plot_cls(self, title=None, plot_theory=True, theory_spectra='unlensed_scalar'):               
        plt.figure()
        if plot_theory:
            theory = self.get_camb_cmb_pk(theory_spectra)
            plt.loglog(np.arange(len(theory)), theory[:, 0], label='Theory')
            
        ls, dls, counts = self.calculate_cls()
        print('dls >0: ',dls[dls >0])
        m = dls > 0
        plt.loglog(ls[m], dls[m]*10**7, label=r'$D_{\ell}$')
        
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
