import camb 
import logging 
import numpy as np
from numba import njit
from scipy.interpolate import interp1d
    
class CAMBHelper():
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
        'accuracy_boost': 4,
    }
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-picklable attributes
        del state['params']
        del state['results']
        del state['transfers']
        del state['d_A']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate non-picklable attributes
        self.cosmo = state.get('cosmo', self.default_cosmo_params)
        self.params = camb.CAMBparams()
        self.params.set_cosmology(H0=self.cosmo['h'] * 100, ombh2=self.cosmo['ombh2'], omch2=self.cosmo['omch2'], tau=self.cosmo['tau'])
        self.params.InitPower.set_params(As=self.cosmo['As'], ns=self.cosmo['ns'], r=self.cosmo['r'], pivot_scalar=self.cosmo['kpivot'])
        self.params.set_for_lmax(self.cosmo['lmax'], lens_potential_accuracy=self.cosmo['accuracy_boost'])
        self.params.set_accuracy(AccuracyBoost=self.cosmo['accuracy_boost'])
        
        self.results = camb.get_results(self.params)
        self.d_A = self.results.angular_diameter_distance(self.cosmo['z_recomb']) * 1000
        
        if self.ells is not None:
            self.calculate_transfers()
            
    def __reduce__(self):
        # Define a tuple of picklable attributes
        picklable_state = (None, True, self.log.level, self.cosmo, True, self.ells, self.Ls, self.qs, self.transfer_func)

        # Return a tuple with the class as the callable and the picklable attributes as its arguments
        return (CAMBHelper, picklable_state)
    
    def __init__(self, kgrid=None, get_transfers=True, log_level=logging.WARNING, cosmo_params=default_cosmo_params,
                 have_data=False, ells=None, Ls=None, qs=None, transfer_func=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(log_level)
        
        # Setup Camb parameters and run camb if given
        self.cosmo = cosmo_params if cosmo_params is not None else self.default_cosmo_params
        self.h = self.cosmo['h']
        
        self.log.debug('Setting up CAMB parameters object')
        self.params = camb.CAMBparams()
        self.params.set_cosmology(H0=self.cosmo['h']*100, ombh2=self.cosmo['ombh2'], omch2=self.cosmo['omch2'], tau=self.cosmo['tau'])
        self.params.InitPower.set_params(As=self.cosmo['As'], ns=self.cosmo['ns'], r=self.cosmo['r'], pivot_scalar=self.cosmo['kpivot'])  # type: ignore
        self.params.set_for_lmax(self.cosmo['lmax'], lens_potential_accuracy=self.cosmo['accuracy_boost'])
        self.params.set_accuracy(AccuracyBoost=self.cosmo['accuracy_boost'])
        self.log.debug('Done! (Setting up CAMB parameters object)')

        if not have_data:
            self.log.debug('Getting CAMB results...')
            self.results = camb.get_results(self.params)
            self.log.debug('CAMB results obtained!') 
            self.d_A = self.results.angular_diameter_distance(self.cosmo['z_recomb']) * 1000
                
            if kgrid is not None:
                self.ells = np.round(kgrid*self.h*self.d_A)
            else:
                self.ells = None
                
            if get_transfers:
                self.calculate_transfers()
        else:
            self.ells = ells
            self.Ls = Ls
            self.qs = qs
            self.transfer_func = transfer_func
            
            self.log.debug('Interpolating transfer functions with cubic splines for given data.')
            self.transfers = [interp1d(self.Ls, self.transfer_func[:, i], kind='cubic') for i in range(len(self.qs))]
            self.log.debug('Done!')
    
    def calculate_transfers(self):
        self.log.debug('Getting transfer functions...')
        self.Ls, self.qs, self.transfer_func = self.results.get_cmb_transfer_data().get_transfer()
        self.log.debug('Done! (Getting transfer functions)')
        
        self.log.debug('Interpolating transfer functions with cubic splines.')
        self.transfers = [interp1d(self.Ls, self.transfer_func[:, i], kind='cubic') for i in range(len(self.qs))]
        self.log.debug('Done!')
        return self.transfers
    
    def calculate_primordial_power(self, kgrid, cosmo=None, seed=None):
        """
        Calculate the primordial power spectrum
        """
        if cosmo is None:
            cosmo = self.cosmo

        pk = np.zeros_like(kgrid)
        idx = np.where(kgrid > 0)
        pfactor = cosmo['A']*cosmo['kpivot']**(1.-cosmo['ns'])
        pk[idx] = np.power(kgrid[idx], cosmo['ns']-4.)*pfactor
        return pk