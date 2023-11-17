# -*- coding: utf-8 -*-
"""
Created 2023 March 02? at ??:?? ?M.

@author: jwryan
"""

import sys
import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import camb
import lenspyx
from lenspyx.utils_hp import synalm, almxfl, alm2cl
#from IPython.display import clear_output
import h5py

n = sys.argv[1]
scount = 0

for N in range(0, 1000, 1):
    scount += 1
    if str(scount) == n:
        break

xp = 512
yp = 512

"""
data_path = "/work/users/jwryan/"
base1 = "alm_l_0001_v3.fits"
base2 = "alm_nl_0001_v3.fits"

with fits.open(data_path + base1) as hdul:
#hdul = fits.open(data_path + base1)
    print(hdul[1].header) # Printing more information for data check

''' #Probably unnecessary.
alm_l = hp.read_alm(data_path + base1, hdu=(1, 2, 3))
alm_nl = hp.read_alm(data_path + base2, hdu=(1, 2, 3))

f_NL = 100#250
alm = alm_l + (f_NL * alm_nl)
ngmap = hp.alm2map(alm, 1024, pol=True)
'''
"""

DATA_FILE = '/work/group/jrmeyers/png_ml/data-bs/alm_cache/512_T_1000.alms.hdf5'

# Defining relevant functions

def load_single_data(data_file, key, index, verbose=False):
    if verbose:
        print('Loading data', key, 'from', data_file, 'index', index)
        
    with h5py.File(data_file, 'r', swmr=True, locking=False) as hdf:
        kv = hdf.get(key, None)
        if kv is None:
            raise ValueError(f'Key {key} not found in {data_file}')
        else:
            return np.array(kv[index]) # type: ignore
            
def ngTemperatureMapFrom_hdf5(idx, f_NL_val):
    alm = load_single_data(DATA_FILE, 'alm', idx, verbose=False).astype(complex)
    print(alm.shape)
    almng = load_single_data(DATA_FILE, 'almng', idx, verbose=False).astype(complex)
    print(almng.shape)
    #fnl = load_single_data(DATA_FILE, 'fnls', idx, verbose=False)
    ngmap = hp.alm2map(alm + (f_NL_val*almng), 512, pol=False)
    return ngmap

def cutSqPatches(fullsky_map, npix_side, side_deg, num_patches=10):
    """Cuts square patches by default (Even number for now)... without taking into account edge cases... with a simple method
    
    Output: numpy array with (num_patches, npix_side, npix_side) shape
    """
    
    import pylab #addition
    wasinteractive = pylab.isinteractive() #addition
    pylab.ioff() #addition
    
    # Initialize numpy array
    Tmap_datat = np.zeros((num_patches//2, int(npix_side), int(npix_side)))
    Tmap_datab = np.zeros((num_patches//2, int(npix_side), int(npix_side)))

    for counter in range(num_patches//2):
        cv = hp.cartview(fullsky_map, xsize=xp, ysize=yp, rot=[0, 0], lonra=[side_deg*counter, side_deg*(counter+1)], latra=[0, 10],
                                                       title="CartView", unit="mK", format="%.2g", return_projected_map=True)
        d = np.ma.getdata(cv)
        print('d shape:', np.shape(d))
        Tmap_datat[counter] = d
        Tmap_datab[counter] = np.ma.getdata(hp.cartview(fullsky_map, xsize=xp, ysize=yp, rot=[0, 0], lonra=[side_deg*counter, side_deg*(counter+1)], latra=[-10, 0],
                                                       title="CartView", unit="mK", format="%.2g", return_projected_map=True))
        
        pylab.close('all') #addition
        
    if wasinteractive: #addition
        pylab.ion() #addition
    return np.concatenate((Tmap_datat, Tmap_datab))

str_idx = '1000'

# Creating the initial data set of 10000 maps
# res_arcmin = 2.34375
xnpix = xp
ynpix = yp
# m_deg_size = int(xnpix * (res_arcmin / 60))
m_deg_size = 10 # in cart projection
f_NL_base = 100

#Tmap_data = np.zeros((10000, xnpix, ynpix))
npatches_per_map = patch_set = 10 #10, 72, 36, 6
#f_NL_labels = np.zeros((10000, 1))

f_NL_history = []

for bcount in range(1, 21, 1): #1, 11, 1
    for IDX in range(0, int(str_idx), 1):
        if (scount - 1) == IDX:
            print(scount - 1)
            print(IDX)
            f_NL_labels = np.zeros((npatches_per_map, 1))
            
            #f_NL = (1/100)*np.random.randint(0, 1000)
            f_NL = np.random.randint(-1000, 1000) #1000, 500
            #f_NL = 10
            
            
            print(f_NL)
            
            if bcount > 1:
                while f_NL in f_NL_history:
                    f_NL = np.random.randint(-1000, 1000)
            
            f_NL_history.append(f_NL)
            print(f_NL_history, type(f_NL_history))
            
            #print(type(hp.read_alm(f[0], hdu=1)))
            
            
            full_Tmap = ngTemperatureMapFrom_hdf5(IDX, f_NL) ##UNLENSED
            #full_Tmap = lensed_ngTemperatureMapFromFits(f[0], f[1], f_NL) ##LENSED
            Tmap_data = cutSqPatches(full_Tmap[0], xnpix, m_deg_size, npatches_per_map)
    
            # saving f_NL values used
            f_NL_labels[:] = f_NL
        
            #full_Tmap = ngTemperatureMapFromFits(f[0], f[1], f_NL)
            #Tmap_data[(0 + patch_set*count) : patch_set*(count+1)] = cutSqPatches(full_Tmap, xnpix, m_deg_size, npatches_per_map)
            #print((0 + patch_set*count), "-" , patch_set*(count+1), "with", "f_NL of", f_NL)
        
            ## saving f_NL values used
            #f_NL_labels[(0 + patch_set*count) : patch_set*(count+1)] = f_NL
            
            #np.savez("/work/users/jwryan/NGM/NGTmap_sampled_fNL_pos_10_range_128x128_ngmap_512_" + str(npatches_per_map) + "k_batch_" + str(bcount) + "_" + str(scount - 1), Tmapdata=Tmap_data, fNLs=f_NL_labels)
            #np.savez("/work/users/jwryan/NGM/NGTmap_sampled_fNL_1000_range_128x128_ngmap_512_" + str(npatches_per_map) + "k_batch_" + str(bcount) + "_" + str(scount - 1) + '_tvt_uncontaminated', Tmapdata=Tmap_data, fNLs=f_NL_labels)
            np.savez("Brandon_sim_unlensed_" + str(k) + "Cl_NGTmap_sampled_fNL_1000_range_128x128_ngmap_512_" + str(npatches_per_map) + "k_batch_" + str(bcount) + "_" + str(scount - 1) + '_tvt_uncontaminated', Tmapdata=Tmap_data, fNLs=f_NL_labels)
            #np.savez("/work/users/jwryan/NGM/lensed_" + str(k) + "Cl_NGTmap_sampled_fNL_1000_range_128x128_ngmap_512_" + str(npatches_per_map) + "k_batch_" + str(bcount) + "_" + str(scount - 1) + '_tvt_uncontaminated', Tmapdata=Tmap_data, fNLs=f_NL_labels)
            #np.savez("/work/users/jwryan/NGM/Bispec_calc_NGTmap_sampled_fNL_10_128x128_ngmap_512_" + str(npatches_per_map) + "k_batch_" + str(bcount) + "_" + str(scount - 1), Tmapdata=Tmap_data, fNLs=f_NL_labels)
            
            break



