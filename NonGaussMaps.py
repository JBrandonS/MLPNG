import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pylab

def ngTemperatureMapFromFits(alml_fname, almnl_fname, f_NL=100, pol=True, nside=512):
    """
    Returns a non-gaussian temperature map from the alm files
    """
    lhdus = (1, 2, 3) if pol else 1
    alm_l = hp.read_alm(alml_fname, hdu=lhdus)
    alm_nl = hp.read_alm(almnl_fname, hdu=lhdus)
    alm = alm_l + (f_NL * alm_nl)
    
    ngmap = hp.alm2map(alm, nside=nside, pol=pol)
    return ngmap[0]

def cutSqPatches(fullsky_map, npix_side, side_deg, num_patches=10):
    """Cuts square patches by default (Even number for now)... without taking into account edge cases... with a simple method
    
    Output: numpy array with (num_patches, npix_side, npix_side) shape
    """
    # Prevents a bug with cartview 
    wasinteractive = pylab.isinteractive()
    pylab.ioff()
    
    # Initialize numpy array
    Tmap_datat = np.zeros((num_patches//2, int(npix_side), int(npix_side)))
    Tmap_datab = np.zeros((num_patches//2, int(npix_side), int(npix_side)))

    for counter in range(num_patches//2):
        Tmap_datat[counter] = np.ma.getdata(hp.cartview(fullsky_map, fig=0, xsize=256, ysize=256, rot=[0, 0], lonra=[side_deg*counter, side_deg*(counter+1)], latra=[0, 10],
                                                       title="CartView", unit="mK", format="%.2g", return_projected_map=True))
        Tmap_datab[counter] = np.ma.getdata(hp.cartview(fullsky_map, fig=1, xsize=256, ysize=256, rot=[0, 0], lonra=[side_deg*counter, side_deg*(counter+1)], latra=[-10, 0],
                                                       title="CartView", unit="mK", format="%.2g", return_projected_map=True))
    if wasinteractive:
        pylab.ion()
        pylab.draw()
    return np.concatenate((Tmap_datat, Tmap_datab))

def getTmapData(data_path="/users/stevensonb/scratch/mlpng-fits/alm_", 
                num_files=1000, 
                f_NL_base=100,
                f_NL_min=0,
                f_NL_chance=0.9, 
                xnpix=256, 
                ynpix=256, 
                m_deg_size=10,  
                num_maps=10000, 
                num_patches_per_map=10, 
                patch_set=10,
                verbose=False,
                save=True, 
                save_name="NGTmap_100_fNL"):
    
    fnames = [(data_path + "l_" + str(num).zfill(4) + "_v3.fits", data_path + "nl_" + str(num).zfill(4) + "_v3.fits") for num in range(1, num_files+1)]

    Tmap_data = np.zeros((num_maps, xnpix, ynpix))
    f_NL_labels = np.zeros((num_maps, 1))
    for count, f in enumerate(fnames):    
        if np.random.random() < f_NL_chance:
            f_NL = f_NL_base
        else:
            f_NL = f_NL_min

        full_Tmap = ngTemperatureMapFromFits(f[0], f[1], f_NL)
        Tmap_data[(0 + patch_set*count) : patch_set*(count+1)] = cutSqPatches(full_Tmap, xnpix, m_deg_size, num_patches_per_map)
        if verbose:
            print((0 + patch_set*count), "-" ,patch_set*(count+1), "with f_NL of", f_NL)

        # saving f_NL values used
        f_NL_labels[(0 + patch_set*count) : patch_set*(count+1)] = f_NL
    
    if save:
        np.savez(save_name, Tmapdata=Tmap_data, fNLs=f_NL_labels)
        
    return Tmap_data, f_NL_labels