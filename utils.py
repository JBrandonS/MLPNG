from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from threading import Thread

from pixell import enmap, lensing, curvedsky
from psutil import Process

import h5py

import time
import os

class DataLoader:
    def __init__(self, file, shuffle=True, seed=0):
        self.file = file
        self.shuffle = shuffle
        self.seed = seed

    def __call__(self):
        with h5py.File(self.file, mode='r', swmr=True, locking=False) as f:
            # This does not load the data into memory
            fnls = f['fnls']
            patches = f['patches']

            nsims, npol, npatches, nside, _ = patches.shape

            patch_vec = product(range(nsims), range(npol), range(npatches))
            if self.shuffle:
                np.random.seed(self.seed)
                patch_vec = np.random.permutation(list(patch_vec))
                 
            for (i,j,k) in patch_vec:
                yield np.ascontiguousarray(patches[i, j, k]), fnls[i]

class MemoryMonitor(Thread):
    """Monitor the memory usage in MB in a separate thread.

    Note that this class is good enough to highlight the memory profile of
    Parallel in this example, but is not a general purpose profiler fit for
    all cases.
    """
    def __init__(self):
        super().__init__()
        self.stop = False
        self.memory_buffer = []
        self.start()

    def get_memory(self):
        "Get memory of a process and its children."
        p = Process()
        memory = p.memory_info().rss
        for c in p.children():
            memory += c.memory_info().rss
        return memory

    def run(self):
        memory_start = self.get_memory()
        while not self.stop:
            self.memory_buffer.append(self.get_memory() - memory_start)
            time.sleep(0.2)

    def join(self):
        self.stop = True
        super().join()

    def join_and_plot(self, plot_dir, save_name):
        self.join()
        peak = max(self.memory_buffer) / 1e9
        dp(f"Peak memory usage: {peak:.2f}GB")
        plt.figure()
        plt.title(f"Peak memory usage: {peak:.2f}GB")

        plt.semilogy(
            np.maximum.accumulate(self.memory_buffer),
        )
        plt.xlabel("Time")
        plt.xticks([], [])
        plt.ylabel("Memory usage")
        plt.yticks([1e9, 1e10, 1e11, 1e12], ['1GB', '10GB', '100GB', '1TB'])
        plt.show()
        if save_name is not None:
            save_plt(plot_dir, save_name)

def dp(*args, **kwargs):
    for arg in args:
        print(arg, end=' ')
    for key, value in kwargs.items():
        print(f"{key}: {value}", end=' ')
    print()  # Print a newline at the end

def safe_makedirs(dir):
    if not os.path.exists(dir): 
        # Handles a race condition found during array jobs
        try:
            os.makedirs(dir)
            dp(f'Created directory {dir}')
            return
        except FileExistsError:
            pass

    # dp(f'Reusing directory {dir}')

def save_data(file_path, data_dict):
    with h5py.File(file_path, 'a') as hf:
        for key, value in data_dict.items():
            if key in hf:
                # Resize the dataset to accommodate the new data
                hf[key].resize((hf[key].shape[0] + value.shape[0],) + value.shape[1:]) # type: ignore
                # Append the new data
                hf[key][-value.shape[0]:] = value # type: ignore
            else:
                # Create a new dataset for this key
                hf.create_dataset(key, data=value, maxshape=(None,) + value.shape[1:])

def rename_save(old, new):
    os.replace(old, new)

def load_data(data_file, key, start_index=None, end_index=None):
    dp('Loading data', key, 'from', data_file)
    with h5py.File(data_file, 'r') as hdf:
        if start_index is not None and end_index is not None:
            dp(' => Loading data from', start_index, 'to', end_index)
            return np.array(hdf[key][start_index:end_index]) # type: ignore
        else:
            return np.array(hdf.get(key)[()]) # type: ignore

def save_plt(plot_dir, name):
    file = f'{plot_dir}/{name}.png'
    plt.savefig(file)

def plot_cl(cl,
            plot_noise=True, 
            plt_func=plt.semilogy,
            plt_camb=True,
            title='Angular power spectrum from cl',
            label='data',
            save_name=None,
            lmax=2000):
    ell = np.arange(len(cl))
    plt_func(ell[2:], (ell * (ell + 1) / 2 / np.pi)[2:] * cl[2:], label=label)

    if plt_camb:
        camb_ls = np.arange(2, lmax)
        if plot_noise:
            noise_ell_b = np.array([noise_scale_tt.to_value(u.radian)**2 * np.exp( (l*(l+1) * beam_width.to_value(u.radian)**2) / (8*np.log(2)) ) for l in range(nell)])
            camb_cls_n = c_ells['c_ell'][2:lmax] + noise_ell_b[2:lmax, np.newaxis]
            camb_n_inner_plt = camb_ls * (camb_ls + 1) / 2 / np.pi * camb_cls_n[:, 0]
            plt_func(camb_ls, camb_n_inner_plt, label='camb')
        else:
            camb_inner_plt = camb_ls * (camb_ls + 1) / 2 / np.pi * c_ells['c_ell'][2:lmax][:, 0]
            plt_func(camb_ls, camb_inner_plt, label='camb + noise')

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\ell(\ell+1)/2\pi\;C_{\ell}$")
    plt.title(title)
    plt.legend()
    plt.grid()

    if save_name is not None:
        save_plt(plot_dir, save_name)

    plt.show()

def plot_cl_alm(alm, plt_func=plt.semilogy, plt_camb=True, title='Angular power spectrum from alm', save_name=None):
    cl = curvedsky.alm2cl(alm)
    plot_cl(cl, plt_func, plt_camb, title, save_name)

def plot_cl_map(map, wcs, plt_func=plt.semilogy, plt_camb=True, title='Angular power spectrum from map', save_name=None):   
    tmap = enmap.ndmap(map, wcs)
    almsd = curvedsky.map2alm(tmap, lmax=lmax)
    cl = curvedsky.alm2cl(almsd)
    plot_cl(cl, plt_func, plt_camb, title, save_name)

def get_radii(r_min, r_max):
    # For the radii we follow Table 2. of Smith and Zaldarriaga which gives a greater density of points near reionization and recombination. 
    # 
    # Spacing for all ranges but the last row are linear, with the last row having log spacing.
    # 
    # radii are in Mpc
    radii = []
    #          start,  stop, resolution
    ranges = [(    0,  9500, 150), 
            ( 9500, 11000, 300), 
            (11000, 13800, 150), 
            (13800, 14600, 400), 
            (14600, 16000, 100), 
            (16000, 50000, 100)]

    for r in ranges:
        start = max(r_min, r[0])
        end = min(r_max, r[1])

        if start > end:
            continue

        if r == ranges[-1]: # For the last range, use logspace
            temp_radii = np.logspace(np.log10(start), np.log10(end), num=r[2])
        else:
            temp_radii = np.linspace(start, end, num=r[2], endpoint=False)

        radii.extend(temp_radii)

    radii = np.array([r for r in radii if r_min <= r < r_max])
    drs = np.diff(radii)
    return radii, drs