from itertools import product

import numpy as np

import h5py

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

            nsims, npol, npatches, _, _ = patches.shape
            # npol = patches.shape[1]
            # npatches = patches.shape[2]

            patch_vec = product(range(nsims), range(npol), range(npatches))

            if self.shuffle:
                np.random.seed(self.seed)
                patch_vec = np.random.permutation(list(patch_vec))
                 
            for (i,j,k) in patch_vec:
                yield patches[i, j, k], fnls[i]
