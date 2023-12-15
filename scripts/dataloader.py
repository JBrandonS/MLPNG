import numpy as np
import h5py
from itertools import product

class DataLoader:
    def __init__(self, file, shuffle=True, seed=None, normalize=False):
        """
        Initializes the DataLoader object.

        Args:
            file (str): The path to the file to load data from.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            seed (int, optional): The seed for the random number generator used for shuffling. 
                If None, a random seed is generated. Defaults to None.
            normalize (bool, optional): Whether to normalize the data. Defaults to False.
        """

        self.file = file
        self.shuffle = shuffle
        self.normalize = normalize

        if seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            self.seed = seed

    def __call__(self):
        with h5py.File(self.file, mode='r', swmr=True, locking=False) as f:
            # This does not load the data into memory
            fnls = f['fnls']
            patches = f['patches']

            nsims, ndup, npol, npatches, nside, _ = patches.shape
            

            patch_vec = product(range(nsims), range(ndup), range(npol), range(npatches))
            if self.shuffle:
                np.random.seed(self.seed)
                patch_vec = np.random.permutation(list(patch_vec))
                 
            for (i,j,k,l) in patch_vec:
                # we also need to add the channel dimension as TF expects it
                patch = patches[i,j,k,l][:, :, None]
                
                if self.normalize:
                    min_val = np.min(patch)
                    max_val = np.max(patch)
                    patch = (patch - min_val) / (max_val - min_val)

                yield patch, fnls[i, j]