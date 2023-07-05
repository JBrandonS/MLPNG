import h5py
import numpy as np
from contextlib import contextmanager

class DataLoader:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, mode='r', swmr=True, locking=False) as f:
            # This does not load the data into memory
            fnls = f['fnls']
            patches = f['patches']

            nsims = fnls.shape[0]
            npol = patches.shape[1]
            npatches = patches.shape[2]

            for j in range(npol):
                for k in range(npatches):
                    for i in range(nsims):
                        yield patches[i, j, k], fnls[i]
