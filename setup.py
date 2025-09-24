from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="MLPNG",
    ext_modules=cythonize(
        [
            Extension(
                "mlpng.bispectrum.bispectrum",
                sources=["mlpng/bispectrum/bispectrum.pyx"],
                include_dirs=[numpy.get_include(), "mlpng/bispectrum"],
                language="c",
            )
        ],
    ),
    packages=["mlpng", "mlpng.bispectrum"],
)
