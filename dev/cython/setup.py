# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 18:14:08 2020

@author: jmmau
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy

#setup(
#    ext_modules = cythonize("convolve2.pyx",include_path = [numpy.get_include()])
#)



from distutils.core import setup, Extension


# setup(
#     ext_modules=[
#         Extension("convolve2", ["convolve2.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize("convolve2.pyx"),
    include_dirs=[numpy.get_include()]
)