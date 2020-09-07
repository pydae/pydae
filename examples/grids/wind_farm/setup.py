# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:48:20 2020

@author: jmmau
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fib.pyx"),
)