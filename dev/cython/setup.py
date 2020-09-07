# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 18:14:08 2020

@author: jmmau
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("hello_memview.pyx")
)