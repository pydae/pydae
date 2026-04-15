"""
pydae.bps - Balanced Power Systems Builder
============================================

Previously known as "bmapu". This package reads power system network
descriptions (JSON/HJSON) and constructs the DAE system dictionary
that pydae.core.Builder can compile and simulate.

Usage::

    from pydae.bps import BpsBuilder

    grid = BpsBuilder('ieee39.json')
    grid.construct('ieee39')

    from pydae.core import Builder
    bld = Builder(grid.sys_dict, target='ctypes')
    bld.build()

Migration from bmapu
---------------------
Old:  from pydae.bmapu import bmapu_builder
New:  from pydae.bps import BpsBuilder

The API is the same, only the import path and class name changed.
"""

# TODO: Move bmapu source code here and rename the main class
# from pydae.bps.builder import BpsBuilder
# __all__ = ["BpsBuilder"]
