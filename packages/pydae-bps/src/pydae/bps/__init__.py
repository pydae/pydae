"""
pydae.bps - Balanced Power Systems
==================================

This is the main entry point for building and running DAE models.

Quick start::

    from pydae.core import Builder, Model

    # Define your system symbolically
    sys_dict = { ... }

    # Build (generates compiled C code)
    builder = Builder(sys_dict, target='ctypes')
    builder.build()

"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pydae-bps")
except PackageNotFoundError:
    __version__ = "unknown"

from pydae.bps.bps_builder import BpsBuilder

__all__ = ["BpsBuilder", "__version__"]
