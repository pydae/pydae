"""
pydae.core - Core DAE solver engine
====================================

This is the main entry point for building and running DAE models.

Quick start::

    from pydae.core import Builder, Model

    # Define your system symbolically
    sys_dict = { ... }

    # Build (generates compiled C code)
    builder = Builder(sys_dict, target='ctypes')
    builder.build()

    # Run
    model = Model('my_system')
    model.ini({'param': value}, xy_0=initial_guess)
    model.run(t_end, {'input': new_value})
    model.post()
"""

from pydae.core.builder.core import Builder
from pydae.core.model_class import Model

__all__ = ["Builder", "Model"]
