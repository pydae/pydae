"""
pydae.uds - Unbalanced Distribution Systems Builder
=====================================================

Previously known as "urisi". This package reads 3-phase unbalanced
distribution network descriptions and constructs the DAE system
dictionary for pydae.core.Builder.

Usage::

    from pydae.uds import UdsBuilder

    net = UdsBuilder('distribution_network.json')
    net.construct('my_network')

    from pydae.core import Builder
    bld = Builder(net.sys_dict, target='ctypes')
    bld.build()

Migration from urisi
---------------------
Old:  from pydae.urisi import urisi_builder
New:  from pydae.uds import UdsBuilder
"""

# TODO: Move urisi source code here and rename the main class
# from pydae.uds.builder import UdsBuilder
# __all__ = ["UdsBuilder"]
