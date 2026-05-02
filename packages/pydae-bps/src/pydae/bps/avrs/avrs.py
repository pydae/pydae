# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import sympy as sym
from pydae.bps.avrs.avr_1 import avr_1
from pydae.bps.avrs.kundur import kundur
from pydae.bps.avrs.kundur_tgr import kundur_tgr
from pydae.bps.avrs.ntsst1 import ntsst1
from pydae.bps.avrs.ntsst4 import ntsst4
from pydae.bps.avrs.sexs import sexs
from pydae.bps.avrs.sst1 import sst1
from pydae.bps.avrs.st1 import st1
from pydae.bps.avrs.st4b import st4b


def add_avr(dae, syn_data, name, bus_name, backend=None):
    if backend is None:
        backend = type('Backend', (), {
            'symbols': lambda _, n, **k: sym.Symbol(n, real=True),
            'sin': sym.sin, 'cos': sym.cos, 'sqrt': sym.sqrt,
            'exp': sym.exp, 're': sym.re, 'im': sym.im,
            'Piecewise': sym.Piecewise, 'zeros': sym.zeros,
            'Matrix': sym.Matrix, 'I': sym.I,
            'func': lambda name: sym.Function(name),
            'use_casadi': False,
        })()

    if syn_data['avr']['type'] == 'sexs':
        sexs(dae, syn_data, name, bus_name, backend)

    if syn_data['avr']['type'] == 'kundur':
        kundur(dae, syn_data, name, bus_name, backend)

    if syn_data['avr']['type'] == 'ntsst4':
        ntsst4(dae, syn_data, name, bus_name, backend)

    if syn_data['avr']['type'] == 'ntsst1':
        ntsst1(dae, syn_data, name, bus_name, backend)

    if syn_data['avr']['type'] == 'sst1':
        sst1(dae, syn_data, name, bus_name, backend)

    if syn_data['avr']['type'] == 'st1':
        st1(dae, syn_data, name, bus_name, backend)

    if syn_data['avr']['type'] == 'st4b':
        st4b(dae, syn_data, name, bus_name, backend)

    if syn_data['avr']['type'] == 'kundur_tgr':
        kundur_tgr(dae, syn_data, name, bus_name, backend)

    if syn_data['avr']['type'] == 'avr_1':
        avr_1(dae, syn_data, name, bus_name, backend)

