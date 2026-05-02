# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import sympy as sym
from pydae.bps.govs.agov1 import agov1
from pydae.bps.govs.dgov import dgov
from pydae.bps.govs.ggov1 import ggov1
from pydae.bps.govs.hygov import hygov
from pydae.bps.govs.ieeeg1 import ieeeg1
from pydae.bps.govs.ntsieeeg1 import ntsieeeg1
from pydae.bps.govs.tgov1 import tgov1
from pydae.bps.govs.tgov2 import tgov2


def add_gov(dae, syn_data, name, bus_name, backend=None):
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

    if syn_data['gov']['type'] == 'tgov1':
        tgov1(dae, syn_data, name, bus_name, backend)
    if syn_data['gov']['type'] == 'tgov2':
        tgov2(dae, syn_data, name, bus_name, backend)
    if syn_data['gov']['type'] == 'hygov':
        hygov(dae, syn_data, name, bus_name, backend)
    if syn_data['gov']['type'] == 'ggov1':
        ggov1(dae, syn_data, name, bus_name, backend)
    if syn_data['gov']['type'] == 'agov1':
        agov1(dae, syn_data, name, bus_name, backend)
    if syn_data['gov']['type'] == 'ntsieeeg1':
        ntsieeeg1(dae, syn_data, name, bus_name, backend)
    if syn_data['gov']['type'] == 'ieeeg1':
        ieeeg1(dae, syn_data, name, bus_name, backend)
    if syn_data['gov']['type'] == 'dgov':
        dgov(dae, syn_data, name, bus_name, backend)
