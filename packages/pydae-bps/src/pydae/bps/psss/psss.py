# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import sympy as sym
from pydae.bps.psss.pss1a import pss1a
from pydae.bps.psss.pss2a import pss2a
from pydae.bps.psss.pss_kundur_1 import pss_kundur_1
from pydae.bps.psss.pss_kundur_2 import pss_kundur_2
from pydae.bps.psss.pss_pss2 import pss_pss2


def add_pss(dae, syn_data, name, bus_name, backend=None):
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

    if syn_data['pss']['type'] == 'pss_kundur_1':
        pss_kundur_1(dae, syn_data, name, bus_name, backend)
    elif syn_data['pss']['type'] == 'pss2':
        pss_pss2(dae, syn_data, name, bus_name, backend)
    elif syn_data['pss']['type'] == 'pss_kundur_2':
        pss_kundur_2(dae, syn_data, name, bus_name, backend)
    elif syn_data['pss']['type'] == 'pss2a':
        pss2a(dae, syn_data, name, bus_name, backend)
    elif syn_data['pss']['type'] == 'pss1a':
        pss1a(dae, syn_data, name, bus_name, backend)
    else:
        print(f"PSS type {syn_data['pss']['type']} not found.")
