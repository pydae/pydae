# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np


def agov1(dae, syn_data, name, bus_name, backend=None):
    '''
    Governor AGOV1

    '''
    if backend is None:
        import sympy as sym
        backend = type('Backend', (), {
            'symbols': lambda _, n, **k: sym.Symbol(n, real=True),
        })()

    gov_data = syn_data['gov']

    # inputs
    omega = backend.symbols(f"omega_{name}")
    p_c = backend.symbols(f"p_c_{name}")
    p_g = backend.symbols(f"p_g_{name}")
    p_agc = backend.symbols("p_agc")
    p_r = backend.symbols(f"p_r_{name}")

    # dynamic states
    x_gov_1 = backend.symbols(f"x_gov_1_{name}")
    x_gov_2 = backend.symbols(f"x_gov_2_{name}")
    xi_imw  = backend.symbols(f"xi_imw_{name}")

    # algebraic states
    p_m = backend.symbols(f"p_m_{name}")
    p_m_ref = backend.symbols(f"p_m_ref_{name}")

    # parameters
    T_1 = backend.symbols(f"T_gov_1_{name}")
    T_2 = backend.symbols(f"T_gov_2_{name}")
    T_3 = backend.symbols(f"T_gov_3_{name}")
    Droop = backend.symbols(f"Droop_{name}")
    K_imw = backend.symbols(f"K_imw_{name}")
    K_sec = backend.symbols(f"K_sec_{name}")
    omega_ref = backend.symbols(f"omega_ref_{name}")

    # differential equations
    dx_gov_1 =   (p_m_ref - x_gov_1)/T_1
    dx_gov_2 =   (x_gov_1 - x_gov_2)/T_3
    dxi_imw =   K_imw*(p_c - p_g) - 1e-6*xi_imw

    g_p_m_ref  = -p_m_ref + xi_imw + p_r + K_sec*p_agc - 1/Droop*(omega - omega_ref)
    g_p_m = (x_gov_1 - x_gov_2)*T_2/T_3 + x_gov_2 - p_m

    dae['f'] += [dx_gov_1,dx_gov_2,dxi_imw]
    dae['x'] += [ x_gov_1, x_gov_2, xi_imw]
    dae['g'] += [g_p_m_ref,g_p_m]
    dae['y_ini'] += [  p_m_ref,  p_m]
    dae['y_run'] += [  p_m_ref,  p_m]
    dae['params_dict'].update({str(Droop):gov_data['Droop']})
    dae['params_dict'].update({str(T_1):gov_data['T_1']})
    dae['params_dict'].update({str(T_2):gov_data['T_2']})
    dae['params_dict'].update({str(T_3):gov_data['T_3']})
    dae['params_dict'].update({str(K_imw):gov_data['K_imw']})
    dae['params_dict'].update({str(K_sec):syn_data['K_sec']})
    dae['params_dict'].update({str(omega_ref):1.0})

    dae['u_ini_dict'].update({str(p_c):gov_data.get('p_c', 0.5)})
    dae['u_run_dict'].update({str(p_c):gov_data.get('p_c', 0.5)})
    dae['u_ini_dict'].update({str(p_r):0.0})
    dae['u_run_dict'].update({str(p_r):0.0})