# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np


def genape_inf(grid, name, bus_name, data_dict):
    '''

    parameters
    ----------

    S_n: nominal power in VA
    F_n: nominal frequency in Hz
    X_v: coupling reactance in pu (base machine S_n)
    R_v: coupling resistance in pu (base machine S_n)
    K_delta: if K_delta>0.0 current generator is converted to reference machine
    K_alpha: alpha gain to obtain Domega integral

    inputs
    ------

    alpha: RoCoF in pu if K_alpha = 1.0
    omega_ref: frequency in pu
    v_ref: internal voltage reference

    example
    -------

    "genapes": [{"S_n":1e9,"F_n":50.0,"X_v":0.001,"R_v":0.0,"K_delta":0.001,"K_alpha":1e-6}]

    S_n = sym.Symbol(f"S_n_{name}", real=True)
    F_n = sym.Symbol(f"F_n_{name}", real=True)
    X_v = sym.Symbol(f"X_v_{name}", real=True)
    R_v = sym.Symbol(f"R_v_{name}", real=True)
    K_delta = sym.Symbol(f"K_delta_{name}", real=True)
    K_alpha = sym.Symbol(f"K_alpha_{name}", real=True)


    '''

    backend = grid.backend
    sin = backend.sin
    cos = backend.cos

    # inputs
    V = backend.symbols(f"V_{bus_name}")
    theta = backend.symbols(f"theta_{bus_name}")
    omega_coi = backend.symbols("omega_coi")
    alpha = backend.symbols(f"alpha_{name}")
    v_ref = backend.symbols(f"v_ref_{name}")
    omega_ref = backend.symbols(f"omega_ref_{name}")
    phi = backend.symbols(f"phi_{name}")
    delta_ref = backend.symbols(f"delta_ref_{name}")
    rocov = backend.symbols(f"rocov_{name}")

    # dynamic states
    delta = backend.symbols(f"delta_{name}")
    omega = backend.symbols(f"omega_{name}")
    Domega = backend.symbols(f"Domega_{name}")
    Dv = backend.symbols(f"Dv_{name}")

    # algebraic states
    i_d = backend.symbols(f"i_d_{name}")
    i_q = backend.symbols(f"i_q_{name}")
    p_s = backend.symbols(f"p_s_{name}")
    q_s = backend.symbols(f"q_s_{name}")

    # parameters
    S_n = backend.symbols(f"S_n_{name}")
    F_n = backend.symbols(f"F_n_{name}")
    X_v = backend.symbols(f"X_v_{name}")
    R_v = backend.symbols(f"R_v_{name}")
    K_delta = backend.symbols(f"K_delta_{name}")
    K_alpha = backend.symbols(f"K_alpha_{name}")
    K_rocov = backend.symbols(f"K_rocov_{name}")

    params_list = ['S_n', 'F_n', 'X_v', 'R_v', 'K_delta', 'K_alpha']

    # auxiliar
    theta_v = delta + phi
    v_d = V * sin(theta_v - theta)
    v_q = V * cos(theta_v - theta)
    Omega_b = 2 * np.pi * F_n
    omega_s = omega_coi
    e_dv = 0
    e_qv = v_ref + Dv

    # dynamic equations
    ddelta = Omega_b * (omega - omega_s) - K_delta * (delta - delta_ref)
    dDomega = alpha - K_alpha * Domega
    dDv = rocov - K_rocov * Dv

    # algebraic equations
    g_omega = -omega + Domega + omega_ref
    g_i_d = -R_v * i_d + X_v * i_q - v_d + e_dv
    g_i_q = -R_v * i_q - X_v * i_d - v_q + e_qv
    g_p_s = i_d * v_d + i_q * v_q - p_s
    g_q_s = i_d * v_q - i_q * v_d - q_s

    # dae
    f_vsg = [ddelta, dDomega, dDv]
    x_vsg = [delta, Domega, Dv]
    g_vsg = [g_omega, g_i_d, g_i_q, g_p_s, g_q_s]
    y_vsg = [omega, i_d, i_q, p_s, q_s]

    H = 1e6
    grid.H_total += H
    grid.omega_coi_numerator += omega * H * S_n
    grid.omega_coi_denominator += H * S_n

    grid.dae['f'] += f_vsg
    grid.dae['x'] += x_vsg
    grid.dae['g'] += g_vsg
    grid.dae['y_ini'] += y_vsg
    grid.dae['y_run'] += y_vsg

    grid.dae['u_ini_dict'].update({str(alpha): 0})
    grid.dae['u_run_dict'].update({str(alpha): 0})

    grid.dae['u_ini_dict'].update({str(v_ref): 1.0})
    grid.dae['u_run_dict'].update({str(v_ref): 1.0})

    grid.dae['u_ini_dict'].update({str(omega_ref): 1.0})
    grid.dae['u_run_dict'].update({str(omega_ref): 1.0})

    grid.dae['u_ini_dict'].update({str(delta_ref): 0.0})
    grid.dae['u_run_dict'].update({str(delta_ref): 0.0})

    grid.dae['u_ini_dict'].update({str(phi): 0.0})
    grid.dae['u_run_dict'].update({str(phi): 0.0})

    grid.dae['u_ini_dict'].update({str(rocov): 0.0})
    grid.dae['u_run_dict'].update({str(rocov): 0.0})

    grid.dae['xy_0_dict'].update({str(omega): 1.0})

    # outputs
    grid.dae['h_dict'].update({f"alpha_{name}": alpha})
    grid.dae['h_dict'].update({f"Dv_{name}": Dv})
    grid.dae['h_dict'].update({f"theta_v_{name}": theta_v})

    for item in params_list:
        grid.dae['params_dict'].update({f"{item}_{name}": data_dict[item]})

    grid.dae['params_dict'].update({f"K_rocov_{name}": 1e-6})

    p_W = p_s * S_n
    q_var = q_s * S_n

    return p_W, q_var
