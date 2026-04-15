# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def leon_gvsg(grid,name,bus_name,data_dict):
    """
 
    """

    sin = sym.sin
    cos = sym.cos
    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]                  
    bus_name = vsg_data['bus']
    name = bus_name

    idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
    if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
    buses[idx_bus]['idx_powers'] += 1

    # inputs:
    p_ref,q_ref = sym.symbols(f'p_ref_{name},q_ref_{name}', real=True)

    # dynamical states:
    delta,x_v,x_w,xi_q = sym.symbols(f'delta_{name},x_v_{name},x_w_{name},xi_q_{name}', real=True)
    
    # algebraic states:
    i_d_ref,i_q_ref,p,q,e_qv = sym.symbols(f'i_d_ref_{name},i_q_ref_{name},p_{name},q_{name},e_qv_{name}', real=True)

    # params:
    S_base = sym.Symbol('S_base', real = True) # S_base = global power base, S_n = machine power base
    S_n,F_n,K_delta = sym.symbols(f'S_n_{name},F_n_{name},K_delta_{name}', real=True)
    F,T_d,K,H,D_s = sym.symbols(f'F_{name},T_d_{name},K_{name},H_{name},D_s_{name}', real=True)
    R_v,X_v = sym.symbols(f'R_v_{name},X_v_{name}', real=True)
    K_q,T_q = sym.symbols(f'K_q_{name},T_q_{name}', real=True)
    params_list = ['S_n','F_n','K_delta','F','T_d','K','H','D_s','R_v','X_v','K_q','T_q']

    # auxiliar variables and constants
    omega_coi = sym.Symbol("omega_coi", real=True) # from global system
    V = sym.Symbol(f"V_{bus_name}", real=True) # from global system
    theta = sym.Symbol(f"theta_{bus_name}", real=True) # from global system
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)

    # auxiliar equations
    Omega_b = 2*np.pi*F_n
    omega_s = omega_coi
    v_D = V*sin(theta)  # e^(-j)
    v_Q = V*cos(theta) 
    v_d = v_D * cos(delta) - v_Q * sin(delta)   
    v_q = v_D * sin(delta) + v_Q * cos(delta)

    Domega = (x_v + F*(p_ref - p))/K
    e_dv = 0.0
    epsilon_q = q_ref - q
    i_d = i_d_ref
    i_q = i_q_ref
    omega_v = Domega + 1.0

    # dynamical equations
    ddelta   = Omega_b*(omega_v - omega_s) - K_delta*delta
    dx_v = p_ref - p - D_s*Domega
    dx_w = x_v + T_d*(p_ref - p) - 2*H*Domega
    dxi_q = epsilon_q # PI agregado

    # algebraic equations
    g_i_d_ref  = e_dv - R_v * i_d_ref - X_v * i_q_ref - v_d 
    g_i_q_ref  = e_qv - R_v * i_q_ref + X_v * i_d_ref - v_q 
    g_p  = v_d*i_d + v_q*i_q - p  
    g_q  = v_d*i_q - v_q*i_d - q 
    g_e_qv = -e_qv +  K_q*(epsilon_q + xi_q/T_q) + 1.0
    
    # DAE system update
    grid.dae['f'] += [ddelta,dx_v,dx_w,dxi_q]
    grid.dae['x'] += [ delta, x_v, x_w, xi_q]
    grid.dae['g'] +=     [g_i_d_ref,g_i_q_ref,g_p,g_q,g_e_qv]
    grid.dae['y_ini'] += [  i_d_ref,  i_q_ref,  p,  q,  e_qv]
    grid.dae['y_run'] += [  i_d_ref,  i_q_ref,  p,  q,  e_qv]
            
    # default inputs
    grid.dae['u_ini_dict'].update({f'p_ref_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_ref_{name}':0.0})
    grid.dae['u_run_dict'].update({f'p_ref_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_ref_{name}':0.0})


    # default parameters
    for item in params_list:
        grid.dae['params_dict'].update({item + f'_{name}':data_dict[item]})


    # add speed*H term to global for COI speed computing
    grid.H_total += H
    grid.omega_coi_numerator += omega_v*H*S_n
    grid.omega_coi_denominator += H*S_n

    # DAE outputs update
    grid.dae['h_dict'].update({f"omega_v_{bus_name}":omega_v})


    p_W   = p * S_n
    q_var = q * S_n

    return p_W,q_var

