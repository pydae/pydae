# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def leon_evsg(grid,name,bus_name,data_dict):
    """

    "vsgs":[
        {"bus":"1","type":"leon_evsg",'S_n':100e6,'F_n':50,'K_delta':0.0,
        'T':0.1520,'F':0.0019,'D_s':0,'H':5,'K':0.1382,'R_v':0.0,'X_v':0.2,'K_g':1047.2,
        'K_q':20.0,
        'T_q':0.1,'K_p_v':1e-6,'K_i_v':1e-6,
        'K_delta':0}],
    
    """

    sin = sym.sin
    cos = sym.cos
    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]                  

    idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
    if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
    buses[idx_bus]['idx_powers'] += 1

    # inputs:
    q_l,q_r,v_ref = sym.symbols(f'q_l_{name},q_r_{name},v_ref_{name}', real=True)
    p_l,p_r = sym.symbols(f'p_l_{name},p_r_{name}', real=True)

    # dynamical states:
    delta,x_v,e_qm,xi_v,x_w = sym.symbols(f'delta_{name},x_v_{name},e_qm_{name},xi_v_{name},x_w_{name}', real=True)
    
    # algebraic states:
    i_d_ref,i_q_ref,p,q,e_qv = sym.symbols(f'i_d_ref_{name},i_q_ref_{name},p_{name},q_{name},e_qv_{name}', real=True)

    # params:
    S_base = sym.Symbol('S_base', real = True) # S_base = global power base, S_n = machine power base
    S_n,F_n,K_delta = sym.symbols(f'S_n_{name},F_n_{name},K_delta_{name}', real=True)
    R_v,X_v = sym.symbols(f'R_v_{name},X_v_{name}', real=True)
    K_q,T_q = sym.symbols(f'K_q_{name},T_q_{name}', real=True)
    T,F,D_s,H,K = sym.symbols(f'T_{name},F_{name},D_s_{name},H_{name},K_{name}', real=True)
    K_p_v,K_i_v = sym.symbols(f'K_p_v_{name},K_i_v_{name}', real=True)

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

    p_ref = p_l + p_r
    q_ref = q_l + q_r
    epsilon_p = p_ref - p
    Domega = (x_w + F*epsilon_p)/K

    e_dv = 0.0
    epsilon_v = v_ref - V
    i_d = i_d_ref
    i_q = i_q_ref
    omega_v = Domega + 1.0
    q_ref_0 = K_p_v * epsilon_v + K_i_v * xi_v


    # dynamical equations
    ddelta   = Omega_b*(omega_v - omega_s) - K_delta*delta
    dx_v = epsilon_p - D_s*Domega
    dx_w = x_v + T*epsilon_p - 2*H*Domega
    de_qm = 1.0/T_q * (q - q_ref_0 - q_ref - e_qm) 
    dxi_v = epsilon_v # PI agregado

    # algebraic equations
    g_i_d_ref  =  - R_v * i_d_ref - X_v * i_q_ref - v_d + e_dv 
    g_i_q_ref  =  - R_v * i_q_ref + X_v * i_d_ref - v_q + e_qv 
    
    g_p  = v_d*i_d + v_q*i_q - p  
    g_q  = v_d*i_q - v_q*i_d - q 
    g_e_qv = 1.0 - e_qv - K_q*e_qm 
    
    # DAE system update
    grid.dae['f'] += [ddelta,dx_v,dx_w,de_qm,dxi_v]
    grid.dae['x'] += [ delta, x_v, x_w, e_qm, xi_v]
    grid.dae['g'] +=     [g_i_d_ref,g_i_q_ref,g_p,g_q,g_e_qv]
    grid.dae['y_ini'] += [  i_d_ref,  i_q_ref,  p,  q,  e_qv]
    grid.dae['y_run'] += [  i_d_ref,  i_q_ref,  p,  q,  e_qv]
            
    # default inputs
    grid.dae['u_ini_dict'].update({f'p_l_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_l_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'p_r_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_r_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'v_ref_{name}':1.0})
    grid.dae['u_run_dict'].update({f'p_l_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_l_{name}':0.0})
    grid.dae['u_run_dict'].update({f'p_r_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_r_{name}':0.0})
    grid.dae['u_run_dict'].update({f'v_ref_{name}':1.0})

    # default parameters
    params_list = ['S_n','F_n',
                   'T','F','D_s','H','K','R_v','X_v','K_g',
                   'T_q','K_q','K_p_v','K_i_v','K_delta']
    for item in params_list:
        grid.dae['params_dict'].update({f'{item}_{name}':data_dict[item]})

    # add speed*H term to global for COI speed computing
    H = data_dict['H']
    grid.H_total += H
    grid.omega_coi_numerator += omega_v*H*S_n
    grid.omega_coi_denominator += H*S_n

    # DAE outputs update
    grid.dae['h_dict'].update({f"omega_v_{name}":omega_v})
    grid.dae['h_dict'].update({f"p_ref_{name}":p_ref})
    grid.dae['h_dict'].update({f"q_ref_{name}":q_ref})
    grid.dae['h_dict'].update({f"v_ref_{name}":v_ref})
    grid.dae['h_dict'].update({f"i_sd_{name}":i_d_ref})    
    grid.dae['h_dict'].update({f"i_sq_{name}":i_q_ref})

    grid.dae['xy_0_dict'].update({str(e_qv):1.0}) 

    p_W   = p * S_n
    q_var = q * S_n

    return p_W,q_var


def test_build():

    import pydae.build_cffi as db
    from pydae.bmapu import bmapu_builder
    import pydae.build_cffi as db

    grid = bmapu_builder.bmapu('leon_evsg.hjson')
    grid.uz_jacs = True
    grid.verbose = True
    grid.build('temp')

def test_ini():

    import temp 
    model = temp.model()
    model.ini({},'xy_0.json')

    model.report_x()
    model.report_y()


if __name__ == "__main__":

    test_build()
    test_ini()