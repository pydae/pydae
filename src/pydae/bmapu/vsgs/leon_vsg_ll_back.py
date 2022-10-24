# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def leon_vsg_ll(grid,name,bus_name,data_dict):
    """
    # auxiliar equations
    Omega_b = 2*np.pi*F_n
    omega_s = omega_coi
    v_D = V*sin(theta)  # e^(-j)
    v_Q = V*cos(theta) 
    v_d = v_D * cos(delta) - v_Q * sin(delta)   
    v_q = v_D * sin(delta) + v_Q * cos(delta)

    Domega = x_v + K_p * (p_ref - p)
    e_dv = 0.0
    epsilon_v = v_ref - V
    i_d = i_d_ref
    i_q = i_q_ref
    omega_v = Domega + 1.0
    q_ref_0 = K_p_v * epsilon_v + K_i_v * xi_v

    # dynamical equations
    ddelta   = Omega_b*(omega_v - omega_s) - K_delta*delta
    dx_v = K_i*(p_ref - p) - K_g*(omega_v - 1.0)
    de_qm = 1.0/T_q * (q - q_ref_0 - q_ref - e_qm) 
    dxi_v = epsilon_v # PI agregado

    # algebraic equations
    g_i_d_ref  = e_dv - R_v * i_d_ref - X_v * i_q_ref - v_d 
    g_i_q_ref  = e_qv - R_v * i_q_ref + X_v * i_d_ref - v_q 
    g_p  = v_d*i_d + v_q*i_q - p  
    g_q  = v_d*i_q - v_q*i_d - q 
    g_e_qv = 1.0 - e_qv - K_q*e_qm 

    {"bus":"1","type":"vsg_ll",'S_n':10e6,'F_n':50,'K_delta':0.0,
    'R_v':0.01,'X_v':0.1,'K_p':1.0,'K_i':0.1,'K_g':0.0,'K_q':20.0,
    'T_q':0.1,'K_p_v':1e-6,'K_i_v':1e-6}
    
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
    delta,x_v,e_qm,xi_v = sym.symbols(f'delta_{name},x_v_{name},e_qm_{name},xi_v_{name}', real=True)
    
    # algebraic states:
    i_d_ref,i_q_ref,p,q,e_qv = sym.symbols(f'i_d_ref_{name},i_q_ref_{name},p_{name},q_{name},e_qv_{name}', real=True)

    # params:
    S_base = sym.Symbol('S_base', real = True) # S_base = global power base, S_n = machine power base
    S_n,F_n,K_delta = sym.symbols(f'S_n_{name},F_n_{name},K_delta_{name}', real=True)
    K_p,K_i,K_g,K_i_q = sym.symbols(f'K_p_{name},K_i_{name},K_g_{name},K_i_q_{name}', real=True)
    R_v,X_v = sym.symbols(f'R_v_{name},X_v_{name}', real=True)
    K_q,T_q = sym.symbols(f'K_q_{name},T_q_{name}', real=True)
    K_p_v,K_i_v = sym.symbols(f'K_p_v_{name},K_i_v_{name}', real=True)
    params_list = ['S_n','F_n','K_delta','K_p','K_i','K_g','R_v','X_v','K_q','T_q','K_p_v','K_i_v']

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
    Domega = x_v + K_p * (p_ref - p)
    e_dv = 0.0
    epsilon_v = v_ref - V
    i_d = i_d_ref
    i_q = i_q_ref
    omega_v = Domega + 1.0
    q_ref_0 = K_p_v * epsilon_v + K_i_v * xi_v


    # dynamical equations
    ddelta   = Omega_b*(omega_v - omega_s) - K_delta*delta
    dx_v = K_i*(p_ref - p) - K_g*(omega_v - 1.0)
    de_qm = 1.0/T_q * (q - q_ref_0 - q_ref - e_qm) 
    dxi_v = epsilon_v # PI agregado

    # algebraic equations
    g_i_d_ref  = e_dv - R_v * i_d_ref - X_v * i_q_ref - v_d 
    g_i_q_ref  = e_qv - R_v * i_q_ref + X_v * i_d_ref - v_q 
    
    g_p  = v_d*i_d + v_q*i_q - p  
    g_q  = v_d*i_q - v_q*i_d - q 
    g_e_qv = 1.0 - e_qv - K_q*e_qm 
    
    # DAE system update
    grid.dae['f'] += [ddelta,dx_v,de_qm,dxi_v]
    grid.dae['x'] += [ delta, x_v, e_qm, xi_v]
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
    for item in params_list:
        grid.dae['params_dict'].update({item + f'_{name}':data_dict[item]})

    # add speed*H term to global for COI speed computing
    H = 4.0
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


if __name__ == "__main__":

    import pydae.build_cffi as db
    from pydae.bmapu import bmapu_builder
    from pydae.bmapu.vsgs.vsgs import add_vsgs
    import pydae.build_cffi as db
    import sympy as sym

    data = {
        "sys":{"name":"sys2buses","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.15,"R_pu":0.0, "S_mva":900.0}],
        "vsgs":[
            {"bus":"1","type":"vsg_ll",'S_n':10e6,'F_n':50,'K_delta':0.0,
            'R_v':0.01,'X_v':0.1,'K_p':1.0,'K_i':0.1,'K_g':0.0,'K_q':20.0,
            'T_q':0.1,'K_p_v':1e-6,'K_i_v':1e-6}],
        "genapes":[{"bus":"2","S_n":100e6,"F_n":50.0,"R_v":0.0,"X_v":0.1,"K_delta":0.001,"K_alpha":1.0}],
        }

    grid = bmapu_builder.bmapu(data)

    add_vsgs(grid)
    omega_coi = sym.Symbol("omega_coi", real=True)  

    grid.dae['g'] += [ -omega_coi + grid.omega_coi_numerator/grid.omega_coi_denominator]
    grid.dae['y_ini'] += [ omega_coi]
    grid.dae['y_run'] += [ omega_coi]

    sys_dict = {'name':'prueba','uz_jacs':True,
        'params_dict':grid.dae['params_dict'],
        'f_list':grid.dae['f'],
        'g_list':grid.dae['g'],
        'x_list':grid.dae['x'],
        'y_ini_list':grid.dae['y_ini'],
        'y_run_list':grid.dae['y_run'],
        'u_run_dict':grid.dae['u_run_dict'],
        'u_ini_dict':grid.dae['u_ini_dict'],
        'h_dict':grid.dae['h_dict']}

    bldr = db.builder(sys_dict)
    bldr.build()

    import prueba 
    model = prueba.model()
    model.ini({'p_ref_1':0.8},'xy_0_2.json')
    model.report_y()
