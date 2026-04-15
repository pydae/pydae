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

    ctrl_data = data_dict['ctrl']

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]                  

    idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
    if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
    buses[idx_bus]['idx_powers'] += 1

    # inputs:
    q_l,q_r,v_ref = sym.symbols(f'q_l_{name},q_r_{name},v_ref_{name}', real=True)
    p_l,p_r = sym.symbols(f'p_l_{name},p_r_{name}', real=True)
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)   
    Ddelta_ff,Domega_ff = sym.symbols(f'Ddelta_ff_{name},Domega_ff_{name}', real=True)

    # dynamical states:
    delta,x_v,e_qm,xi_v = sym.symbols(f'delta_{name},x_v_{name},e_qm_{name},xi_v_{name}', real=True)
    
    # algebraic states:
    p_s = sym.Symbol(f'p_s_{name}', real=True)
    q_s = sym.Symbol(f'q_s_{name}', real=True)
    e_vq = sym.Symbol(f"e_vq_{name}", real=True)
    v_td_ref = sym.Symbol(f"v_td_ref_{name}", real=True)
    v_tq_ref = sym.Symbol(f"v_tq_ref_{name}", real=True) 

    # params:
    S_base = sym.Symbol('S_base', real = True) # S_base = global power base, S_n = machine power base
    S_n,F_n,K_delta = sym.symbols(f'S_n_{name},F_n_{name},K_delta_{name}', real=True)
    K_p,K_i,K_g,K_i_q = sym.symbols(f'K_p_{name},K_i_{name},K_g_{name},K_i_q_{name}', real=True)
    R_v,X_v = sym.symbols(f'R_v_{name},X_v_{name}', real=True)
    K_q,T_q = sym.symbols(f'K_q_{name},T_q_{name}', real=True)
    K_p_v,K_i_v = sym.symbols(f'K_p_v_{name},K_i_v_{name}', real=True)
    params_list = ['F_n','K_delta','K_p','K_i','K_g','R_v','X_v','K_q','T_q','K_p_v','K_i_v']

    # auxiliar variables and constants
    omega_coi = sym.Symbol("omega_coi", real=True) # from global system
    V_s = sym.Symbol(f"V_{bus_name}", real=True) # from global system
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True) # from global system
    i_si = sym.Symbol(f"i_si_{name}", real=True)
    i_sr = sym.Symbol(f"i_sr_{name}", real=True)  
    m = sym.Symbol(f"m_{name}", real=True) 
    theta_t = sym.Symbol(f"theta_t_{name}", real=True) 

    # auxiliar equations
    Omega_b = 2*np.pi*F_n
    omega_s = omega_coi

    i_sD = i_si  # e^(-j)
    i_sQ = i_sr


    delta_ff =  delta + Ddelta_ff
    i_sd = i_sD * cos(delta_ff) - i_sQ * sin(delta_ff)   
    i_sq = i_sD * sin(delta_ff) + i_sQ * cos(delta_ff)

    p_ref = p_l + p_r
    q_ref = q_l + q_r
    Domega = x_v + K_p * (p_ref - p_s)
    e_vd = 0.0
    epsilon_v = v_ref - V_s
    omega_v = Domega + 1.0 + Domega_ff
    q_ref_0 = K_p_v * epsilon_v + K_i_v * xi_v 

    v_tD_ref =  v_td_ref * cos(delta_ff) + v_tq_ref * sin(delta_ff)   
    v_tQ_ref = -v_td_ref * sin(delta_ff) + v_tq_ref * cos(delta_ff) 

    v_ti_ref = v_tD_ref
    v_tr_ref = v_tQ_ref   
    m_ref = sym.sqrt(v_tr_ref**2 + v_ti_ref**2)/v_dc
    theta_t_ref = sym.atan2(v_ti_ref,v_tr_ref) 
    

    # dynamical equations
    ddelta   = Omega_b*(omega_v - omega_s) - K_delta*delta
    dx_v = K_i*(p_ref - p_s) - K_g*(omega_v - 1.0)
    de_qm = 1.0/T_q * (q_s - q_ref_0 - q_ref - e_qm) 
    dxi_v = epsilon_v # PI agregado

    # algebraic equations
    g_v_td_ref  = e_vd - R_v * i_sd - X_v * i_sq - v_td_ref 
    g_v_tq_ref  = e_vq - R_v * i_sq + X_v * i_sd - v_tq_ref 

    g_e_vq = 1.0 - e_vq - K_q*e_qm 
    g_m  = m-m_ref
    g_theta_t  = theta_t-theta_t_ref
    
    # DAE system update
    grid.dae['f'] += [ddelta,dx_v,de_qm,dxi_v]
    grid.dae['x'] += [ delta, x_v, e_qm, xi_v]
    grid.dae['g'] +=     [g_v_td_ref,g_v_tq_ref,g_e_vq,g_m,g_theta_t]
    grid.dae['y_ini'] += [  v_td_ref,  v_tq_ref,  e_vq,  m,  theta_t]
    grid.dae['y_run'] += [  v_td_ref,  v_tq_ref,  e_vq,  m,  theta_t]
            
    # default inputs
    grid.dae['u_ini_dict'].update({f'p_l_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_l_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'p_r_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'q_r_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'v_ref_{name}':1.0})
    grid.dae['u_ini_dict'].update({f'Ddelta_ff_{name}':0.0})
    grid.dae['u_ini_dict'].update({f'Domega_ff_{name}':0.0})

    grid.dae['u_run_dict'].update({f'p_l_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_l_{name}':0.0})
    grid.dae['u_run_dict'].update({f'p_r_{name}':0.0})
    grid.dae['u_run_dict'].update({f'q_r_{name}':0.0})
    grid.dae['u_run_dict'].update({f'v_ref_{name}':1.0})
    grid.dae['u_run_dict'].update({f'Ddelta_ff_{name}':0.0})
    grid.dae['u_run_dict'].update({f'Domega_ff_{name}':0.0})

    # default parameters
    for item in params_list:
        grid.dae['params_dict'].update({item + f'_{name}':ctrl_data[item]})

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
    grid.dae['h_dict'].update({f"i_sd_{name}":i_sd})
    grid.dae['h_dict'].update({f"i_sq_{name}":i_sq})
    grid.dae['h_dict'].update({f"delta_ff_{name}":delta_ff})
    
    grid.dae['xy_0_dict'].update({str(e_vq):1.0}) 
    grid.dae['xy_0_dict'].update({str(v_tq_ref):1.0}) 

    p_W   = p_s * S_n
    q_var = q_s * S_n

    return p_W,q_var


if __name__ == "__main__":
    pass  