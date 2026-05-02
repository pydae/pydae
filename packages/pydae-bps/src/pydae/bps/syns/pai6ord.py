# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def pai6(grid,name,bus_name,data_dict):
    """
    # auxiliar

    .. math::
        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)     
        omega_s = omega_coi
                
    # dynamic equations            
    ddelta = Omega_b*(omega - omega_s) - K_delta*delta
    domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
    de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f)
    de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q)

    # algebraic equations   
    0  = v_q + R_a*i_q + X1d*i_d - e1q
    0 = v_d + R_a*i_d - X1q*i_q - e1d
    0 = i_d*v_d + i_q*v_q - p_g  
    0 = i_d*v_q - i_q*v_d - q_g 
    
    """

    backend = grid.backend
    sin = backend.sin
    cos = backend.cos

    # inputs
    V = backend.symbols(f"V_{bus_name}")
    theta = backend.symbols(f"theta_{bus_name}")
    p_m = backend.symbols(f"p_m_{name}")
    v_f = backend.symbols(f"v_f_{name}")
    omega_coi = backend.symbols("omega_coi")

    # dynamic states
    delta = backend.symbols(f"delta_{name}")
    omega = backend.symbols(f"omega_{name}")
    e1q = backend.symbols(f"e1q_{name}")
    e1d = backend.symbols(f"e1d_{name}")
    psi2d = backend.symbols(f"psi2d_{name}")
    psi2q = backend.symbols(f"psi2q_{name}")

    # algebraic states
    i_d = backend.symbols(f"i_d_{name}")
    i_q = backend.symbols(f"i_q_{name}")
    p_g = backend.symbols(f"p_g_{name}")
    q_g = backend.symbols(f"q_g_{name}")

    # parameters
    S_n = backend.symbols(f"S_n_{name}")
    Omega_b = backend.symbols(f"Omega_b_{name}")
    H = backend.symbols(f"H_{name}")
    T1d0 = backend.symbols(f"T1d0_{name}")
    T1q0 = backend.symbols(f"T1q0_{name}")
    T2d0 = backend.symbols(f"T2d0_{name}")
    T2q0 = backend.symbols(f"T2q0_{name}")
    X_l = backend.symbols(f"X_l_{name}")
    X_d = backend.symbols(f"X_d_{name}")
    X_q = backend.symbols(f"X_q_{name}")
    X1d = backend.symbols(f"X1d_{name}")
    X1q = backend.symbols(f"X1q_{name}")
    X2d = backend.symbols(f"X2d_{name}")
    X2q = backend.symbols(f"X2q_{name}")
    D = backend.symbols(f"D_{name}")
    R_a = backend.symbols(f"R_a_{name}")
    K_delta = backend.symbols(f"K_delta_{name}")
    params_list = ['S_n','Omega_b','H','T1d0','T1q0','T2d0','T2q0']
    params_list+= ['X_l','X_d','X_q','X1d','X1q','X2d','X2q','D','R_a','K_delta','K_sec']
    
    # auxiliar
    v_d = V*sin(delta - theta) 
    v_q = V*cos(delta - theta) 
    p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)     

    omega_s = omega_coi

    gamma_d1 = (X2d - X_l)/(X1d - X_l)
    gamma_q1 = (X2q - X_l)/(X1q - X_l)
    gamma_d2 = (1.0 - gamma_d1)/(X1d - X_l)
    gamma_q2 = (1.0 - gamma_q1)/(X1q - X_l)

    psi_d = -(X2d*i_d - gamma_d1*e1q - (1-gamma_d1)*psi2d)
    psi_q = -(X2q*i_q + gamma_q1*e1d - (1-gamma_q1)*psi2q)

    # dynamic equations            
    ddelta = Omega_b*(omega - omega_s) - K_delta*delta
    domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
    de1q = (-e1q - (X_d - X1d)*(i_d - gamma_d2*psi2d - (1-gamma_d1)*i_d + gamma_d2*e1q)+v_f)/T1d0
    de1d = (-e1d + (X_q - X1q)*(i_q - gamma_q2*psi2q - (1-gamma_q1)*i_q - gamma_q2*e1d))/T1q0
    dpsi2d = (-psi2d + e1q - (X1d-X_l)*i_d)/T2d0
    dpsi2q = (-psi2q - e1d - (X1q-X_l)*i_q)/T2q0

    # esto esta bien gamma_d2*e1d? se cambia a  gamma_q2*e1d


    # algebraic equations   
    g_i_q  = R_a*i_d + omega*psi_q + v_d
    g_i_d  = R_a*i_q - omega*psi_d + v_q
    g_p_g  = i_d*v_d + i_q*v_q - p_g  
    g_q_g  = i_d*v_q - i_q*v_d - q_g 
    
    # dae 
    f_syn = [ddelta,domega,de1q,de1d,dpsi2d,dpsi2q]
    x_syn = [ delta, omega, e1q, e1d, psi2d, psi2q]
    g_syn = [g_i_d,g_i_q,g_p_g,g_q_g]
    y_syn = [  i_d,  i_q,  p_g,  q_g]
    
    grid.H_total += H
    grid.omega_coi_numerator += omega*H*S_n
    grid.omega_coi_denominator += H*S_n

    grid.dae['f'] += f_syn
    grid.dae['x'] += x_syn
    grid.dae['g'] += g_syn
    grid.dae['y_ini'] += y_syn  
    grid.dae['y_run'] += y_syn  
    
    if 'v_f' in data_dict:
        grid.dae['u_ini_dict'].update({f'{v_f}':{data_dict['v_f']}})
        grid.dae['u_run_dict'].update({f'{v_f}':{data_dict['v_f']}})
    else:
        grid.dae['u_ini_dict'].update({f'{v_f}':1.0})
        grid.dae['u_run_dict'].update({f'{v_f}':1.0})

    if 'p_m' in data_dict:
        grid.dae['u_ini_dict'].update({f'{p_m}':{data_dict['p_m']}})
        grid.dae['u_run_dict'].update({f'{p_m}':{data_dict['p_m']}})
    else:
        grid.dae['u_ini_dict'].update({f'{p_m}':1.0})
        grid.dae['u_run_dict'].update({f'{p_m}':1.0})

    grid.dae['xy_0_dict'].update({str(omega):1.0})
    grid.dae['xy_0_dict'].update({str(e1q):1.0})
   
    # outputs
    grid.dae['h_dict'].update({f"p_e_{name}":p_e})

    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]})

    # for item in params_list:       
    #     grid.dae['params_dict'].update({f"{item}_{name}":syn_data[item]})
    
    # if 'avr' in syn_data:
    #     add_avr(grid.dae,syn_data)
    #     grid.dae['u_ini_dict'].pop(str(v_f))
    #     grid.dae['u_run_dict'].pop(str(v_f))
    #     grid.dae['xy_0_dict'].update({str(v_f):1.5})

    # if 'gov' in syn_data:
    #     add_gov(grid.dae,syn_data)  
    #     grid.dae['u_ini_dict'].pop(str(p_m))
    #     grid.dae['u_run_dict'].pop(str(p_m))
    #     grid.dae['xy_0_dict'].update({str(p_m):0.5})

    # if 'pss' in syn_data:
    #     add_pss(grid.dae,syn_data)  

    p_W   = p_g * S_n
    q_var = q_g * S_n

    return p_W,q_var