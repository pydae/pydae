# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def olives_vsg(grid,name,bus_name,data_dict):
    """
 
    """

    sin = sym.sin
    cos = sym.cos
    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]
                
    bus_name = data_dict['bus']
    
    if 'name' in data_dict:
        name = data_dict['name']
    else:
        name = bus_name
        
    for gen_id in range(100):
        if name not in grid.generators_id_list:
            grid.generators_id_list += [name]
            break
        else:
            name = name + f'_{gen_id}'
            
    data_dict['name'] = name
                        
    idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
    if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
    buses[idx_bus]['idx_powers'] += 1
    
    # inputs
    V = sym.Symbol(f"V_{bus_name}", real=True)
    theta = sym.Symbol(f"theta_{bus_name}", real=True)
    p_m = sym.Symbol(f"p_m_{name}", real=True)
    v_ref = sym.Symbol(f"v_ref_{name}", real=True)  
    omega_coi = sym.Symbol("omega_coi", real=True)   
    p_c = sym.Symbol(f"p_c_{name}", real=True)
    p_agc = sym.Symbol("p_agc", real=True)
    omega_ref = sym.Symbol(f"omega_ref_{name}", real=True)
    q_ref = sym.Symbol(f"q_ref_{name}", real=True) 
        
    # dynamic states
    delta = sym.Symbol(f"delta_{name}", real=True)
    e_qv = sym.Symbol(f"e_qv_{name}", real=True)
    xi_p = sym.Symbol(f"xi_p_{name}", real=True)
    p_ef = sym.Symbol(f"p_ef_{name}", real=True)
    p_cf = sym.Symbol(f"p_cf_{name}", real=True)

    # algebraic states
    omega = sym.Symbol(f"omega_{name}", real=True)
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)            
    p_g = sym.Symbol(f"p_g_{name}", real=True)
    q_g = sym.Symbol(f"q_g_{name}", real=True)


    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
    K_p = sym.Symbol(f"K_p_{name}", real=True)    
    K_q = sym.Symbol(f"K_q_{name}", real=True)   
    T_p = sym.Symbol(f"T_p_{name}", real=True)
    X_v = sym.Symbol(f"X_v_{name}", real=True)
    T_v = sym.Symbol(f"T_v_{name}", real=True)
    X_v = sym.Symbol(f"X_v_{name}", real=True)
    R_v = sym.Symbol(f"R_v_{name}", real=True)
    K_delta = sym.Symbol(f"K_delta_{name}", real=True)
    Droop = sym.Symbol(f"Droop_{name}", real=True) 
    K_sec = sym.Symbol(f"K_sec_{name}", real=True) 
    T_e  = sym.Symbol(f"T_e_{name}", real=True)  
    T_c  = sym.Symbol(f"T_c_{name}", real=True)  

    params_list = ['S_n','Omega_b','K_p','T_p','K_q','T_v','X_v','R_v','K_delta','K_sec','Droop','K_sec']
    
    # auxiliar
    v_d = V*sin(delta - theta) 
    v_q = V*cos(delta - theta) 
    p_e = i_d*(v_d + R_v*i_d) + i_q*(v_q + R_v*i_q)     
    omega_s = omega_coi
    e_dv = 0
    p_r = K_sec*p_agc
    epsilon_p = p_m - p_ef
    epsilon_q = q_ref - q_g
    
                        
    
    # dynamic equations            
    ddelta = Omega_b*(omega - omega_s) - K_delta*delta
    dxi_p = epsilon_p
    de_qv = 1/T_v*(v_ref + K_q*epsilon_q - e_qv)
    dp_ef = 1/T_e*(p_e - p_ef)
    dp_cf = 1/T_c*(p_c - p_cf)

    # algebraic equations   
    g_omega = -omega + K_p*(epsilon_p + xi_p/T_p) + 1
    g_i_d  = -R_v*i_d + X_v*i_q - v_d + e_dv 
    g_i_q  = -R_v*i_q - X_v*i_d - v_q + e_qv
    g_p_g  = i_d*v_d + i_q*v_q - p_g  
    g_q_g  = i_d*v_q - i_q*v_d - q_g 
    g_p_m  = -p_m + p_cf + p_r - 1/Droop*(omega - omega_ref)
    
    # dae 
    f_vsg = [ddelta,dxi_p,de_qv,dp_ef,dp_cf]
    x_vsg = [ delta, xi_p, e_qv, p_ef, p_cf]
    g_vsg = [g_omega,g_i_d,g_i_q,g_p_g,g_q_g,g_p_m]
    y_vsg = [  omega,  i_d,  i_q,  p_g,  q_g,  p_m]
    
    # T_p = K_p*2*H
    H = T_p/(2*K_p)
    grid.H_total += H
    grid.omega_coi_numerator += omega*H*S_n
    grid.omega_coi_denominator += H*S_n

    grid.dae['f'] += f_vsg
    grid.dae['x'] += x_vsg
    grid.dae['g'] += g_vsg
    grid.dae['y_ini'] += y_vsg  
    grid.dae['y_run'] += y_vsg  
    
    if 'v_ref' in data_dict:
        grid.dae['u_ini_dict'].update({f'{str(v_ref)}':data_dict['v_ref']})
        grid.dae['u_run_dict'].update({f'{str(v_ref)}':data_dict['v_ref']})
    else:
        grid.dae['u_ini_dict'].update({f'{str(v_ref)}':1.0})
        grid.dae['u_run_dict'].update({f'{str(v_ref)}':1.0})

    #if 'p_m' in data_dict:
    #    grid.dae['u_ini_dict'].update({f'{str(p_m)}':data_dict['p_m']})
    #    grid.dae['u_run_dict'].update({f'{str(p_m)}':data_dict['p_m']})
    #else:
    #    grid.dae['u_ini_dict'].update({f'{str(p_m)}':1.0})
    #    grid.dae['u_run_dict'].update({f'{str(p_m)}':1.0})
        
    grid.dae['u_ini_dict'].update({f'{str(p_c)}':data_dict['p_c']})
    grid.dae['u_run_dict'].update({f'{str(p_c)}':data_dict['p_c']})

    grid.dae['u_ini_dict'].update({f'{str(omega_ref)}':1.0})
    grid.dae['u_run_dict'].update({f'{str(omega_ref)}':1.0})

    grid.dae['u_ini_dict'].update({f'{str(q_ref)}':0.0})
    grid.dae['u_run_dict'].update({f'{str(q_ref)}':0.0})
    

    grid.dae['xy_0_dict'].update({str(omega):1.0})
    
    # outputs
    grid.dae['h_dict'].update({f"p_e_{name}":p_e})
    
    # 
    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]})

    if 'T_e' in data_dict:
        grid.dae['params_dict'].update({f'{str(T_e)}':data_dict['T_e']})
    else:
        grid.dae['params_dict'].update({f'{str(T_e)}':0.1})

    if 'T_c' in data_dict:
        grid.dae['params_dict'].update({f'{str(T_c)}':data_dict['T_c']})
    else:
        grid.dae['params_dict'].update({f'{str(T_c)}':0.1})


    grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]})

    p_W   = p_g * S_n
    q_var = q_g * S_n

    return p_W,q_var

