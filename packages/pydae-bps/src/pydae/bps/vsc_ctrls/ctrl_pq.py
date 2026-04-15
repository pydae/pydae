# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def ctrl_pq(grid,name,bus_name,data_dict):
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

    sin = sym.sin
    cos = sym.cos

    ctrl_data = data_dict['ctrl']

    # inputs
    V_s = sym.Symbol(f"V_{bus_name}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True)
    omega_coi = sym.Symbol("omega_coi", real=True)   
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)   
    p_s_ref = sym.Symbol(f"p_s_ref_{name}", real=True)
    q_s_ref = sym.Symbol(f"q_s_ref_{name}", real=True)      
   
    # dynamic states
    delta = sym.Symbol(f"delta_{name}", real=True)

    # algebraic states
    i_sd_ref = sym.Symbol(f"i_sd_ref_{name}", real=True)
    i_sq_ref = sym.Symbol(f"i_sq_ref_{name}", real=True)   
    v_td_ref = sym.Symbol(f"v_td_ref_{name}", real=True)
    v_tq_ref = sym.Symbol(f"v_tq_ref_{name}", real=True)    
    m = sym.Symbol(f"m_{name}", real=True) 
    theta_t = sym.Symbol(f"theta_t_{name}", real=True) 

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    F_n = sym.Symbol(f"F_n_{name}", real=True)            
    X_s = sym.Symbol(f"X_s_{name}", real=True)
    R_s = sym.Symbol(f"R_s_{name}", real=True)
    
    params_list = []
    
    # auxiliar
    delta = theta_s
    v_sD = V_s*sin(theta_s)  # v_si   e^(-j)
    v_sQ = V_s*cos(theta_s)  # v_sr
    v_sd = v_sD * cos(delta) - v_sQ * sin(delta)   
    v_sq = v_sD * sin(delta) + v_sQ * cos(delta)
    
    Omega_b = 2*np.pi*F_n
    omega_s = omega_coi
    v_tD_ref = v_td_ref * cos(delta) + v_tq_ref * sin(delta)   
    v_tQ_ref =-v_td_ref * sin(delta) + v_tq_ref * cos(delta)    
    v_ti_ref =  v_tD_ref
    v_tr_ref =  v_tQ_ref   
    m_ref = sym.sqrt(v_tr_ref**2 + v_ti_ref**2)/v_dc
    theta_t_ref = sym.atan2(v_ti_ref,v_tr_ref) 

    # dynamic equations            

   
    # algebraic equations   
    g_i_sd_ref  = i_sd_ref*v_sd + i_sq_ref*v_sq - p_s_ref  
    g_i_sq_ref  = i_sq_ref*v_sd - i_sd_ref*v_sq - q_s_ref
    g_v_td_ref  = v_td_ref - R_s*i_sd_ref - X_s*i_sq_ref - v_sd  
    g_v_tq_ref  = v_tq_ref - R_s*i_sq_ref + X_s*i_sd_ref - v_sq 
    g_m  = m-m_ref
    g_theta_t  = theta_t-theta_t_ref
   
    # dae    
    grid.dae['f'] += []
    grid.dae['x'] += []
    grid.dae['g'] += [g_i_sd_ref,g_i_sq_ref,g_v_td_ref,g_v_tq_ref,g_m,g_theta_t]
    grid.dae['y_ini'] += [  i_sd_ref,  i_sq_ref,  v_td_ref,  v_tq_ref,  m,  theta_t]  
    grid.dae['y_run'] += [  i_sd_ref,  i_sq_ref,  v_td_ref,  v_tq_ref,  m,  theta_t]  
    
    v_sq_ref_0 = 1.0
    p_s_ref_0 = ctrl_data['p_s_ref']
    q_s_ref_0 = ctrl_data['q_s_ref']
    i_sd_ref_0 = p_s_ref_0/v_sq_ref_0
    i_sq_ref_0 = q_s_ref_0/v_sq_ref_0

    grid.dae['u_ini_dict'].update({f'{str(p_s_ref)}':p_s_ref_0})
    grid.dae['u_run_dict'].update({f'{str(p_s_ref)}':p_s_ref_0})

    grid.dae['u_ini_dict'].update({f'{str(q_s_ref)}':q_s_ref_0})
    grid.dae['u_run_dict'].update({f'{str(q_s_ref)}':q_s_ref_0})

    grid.dae['u_ini_dict'].update({f'{str(i_sd_ref)}':i_sd_ref_0})
    grid.dae['u_run_dict'].update({f'{str(i_sd_ref)}':i_sd_ref_0})

    grid.dae['u_ini_dict'].update({f'{str(i_sq_ref)}':i_sq_ref_0})
    grid.dae['u_run_dict'].update({f'{str(i_sq_ref)}':i_sq_ref_0})

    grid.dae['xy_0_dict'].update({str(v_tq_ref):1.0})

    

      
    # outputs
    grid.dae['h_dict'].update({f"m_ref_{name}":m_ref})
    grid.dae['h_dict'].update({f"theta_t_{name}":theta_t})
    
    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":ctrl_data[item]}) 

    return m,theta_t