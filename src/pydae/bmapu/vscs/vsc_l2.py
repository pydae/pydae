# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def vsc_l(grid,name,bus_name,data_dict):
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

    "vscs": [{"type":"vsc_l","S_n":1e6,"F_n":50.0,"X_s":0.1,"R_s":0.01}]
    
    '''

    sin = sym.sin
    cos = sym.cos

    # inputs
    V_s = sym.Symbol(f"V_{bus_name}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True)
    m = sym.Symbol(f"m_{name}", real=True)
    theta_t = sym.Symbol(f"theta_t_{name}", real=True)
    v_dc = sym.Symbol(f"v_dc_{name}", real=True)
      
    # dynamic states
    m_f = sym.Symbol(f"m_f_{name}", real=True)

    # algebraic states
    i_si = sym.Symbol(f"i_si_{name}", real=True)
    i_sr = sym.Symbol(f"i_sr_{name}", real=True)       
    p_s = sym.Symbol(f"p_s_{name}", real=True)
    q_s = sym.Symbol(f"q_s_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    F_n = sym.Symbol(f"F_n_{name}", real=True)            
    X_s = sym.Symbol(f"X_s_{name}", real=True)
    R_s = sym.Symbol(f"R_s_{name}", real=True)
    
    params_list = ['S_n','F_n','X_s','R_s']
    
    # auxiliar
    v_si = V_s*sin(theta_s)  # v_D, e^(-j)
    v_sr = V_s*cos(theta_s)  # v_Q
    Omega_b = 2*np.pi*F_n
    v_t_m = m*v_dc/np.sqrt(6)
    v_tr = v_t_m*cos(theta_t)
    v_ti = v_t_m*sin(theta_t)
    
    # dynamic equations            

    # algebraic equations   
    g_i_si = v_ti - R_s*i_si + X_s*i_sr - v_si  
    g_i_sr = v_tr - R_s*i_sr - X_s*i_si - v_sr 
    g_p_s  = i_si*v_si + i_sr*v_sr - p_s  
    g_q_s  = i_si*v_sr - i_sr*v_si - q_s 

    # dae 
    f_vsg = [(m - m_f)]
    x_vsg = [m_f]
    g_vsg = [g_i_si,g_i_sr,g_p_s,g_q_s]
    y_vsg = [  i_si,  i_sr,  p_s,  q_s]

    grid.dae['f'] += f_vsg
    grid.dae['x'] += x_vsg
    grid.dae['g'] += g_vsg
    grid.dae['y_ini'] += y_vsg  
    grid.dae['y_run'] += y_vsg  
    
    grid.dae['u_ini_dict'].update({f'{m}':1.0})
    grid.dae['u_run_dict'].update({f'{m}':1.0})

    grid.dae['u_ini_dict'].update({f'{theta_t}':0.0})
    grid.dae['u_run_dict'].update({f'{theta_t}':0.0})

    grid.dae['u_ini_dict'].update({f'{v_dc}':1.2})
    grid.dae['u_run_dict'].update({f'{v_dc}':1.2})

    grid.dae['xy_0_dict'].update({str(p_s):0.5})
       
    # outputs
    grid.dae['h_dict'].update({f"{str(p_s)}":p_s})
    grid.dae['h_dict'].update({f"{str(q_s)}":q_s})
    grid.dae['h_dict'].update({f"{str(v_ti)}":v_ti})
    grid.dae['h_dict'].update({f"{str(v_tr)}":v_tr})    
    grid.dae['h_dict'].update({f"{str(i_si)}":i_si})
    grid.dae['h_dict'].update({f"{str(i_sr)}":i_sr})

    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]}) 

    p_W   = p_s * S_n
    q_var = q_s * S_n

    return p_W,q_var