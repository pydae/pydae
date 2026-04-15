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

    v_t = m*v_dc/sqrt(6)

    v_t_pu*v_b_ac = m*v_dc_pu*v_b_dc/sqrt(6)

    v_dc_b = sqrt(2)*v_ac_b
    v_t_pu*v_b_ac = m*v_dc_pu*sqrt(2)*v_ac_b/sqrt(6)
    v_t_pu = m*v_dc_pu*sqrt(2)/sqrt(6) = m*v_dc_pu*sqrt(3)
    
    
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
    p_dc = sym.Symbol(f"p_dc_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    F_n = sym.Symbol(f"F_n_{name}", real=True)            
    X_s = sym.Symbol(f"X_s_{name}", real=True)
    R_s = sym.Symbol(f"R_s_{name}", real=True)
    A_l = sym.Symbol(f"A_l_{name}", real=True)
    B_l = sym.Symbol(f"B_l_{name}", real=True)
    C_l = sym.Symbol(f"C_l_{name}", real=True)

    params_list = ['S_n','F_n','X_s','R_s']
    
    # auxiliar
    v_sr =  V_s*cos(theta_s)  # v_Q
    v_si =  V_s*sin(theta_s)  # v_D, e^(-j)
    Omega_b = 2*np.pi*F_n
    v_t_m = m*v_dc
    v_tr =  v_t_m*cos(theta_t)
    v_ti =  v_t_m*sin(theta_t)
    i_m = sym.sqrt(i_sr**2 + i_si**2)
    p_loss = A_l + B_l*i_m + C_l*i_m**2
    p_ac = p_s + R_s*i_m**2
    i_d = p_dc/v_dc

    # dynamic equations            

    # algebraic equations   
    g_i_si = v_ti - R_s*i_si - X_s*i_sr - v_si  
    g_i_sr = v_tr - R_s*i_sr + X_s*i_si - v_sr 
    g_p_s  =  i_sr*v_sr + i_si*v_si - p_s  
    g_q_s  =  i_sr*v_si - i_si*v_sr - q_s 
    g_p_dc = -p_dc + p_ac + p_loss

    # dae 
    f_vsg = [(m - m_f)]
    x_vsg = [m_f]
    g_vsg = [g_i_si,g_i_sr,g_p_s,g_q_s,g_p_dc]
    y_vsg = [  i_si,  i_sr,  p_s,  q_s,  p_dc]

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

    p_s_0 = 0.5
    grid.dae['xy_0_dict'].update({str(p_s):p_s_0})
    grid.dae['xy_0_dict'].update({str(i_sr):0.1})

    grid.dae['xy_0_dict'].update({str(p_dc):0.0001})

    # outputs
    grid.dae['h_dict'].update({f"{str(p_s)}":p_s})
    grid.dae['h_dict'].update({f"{str(q_s)}":q_s}) 
    grid.dae['h_dict'].update({f"{str(i_si)}":i_si})
    grid.dae['h_dict'].update({f"{str(i_sr)}":i_sr})
    grid.dae['h_dict'].update({f"{str(m_f)}":m_f})
    grid.dae['h_dict'].update({f"p_ac_{name}":p_ac})
    grid.dae['h_dict'].update({f"p_dc_{name}":p_s_0})
    grid.dae['h_dict'].update({f"i_d_{name}":i_d})

    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]}) 

    if 'A_l' in data_dict:
        grid.dae['params_dict'].update({f"A_l_{name}":data_dict['A_l']}) 
        grid.dae['params_dict'].update({f"B_l_{name}":data_dict['B_l']}) 
        grid.dae['params_dict'].update({f"C_l_{name}":data_dict['C_l']}) 
    else:
        grid.dae['params_dict'].update({f"A_l_{name}":0.005}) 
        grid.dae['params_dict'].update({f"B_l_{name}":0.005}) 
        grid.dae['params_dict'].update({f"C_l_{name}":0.005})         


    p_W   = p_s * S_n
    q_var = q_s * S_n

    return p_W,q_var