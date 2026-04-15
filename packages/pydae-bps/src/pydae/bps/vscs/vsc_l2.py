# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def vsc_l(grid,name,bus_name,data_dict):
    '''

    VSC model with L filter coupling and purely algebraic.
    No control is implemented. 

    parameters
    ----------

    S_n: nominal power in VA
    U_n: nominal rms phase to phase voltage in V
    F_n: nominal frequency in Hz
    X_s: coupling reactance in pu (base machine S_n)
    R_s: coupling resistance in pu (base machine S_n)

    inputs
    ------

    v_dc: dc voltage in pu (when v_dc = 1 and m = 1, v_ac = 1)
    m: modulation index (-)
    theta_t: absolute terminal voltage phase angle (rad)

    example
    -------

    "vscs": [{"type":"vsc_l","S_n":1e6,"U_n":400.0,"F_n":50.0,"X_s":0.1,"R_s":0.01,"monitor":True}]
    
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
    #m_f = sym.Symbol(f"m_f_{name}", real=True)

    # algebraic states
    i_si = sym.Symbol(f"i_si_{name}", real=True)
    i_sr = sym.Symbol(f"i_sr_{name}", real=True)       
    p_s = sym.Symbol(f"p_s_{name}", real=True)
    q_s = sym.Symbol(f"q_s_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    U_n = sym.Symbol(f"U_n_{name}", real=True)
    F_n = sym.Symbol(f"F_n_{name}", real=True)            
    X_s = sym.Symbol(f"X_s_{name}", real=True)
    R_s = sym.Symbol(f"R_s_{name}", real=True)
    
    params_list = ['S_n','F_n','U_n','X_s','R_s']
    
    # auxiliar
    v_si = V_s*sin(theta_s)  # v_D, e^(-j)
    v_sr = V_s*cos(theta_s)  # v_Q
    Omega_b = 2*np.pi*F_n
    v_t_m = m*v_dc
    v_tr = v_t_m*cos(theta_t)
    v_ti = v_t_m*sin(theta_t)
    
    # dynamic equations            

    # algebraic equations   
    g_i_si = v_ti - R_s*i_si + X_s*i_sr - v_si  
    g_i_sr = v_tr - R_s*i_sr - X_s*i_si - v_sr 
    g_p_s  = i_si*v_si + i_sr*v_sr - p_s  
    g_q_s  = i_si*v_sr - i_sr*v_si - q_s 

    # dae 
    f_vsg = []
    x_vsg = []
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

    if 'monitor' in data_dict:
        if data_dict['monitor'] == True:
            V_dc_b = U_n*np.sqrt(2)
            grid.dae['h_dict'].update({f"v_dc_v_{name}":v_dc*V_dc_b})
            grid.dae['h_dict'].update({f"v_ac_v_{name}":v_t_m*U_n})
            

    p_W   = p_s * S_n
    q_var = q_s * S_n

    return p_W,q_var