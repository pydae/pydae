# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def tgov1(dae,syn_data,name,bus_name):
    '''
    Governor TGOV1 like in PSS/E
    
    '''
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    # inpunts
    omega = sym.Symbol(f"omega_{name}", real=True)
    p_agc = sym.Symbol(f"p_agc", real=True)
    p_c = sym.Symbol(f"p_c_{name}", real=True) 
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)    
    R_a = sym.Symbol(f"R_a_{name}", real=True)  

    # dynamic states
    x_gov_1 = sym.Symbol(f"x_gov_1_{name}", real=True)
    x_gov_2 = sym.Symbol(f"x_gov_2_{name}", real=True)  
    p_g_f   = sym.Symbol(f"p_g_f_{name}", real=True)  
    xi_gov = sym.Symbol(f"xi_gov_{name}", real=True)  

    # algebraic states
    p_m = sym.Symbol(f"p_m_{name}", real=True)  
    p_m_ref = sym.Symbol(f"p_m_ref_{name}", real=True)  

    # parameters
    T_1 = sym.Symbol(f"T_gov_1_{name}", real=True)  # 1
    T_2 = sym.Symbol(f"T_gov_2_{name}", real=True)  # 2
    T_3 = sym.Symbol(f"T_gov_3_{name}", real=True)  # 10
    D_t = sym.Symbol(f"D_t_{name}", real=True)  # 10 
    Droop = sym.Symbol(f"Droop_{name}", real=True)  # 0.05
    K_sec = sym.Symbol(f"K_sec_{name}", real=True)  # 0.05
    K_ci = sym.Symbol(f"K_ci_gov_{name}", real=True)  # 0.05
    losses_p = R_a*(i_d**2 + i_q**2)
     
    omega_ref = sym.Symbol(f"omega_ref_{name}", real=True)

    # auxiliar
    Domega = (omega - omega_ref)
    p_r = K_sec*p_agc

    # differential equations
    dx_gov_1 =   (p_m_ref - x_gov_1)/T_1
    dx_gov_2 =   (x_gov_1 - x_gov_2)/T_3
    # dp_g_f = 1.0/0.1*(p_g - p_g_f)
    # dxi_gov = p_c - p_g_f - 1e-6*xi_gov

    g_p_m_ref  = -p_m_ref + p_c + p_r - 1/Droop*Domega + K_ci*losses_p
    g_p_m = (x_gov_1 - x_gov_2)*T_2/T_3 + x_gov_2 - D_t*Domega - p_m

    
    dae['f'] += [dx_gov_1,dx_gov_2]
    dae['x'] += [ x_gov_1, x_gov_2]
    dae['g'] += [g_p_m_ref,g_p_m]
    dae['y_ini'] += [  p_m_ref,  p_m]  
    dae['y_run'] += [  p_m_ref,  p_m]  

    dae['params_dict'].update({str(Droop):gov_data['Droop']})
    dae['params_dict'].update({str(T_1):gov_data['T_1']})
    dae['params_dict'].update({str(T_2):gov_data['T_2']})
    dae['params_dict'].update({str(T_3):gov_data['T_3']})
    dae['params_dict'].update({str(D_t):gov_data['D_t']})
    dae['params_dict'].update({str(K_sec):gov_data['K_sec']})

    dae['params_dict'].update({str(omega_ref):1.0})
    dae['params_dict'].update({str(K_ci):1.0e-6})

    dae['u_ini_dict'].update({str(p_c):gov_data['p_c']})
    dae['u_run_dict'].update({str(p_c):gov_data['p_c']})

    dae['h_dict'].update({str(p_c):gov_data['p_c']})