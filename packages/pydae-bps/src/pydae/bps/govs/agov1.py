# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def agov1(dae,syn_data,name,bus_name):
    '''
    Governor AGOV1 
    
    '''
    gov_data = syn_data['gov']

    
    # inpunts
    omega = sym.Symbol(f"omega_{name}", real=True)
    p_c = sym.Symbol(f"p_c_{name}", real=True) 
    p_g = sym.Symbol(f"p_g_{name}", real=True)
    p_agc = sym.Symbol("p_agc", real=True)
    p_r = sym.Symbol(f"p_r_{name}", real=True)
    
    # dynamic states
    x_gov_1 = sym.Symbol(f"x_gov_1_{name}", real=True)
    x_gov_2 = sym.Symbol(f"x_gov_2_{name}", real=True)  
    xi_imw  = sym.Symbol(f"xi_imw_{name}", real=True)  

    # algebraic states
    p_m = sym.Symbol(f"p_m_{name}", real=True)  
    p_m_ref = sym.Symbol(f"p_m_ref_{name}", real=True)  

    # parameters
    T_1 = sym.Symbol(f"T_gov_1_{name}", real=True)  # 1
    T_2 = sym.Symbol(f"T_gov_2_{name}", real=True)  # 2
    T_3 = sym.Symbol(f"T_gov_3_{name}", real=True)  # 10
    Droop = sym.Symbol(f"Droop_{name}", real=True)  # 0.05
    K_imw = sym.Symbol(f"K_imw_{name}", real=True)  # 0.0
    K_sec = sym.Symbol(f"K_sec_{name}", real=True)  # 0.0    
    omega_ref = sym.Symbol(f"omega_ref_{name}", real=True)
    
    # auxiliar

    # differential equations
    dx_gov_1 =   (p_m_ref - x_gov_1)/T_1
    dx_gov_2 =   (x_gov_1 - x_gov_2)/T_3
    dxi_imw =   K_imw*(p_c - p_g) - 1e-6*xi_imw

    g_p_m_ref  = -p_m_ref + xi_imw + p_r + K_sec*p_agc - 1/Droop*(omega - omega_ref) 
    g_p_m = (x_gov_1 - x_gov_2)*T_2/T_3 + x_gov_2 - p_m

    
    dae['f'] += [dx_gov_1,dx_gov_2,dxi_imw]
    dae['x'] += [ x_gov_1, x_gov_2, xi_imw]
    dae['g'] += [g_p_m_ref,g_p_m]
    dae['y_ini'] += [  p_m_ref,  p_m]  
    dae['y_run'] += [  p_m_ref,  p_m] 
    dae['params_dict'].update({str(Droop):gov_data['Droop']})
    dae['params_dict'].update({str(T_1):gov_data['T_1']})
    dae['params_dict'].update({str(T_2):gov_data['T_2']})
    dae['params_dict'].update({str(T_3):gov_data['T_3']})
    dae['params_dict'].update({str(K_imw):gov_data['K_imw']})
    dae['params_dict'].update({str(K_sec):syn_data['K_sec']})
    dae['params_dict'].update({str(omega_ref):1.0})

    dae['u_ini_dict'].update({str(p_c):gov_data['p_c']})
    dae['u_run_dict'].update({str(p_c):gov_data['p_c']})
    dae['u_ini_dict'].update({str(p_r):0.0})
    dae['u_run_dict'].update({str(p_r):0.0})