# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def tgov1(dae,syn_data,name):
    '''
    Governor TGOV1 like in PSS/E
    
    '''
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    # inpunts
    omega = sym.Symbol(f"omega_{bus_name}", real=True)
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    p_c = sym.Symbol(f"p_c_{bus_name}", real=True) 

    # dynamic states
    x_gov_1 = sym.Symbol(f"x_gov_1_{bus_name}", real=True)
    x_gov_2 = sym.Symbol(f"x_gov_2_{bus_name}", real=True)  

    # algebraic states
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)  
    p_m_ref = sym.Symbol(f"p_m_ref_{bus_name}", real=True)  

    # parameters
    T_1 = sym.Symbol(f"T_gov_1_{bus_name}", real=True)  # 1
    T_2 = sym.Symbol(f"T_gov_2_{bus_name}", real=True)  # 2
    T_3 = sym.Symbol(f"T_gov_3_{bus_name}", real=True)  # 10

    Droop = sym.Symbol(f"Droop_{bus_name}", real=True)  # 0.05
    
    omega_ref = sym.Symbol(f"omega_ref_{bus_name}", real=True)

    # differential equations
    dx_gov_1 =   (p_m_ref - x_gov_1)/T_1
    dx_gov_2 =   (x_gov_1 - x_gov_2)/T_3

    g_p_m_ref  = -p_m_ref + p_c + p_r - 1/Droop*(omega - omega_ref)
    g_p_m = (x_gov_1 - x_gov_2)*T_2/T_3 + x_gov_2 - p_m

    
    dae['f'] += [dx_gov_1,dx_gov_2]
    dae['x'] += [ x_gov_1, x_gov_2]
    dae['g'] += [g_p_m_ref,g_p_m]
    dae['y'] += [  p_m_ref,  p_m]  
    dae['params'].update({str(Droop):gov_data['Droop']})
    dae['params'].update({str(T_1):gov_data['T_1']})
    dae['params'].update({str(T_2):gov_data['T_2']})
    dae['params'].update({str(T_3):gov_data['T_3']})
    dae['params'].update({str(omega_ref):1.0})

    dae['u'].update({str(p_c):gov_data['p_c']})