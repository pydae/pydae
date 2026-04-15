# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def pss_kundur_2(dae,syn_data,name):
    '''
    PSS from Kundur's book1 

    Figure E12.9  

    "pss":{"type":"pss_kundur_2","K_stab":20, "T_1":0.05, "T_2":0.02, "T_3":3.0, "T_4":5.4, "T_w":10.0},

    '''
    pss_data = syn_data['pss']
    
    omega = sym.Symbol(f"omega_{name}", real=True)   
       
    x_wo  = sym.Symbol(f"x_wo_{name}", real=True)
    x_lead  = sym.Symbol(f"x_lead_{name}", real=True)

    z_wo  = sym.Symbol(f"z_wo_{name}", real=True)
    x_12  = sym.Symbol(f"x_12_{name}", real=True)
    x_34  = sym.Symbol(f"x_34_{name}", real=True)  

    T_wo = sym.Symbol(f"T_wo_{name}", real=True)  
    T_1 = sym.Symbol(f"T_1_{name}", real=True) 
    T_2 = sym.Symbol(f"T_2_{name}", real=True)
    T_3 = sym.Symbol(f"T_3_{name}", real=True) 
    T_4 = sym.Symbol(f"T_4_{name}", real=True)
    K_stab = sym.Symbol(f"K_stab_{name}", real=True)
    V_lim = sym.Symbol(f"V_lim_{name}", real=True)
    v_s = sym.Symbol(f"v_pss_{name}", real=True) 
    
    
    u_wo = K_stab*(omega - 1.0)
    v_pss_nosat = K_stab*((z_wo - x_lead)*T_1/T_2 + x_lead)
    
    z_wo = u_wo - x_wo
    z_12 = (z_wo - x_12)*T_1/T_2 + x_12
    z_34 = (z_12 - x_34)*T_3/T_4 + x_34

    v_s_nosat = z_34

    dx_wo =  (u_wo - x_wo)/T_wo  # washout state
    dx_12 =  (z_wo - x_12)/T_2      # lead compensator state
    dx_34 =  (z_12 - x_34)/T_4      # lead compensator state

    g_v_s = -v_s + sym.Piecewise((-V_lim,v_s_nosat<-V_lim),(V_lim,v_s_nosat>V_lim),(v_s_nosat,True))  
    
    
    dae['f'] += [dx_wo,dx_12,dx_34]
    dae['x'] += [ x_wo, x_12, x_34]
    dae['g'] += [g_v_s]
    dae['y_ini'] += [v_s]  
    dae['y_run'] += [v_s] 
    dae['params_dict'].update({str(T_wo):pss_data['T_w']})
    dae['params_dict'].update({str(T_1):pss_data['T_1']})
    dae['params_dict'].update({str(T_2):pss_data['T_2']})
    dae['params_dict'].update({str(T_3):pss_data['T_3']})
    dae['params_dict'].update({str(T_4):pss_data['T_4']})
    dae['params_dict'].update({str(K_stab):pss_data['K_stab']})
    dae['params_dict'].update({str(V_lim):0.1})
