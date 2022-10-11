# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def pss_kundur(dae,syn_data,name):
    '''
    PSS from Kundur's book1 

    
    '''
    pss_data = syn_data['pss']
    
    omega = sym.Symbol(f"omega_{name}", real=True)   
       
    x_wo  = sym.Symbol(f"x_wo_{name}", real=True)
    x_lead  = sym.Symbol(f"x_lead_{name}", real=True)

    z_wo  = sym.Symbol(f"z_wo_{name}", real=True)
    x_lead  = sym.Symbol(f"x_lead_{name}", real=True)
    
    T_wo = sym.Symbol(f"T_wo_{name}", real=True)  
    T_1 = sym.Symbol(f"T_1_{name}", real=True) 
    T_2 = sym.Symbol(f"T_2_{name}", real=True)
    K_stab = sym.Symbol(f"K_stab_{name}", real=True)
    V_lim = sym.Symbol(f"V_lim_{name}", real=True)
    v_pss = sym.Symbol(f"v_pss_{name}", real=True) 
    
    
    u_wo = omega - 1.0
    v_pss_nosat = K_stab*((z_wo - x_lead)*T_1/T_2 + x_lead)
    
    dx_wo =   (u_wo - x_wo)/T_wo  # washout state
    dx_lead =  (z_wo - x_lead)/T_2      # lead compensator state
    
    g_z_wo =  (u_wo - x_wo) - z_wo  
    g_v_pss = -v_pss + sym.Piecewise((-V_lim,v_pss_nosat<-V_lim),(V_lim,v_pss_nosat>V_lim),(v_pss_nosat,True))  
    
    
    dae['f'] += [dx_wo,dx_lead]
    dae['x'] += [ x_wo, x_lead]
    dae['g'] += [g_z_wo,g_v_pss]
    dae['y_ini'] += [  z_wo, v_pss]  
    dae['y_run'] += [  z_wo, v_pss] 
    dae['params_dict'].update({str(T_wo):pss_data['T_wo']})
    dae['params_dict'].update({str(T_1):pss_data['T_1']})
    dae['params_dict'].update({str(T_2):pss_data['T_2']})
    dae['params_dict'].update({str(K_stab):pss_data['K_stab']})
    dae['params_dict'].update({str(V_lim):0.1})