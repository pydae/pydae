# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def pss_kundur(dae,syn_data,name,bus_name):
    '''
    PSS as IEEE PSS2
    'K_s3':1.0,'T_wo1':2.0,'T_wo2':2.0,'T_9': 0.1,'K_s1':17.069,'T_1': 0.28,'T_2':0.04,'T_3':0.28,'T_4':0.12,'T_wo3':2.0,'K_s2': 0.158,'T_7':2.0

    '''
    pss_data = syn_data['pss']
    

    omega = sym.Symbol(f"omega_{name}", real=True)   
    p_g   = sym.Symbol(f"p_g_{name}", real=True) 

    # states
    x_wo1 = sym.Symbol(f"x_wo1 _{name}", real=True)
    x_wo2 = sym.Symbol(f"x_wo2 _{name}", real=True)
    x_wo3 = sym.Symbol(f"x_wo3 _{name}", real=True)
    x_lpf7= sym.Symbol(f"x_lpf7_{name}", real=True)
    x_9_1 = sym.Symbol(f"x_9_1 _{name}", real=True)
    x_9_2 = sym.Symbol(f"x_9_2 _{name}", real=True)
    x_9_3 = sym.Symbol(f"x_9_3 _{name}", real=True)
    x_9_4 = sym.Symbol(f"x_9_4 _{name}", real=True)
    x_9_5 = sym.Symbol(f"x_9_5 _{name}", real=True)
    x_ll1 = sym.Symbol(f"x_ll1 _{name}", real=True)
    x_ll3 = sym.Symbol(f"x_ll3 _{name}", real=True)

 
    v_pss = sym.Symbol(f"v_pss_{name}", real=True) 
    
    T_1   = sym.Symbol(f"T_pss_1_{name}", real=True)    
    T_2   = sym.Symbol(f"T_pss_2_{name}", real=True)    
    T_3   = sym.Symbol(f"T_pss_3_{name}", real=True)    
    T_4   = sym.Symbol(f"T_pss_4_{name}", real=True) 
    T_7   = sym.Symbol(f"T_pss_7_{name}", real=True)   
    T_9   = sym.Symbol(f"T_pss_9_{name}", real=True) 

    T_wo1 = sym.Symbol(f"T_wo1_{name}", real=True)  
    T_wo2 = sym.Symbol(f"T_wo2_{name}", real=True)  
    T_wo3 = sym.Symbol(f"T_wo3_{name}", real=True) 

    K_s1  = sym.Symbol(f"K_s1_{name}", real=True)   
    K_s2  = sym.Symbol(f"K_s2_{name}", real=True)   
    K_s3  = sym.Symbol(f"K_s3_{name}", real=True) 

    u_1 = omega
    u_2 = p_g

    y_9 = x_9_5 
    z_wo1 = u_1 - x_wo1 
    z_wo2 = z_wo1 - x_wo2 

    y_3 = x_lpf7 
    u_9 = K_s3*y_3 + z_wo2 
    u_ll = K_s1*(y_9 - y_3) 
    z_wo3 = u_2 - x_wo3 

    dx_wo1 =   (u_1 - x_wo1)/T_wo1
    dx_wo2 =   (z_wo1 - x_wo2)/T_wo2 
    dx_wo3 =   (u_2 - x_wo3)/T_wo3 
    dx_lpf7 =   (K_s2*z_wo3 - x_lpf7)/T_7 
    dx_9_1 = 1/T_9*(u_9 - x_9_1) 
    dx_9_2 = 1/T_9*(x_9_1 - x_9_2)
    dx_9_3 = 1/T_9*(x_9_2 - x_9_3)
    dx_9_4 = 1/T_9*(x_9_3 - x_9_4)
    dx_9_5 = 1/T_9*(x_9_4 - x_9_5)
    dx_ll1 =( u_ll - x_ll1)/T_2;  
    z_ll1 = ( u_ll - x_ll1)*T_1/T_2 + x_ll1 
    dx_ll3 =(z_ll1 - x_ll3)/T_4;   
    z_ll3 = (z_ll1 - x_ll3)*T_3/T_4 + x_ll3

    g_v_pss = -v_pss + z_ll3  
    
    dae['f'] += [dx_wo1,dx_wo2,dx_wo3,dx_lpf7,dx_9_1,dx_9_2,dx_9_3,dx_9_4,dx_9_5,dx_ll1,dx_ll3]
    dae['x'] += [ x_wo1, x_wo2, x_wo3, x_lpf7, x_9_1, x_9_2, x_9_3, x_9_4, x_9_5, x_ll1, x_ll3]
    dae['g'] += [g_v_pss]
    dae['y_ini'] += [  v_pss]  
    dae['y_run'] += [  v_pss] 

    dae['params_dict'].update({str(T_1):pss_data['T_1']})
    dae['params_dict'].update({str(T_2):pss_data['T_2']})
    dae['params_dict'].update({str(T_3):pss_data['T_3']})
    dae['params_dict'].update({str(T_4):pss_data['T_4']})
    dae['params_dict'].update({str(T_7):pss_data['T_7']})
    dae['params_dict'].update({str(T_9):pss_data['T_9']})
    dae['params_dict'].update({str(T_wo1):pss_data['T_wo1']})
    dae['params_dict'].update({str(T_wo2):pss_data['T_wo2']})
    dae['params_dict'].update({str(T_wo3):pss_data['T_wo3']})
    dae['params_dict'].update({str(K_s1):pss_data['K_s1']})
    dae['params_dict'].update({str(K_s2):pss_data['K_s2']})
    dae['params_dict'].update({str(K_s3):pss_data['K_s3']})